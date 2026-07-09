#!/usr/bin/env python3
"""Student version of the realtime 1-second IMU segment classifier GUI.

This version intentionally uses only the resampled six-channel IMU segment as
features. 
The summary-statistics feature block is left as a TODO exercise.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import importlib.util
import math
import queue
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any


DEFAULT_CLASS_NAMES = ("Task 1", "Task 2", "Task 3")
DEFAULT_MODEL_PATH = Path("model_imu_student.joblib")
G_MPS2 = 9.80665
FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
SAMPLE_RE = re.compile(
    rf"accel\[g\]\s+x=\s*({FLOAT_RE})\s+y=\s*({FLOAT_RE})\s+z=\s*({FLOAT_RE})"
    rf"\s+\|\s+gyro\[dps\]\s+x=\s*({FLOAT_RE})\s+y=\s*({FLOAT_RE})\s+z=\s*({FLOAT_RE})"
)

CHANNEL_NAMES = ("ax", "ay", "az", "gx", "gy", "gz")
ACCEL_COLORS = ("#4C72B0", "#55A868", "#C44E52")
GYRO_COLORS = ("#8172B3", "#CCB974", "#64B5CD")


@dataclasses.dataclass
class RuntimeDeps:
    np: Any
    serial: Any
    joblib: Any
    make_pipeline: Any
    StandardScaler: Any
    SVC: Any


@dataclasses.dataclass
class DataInstance:
    label: str | None
    measurements: Any
    sample_count: int
    duration_s: float
    captured_at: float = dataclasses.field(default_factory=time.time)


def missing_dependencies() -> list[str]:
    modules = {
        "numpy": "numpy",
        "serial": "pyserial",
        "sklearn": "scikit-learn",
        "joblib": "joblib",
    }
    return [package for module, package in modules.items() if importlib.util.find_spec(module) is None]


def load_runtime_deps() -> RuntimeDeps:
    import joblib
    import numpy as np
    import serial
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    return RuntimeDeps(
        np=np,
        serial=serial,
        joblib=joblib,
        make_pipeline=make_pipeline,
        StandardScaler=StandardScaler,
        SVC=SVC,
    )


def parse_sample(line: str) -> tuple[float, float, float, float, float, float] | None:
    match = SAMPLE_RE.search(line)
    if not match:
        return None
    return tuple(float(value) for value in match.groups())


class SerialImuInput:
    def __init__(
        self,
        deps: RuntimeDeps,
        *,
        port: str | None,
        baud: int,
        demo_signal: bool,
        demo_sample_rate: float,
        print_unparsed: bool,
    ) -> None:
        self.np = deps.np
        self.serial = deps.serial
        self.port = port
        self.baud = baud
        self.demo_signal = demo_signal
        self.demo_sample_rate = demo_sample_rate
        self.print_unparsed = print_unparsed
        self._queue: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=2000)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._serial_port: Any | None = None
        self._demo_started_at = time.monotonic()
        self._demo_last_sample_at: float | None = None
        self._rng = self.np.random.default_rng()

    def start(self) -> None:
        if self.demo_signal:
            self._demo_last_sample_at = time.monotonic()
            return

        if not self.port:
            raise RuntimeError("A serial --port is required unless --demo-signal is used.")

        self._serial_port = self.serial.Serial(self.port, baudrate=self.baud, timeout=0.1)
        self._serial_port.reset_input_buffer()
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._serial_port is not None:
            self._serial_port.close()
            self._serial_port = None

    def latest_samples(self) -> tuple[list[tuple[float, float, float, float, float, float, float]], list[str]]:
        if self.demo_signal:
            return self._generate_demo_samples(), []

        samples = []
        errors = []
        while True:
            try:
                kind, value = self._queue.get_nowait()
            except queue.Empty:
                break

            if kind == "sample":
                samples.append(value)
            elif kind == "error":
                errors.append(str(value))
        return samples, errors

    def _reader_loop(self) -> None:
        try:
            assert self._serial_port is not None
            while not self._stop_event.is_set():
                raw = self._serial_port.readline()
                if not raw:
                    continue

                line = raw.decode("utf-8", errors="replace").strip()
                sample = parse_sample(line)
                if sample is None:
                    if self.print_unparsed:
                        print(line)
                    continue

                self._put_queue(("sample", (time.monotonic(), *sample)))
        except Exception as exc:
            self._put_queue(("error", str(exc)))

    def _put_queue(self, item: tuple[str, Any]) -> None:
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put_nowait(item)

    def _generate_demo_samples(self) -> list[tuple[float, float, float, float, float, float, float]]:
        now = time.monotonic()
        if self._demo_last_sample_at is None:
            self._demo_last_sample_at = now

        dt = 1.0 / self.demo_sample_rate
        samples = []
        while self._demo_last_sample_at + dt <= now:
            self._demo_last_sample_at += dt
            samples.append(self._synthetic_sample(self._demo_last_sample_at))
        return samples

    def _synthetic_sample(self, timestamp: float) -> tuple[float, float, float, float, float, float, float]:
        t = timestamp - self._demo_started_at
        demo_class = int(t // 8.0) % 3
        phase = 2.0 * math.pi * t
        noise_accel = 0.02 * self._rng.normal(size=3)
        noise_gyro = 1.0 * self._rng.normal(size=3)

        if demo_class == 0:
            accel = self.np.array([0.02 * math.sin(phase), 0.0, 1.0]) + noise_accel
            gyro = self.np.array([0.0, 0.0, 0.0]) + noise_gyro
        elif demo_class == 1:
            accel = self.np.array([0.65 * math.sin(3.0 * phase), 0.1 * math.cos(phase), 1.0]) + noise_accel
            gyro = self.np.array([0.0, 80.0 * math.sin(3.0 * phase), 0.0]) + noise_gyro
        else:
            accel = self.np.array([0.35 * math.cos(1.4 * phase), 0.35 * math.sin(1.4 * phase), 1.0]) + noise_accel
            gyro = self.np.array([30.0 * math.sin(phase), 0.0, 95.0 * math.cos(1.4 * phase)]) + noise_gyro

        return (
            timestamp,
            float(accel[0]),
            float(accel[1]),
            float(accel[2]),
            float(gyro[0]),
            float(gyro[1]),
            float(gyro[2]),
        )


class ImuSegmentFeatureExtractor:
    def __init__(
        self,
        deps: RuntimeDeps,
        *,
        segment_seconds: float,
        target_samples: int,
        min_segment_samples: int,
        min_segment_coverage: float,
    ) -> None:
        self.np = deps.np
        self.segment_seconds = segment_seconds
        self.target_samples = target_samples
        self.min_segment_samples = min_segment_samples
        self.min_segment_coverage = min_segment_coverage
        self.target_time = self.np.linspace(0.0, segment_seconds, target_samples, dtype="float32")

    def capture_instance(
        self,
        label: str | None,
        samples: list[tuple[float, float, float, float, float, float, float]],
    ) -> DataInstance:
        if len(samples) < self.min_segment_samples:
            raise RuntimeError(
                f"Need at least {self.min_segment_samples} IMU samples; got {len(samples)}."
            )

        data = self.np.asarray(samples, dtype="float32")
        times = data[:, 0]
        values = data[:, 1:7]
        duration_s = float(times[-1] - times[0])
        if duration_s <= 0.0:
            raise RuntimeError("Segment duration is zero.")
        if duration_s < self.segment_seconds * self.min_segment_coverage:
            raise RuntimeError(
                f"Need about {self.segment_seconds:.1f} seconds of data; got {duration_s:.2f} seconds."
            )

        rel_time = times - times[0]
        resampled = self.np.column_stack(
            [
                self.np.interp(self.target_time, rel_time, values[:, channel])
                for channel in range(values.shape[1])
            ]
        ).astype("float32")

        means = resampled.mean(axis=0)
        centered = resampled - means

        # STUDENT TODO:
        # Add summary statistics as extra features after the raw segment shape.
        # Good first features to try:
        # - standard deviation of each channel
        # - min and max of each channel
        # - range = max - min for each channel
        # - RMS energy of each channel
        # - mean absolute first difference of each channel
        #
        # When implemented, concatenate your new statistics to the flattened
        # segment below, for example:
        # summary = self.np.concatenate([...]).astype("float32")
        # measurements = self.np.concatenate([centered.T.reshape(-1), summary]).astype("float32")
        measurements = centered.T.reshape(-1).astype("float32")

        return DataInstance(
            label=label,
            measurements=measurements,
            sample_count=len(samples),
            duration_s=duration_s,
        )


class MLClassifier:
    def __init__(self, deps: RuntimeDeps, class_names: tuple[str, ...], feature_config: dict[str, Any]) -> None:
        self.deps = deps
        self.class_names = tuple(class_names)
        self.feature_config = dict(feature_config)
        self.pipeline: Any | None = None

    def train(self, instances_by_label: dict[str, list[DataInstance]]) -> None:
        x_values = []
        y_values = []
        for label in self.class_names:
            for instance in instances_by_label[label]:
                x_values.append(instance.measurements)
                y_values.append(label)

        if len(set(y_values)) != len(self.class_names):
            raise RuntimeError("Training needs examples from all three classes.")

        classifier = self.deps.make_pipeline(
            self.deps.StandardScaler(),
            self.deps.SVC(C=1.0, kernel="linear"),
        )
        classifier.fit(self.deps.np.vstack(x_values), self.deps.np.asarray(y_values))
        self.pipeline = classifier

    def classify(self, instance: DataInstance) -> str:
        if self.pipeline is None:
            return "Unknown"
        return str(self.pipeline.predict(instance.measurements.reshape(1, -1))[0])

    def save(self, path: Path) -> None:
        if self.pipeline is None:
            raise RuntimeError("No trained classifier to save.")
        self.deps.joblib.dump(
            {
                "pipeline": self.pipeline,
                "class_names": self.class_names,
                "feature_config": self.feature_config,
                "saved_at": time.time(),
            },
            path,
        )

    @classmethod
    def load(cls, deps: RuntimeDeps, path: Path) -> "MLClassifier":
        payload = deps.joblib.load(path)
        classifier = cls(
            deps,
            tuple(payload.get("class_names", DEFAULT_CLASS_NAMES)),
            dict(payload.get("feature_config", {})),
        )
        classifier.pipeline = payload["pipeline"]
        return classifier


class ImuSegmentClassifierApp:
    width = 900
    height = 560

    def __init__(self, deps: RuntimeDeps, args: argparse.Namespace) -> None:
        import tkinter as tk
        from tkinter import messagebox, ttk

        self.deps = deps
        self.np = deps.np
        self.args = args
        self.tk = tk
        self.messagebox = messagebox
        self.class_names = tuple(args.classes)
        self.model_path = Path(args.model)
        self.training_data = {label: [] for label in self.class_names}
        self.class_index = 0
        self.classifier: MLClassifier | None = None
        self.classification_window: collections.deque[str] = collections.deque(maxlen=args.vote_window)
        self.vote_threshold = args.vote_threshold
        self.stable_classification: str | None = None
        self.last_classified_at = 0.0
        self.detection_log: set[str] = set()
        self.log_captured = False
        self.status_text = "Starting"
        self.total_samples = 0
        self.latest_sample: tuple[float, float, float, float, float, float, float] | None = None
        self.history: collections.deque[tuple[float, float, float, float, float, float, float]] = collections.deque()
        self.recording = False
        self.record_request_time: float | None = None
        self.recording_label: str | None = None
        self.recorded_samples: list[tuple[float, float, float, float, float, float, float]] = []

        self.imu_input = SerialImuInput(
            deps,
            port=args.port,
            baud=args.baud,
            demo_signal=args.demo_signal,
            demo_sample_rate=args.demo_sample_rate,
            print_unparsed=args.print_unparsed,
        )
        self.features = ImuSegmentFeatureExtractor(
            deps,
            segment_seconds=args.segment_seconds,
            target_samples=args.target_samples,
            min_segment_samples=args.min_segment_samples,
            min_segment_coverage=args.min_segment_coverage,
        )

        self.root = tk.Tk()
        self.root.title(f"Student {self.args.segment_seconds:g}-Second IMU Segment Classifier")
        self.root.resizable(False, False)

        self.canvas = tk.Canvas(
            self.root,
            width=self.width,
            height=self.height,
            bg="#F5F7FB",
            highlightthickness=0,
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.overlay_bg = self.canvas.create_rectangle(0, 0, self.width, 98, fill="#FFFFFF", outline="")
        self.primary_text = self.canvas.create_text(
            22,
            32,
            anchor="w",
            fill="#111827",
            font=("Helvetica", 28, "bold"),
            text=self.class_names[0],
        )
        self.secondary_text = self.canvas.create_text(
            22,
            68,
            anchor="w",
            fill="#475569",
            font=("Helvetica", 15),
            text="Waiting for IMU samples",
        )
        self.progress_bg = self.canvas.create_rectangle(
            22,
            86,
            self.width - 22,
            92,
            fill="#E2E8F0",
            outline="",
            state="hidden",
        )
        self.progress_fg = self.canvas.create_rectangle(
            22,
            86,
            22,
            92,
            fill="#2563EB",
            outline="",
            state="hidden",
        )

        controls = ttk.Frame(self.root, padding=(8, 8))
        controls.grid(row=1, column=0, sticky="ew")
        controls.columnconfigure(1, weight=1)

        self.selected_class = tk.StringVar(value=self.class_names[self.class_index])
        self.class_select = ttk.Combobox(
            controls,
            textvariable=self.selected_class,
            values=self.class_names,
            width=14,
            state="readonly",
        )
        self.class_select.grid(row=0, column=0, padx=(0, 8))
        self.class_select.bind("<<ComboboxSelected>>", self._class_selected)

        self.status_label = ttk.Label(controls, text="", width=46, anchor="w")
        self.status_label.grid(row=0, column=1, sticky="ew")

        ttk.Button(controls, text=f"Capture {self.args.segment_seconds:g} s", command=self.start_capture).grid(row=0, column=2, padx=3)
        self.train_button = ttk.Button(controls, text="Train", command=self.toggle_classifier)
        self.train_button.grid(row=0, column=3, padx=3)
        ttk.Button(controls, text="Save", command=self.save_model).grid(row=0, column=4, padx=3)
        ttk.Button(controls, text="Load", command=self.load_model).grid(row=0, column=5, padx=3)
        self.log_button = ttk.Button(controls, text="Logger", command=self.toggle_logger)
        self.log_button.grid(row=0, column=6, padx=(3, 0))

        self.root.bind("<KeyPress>", self._key_pressed)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def run(self) -> None:
        try:
            self.imu_input.start()
        except Exception as exc:
            self.messagebox.showerror("IMU input failed", str(exc))
            self.root.destroy()
            return

        if self.args.demo_signal:
            self._set_status("Demo signal ready")
        else:
            self._set_status(f"Serial ready: {self.args.port}")
        self._tick()
        self.root.mainloop()

    def close(self) -> None:
        self.imu_input.close()
        self.root.destroy()

    def start_capture(self) -> None:
        if self.recording:
            return

        if self.classifier is not None:
            self.classifier = None
            self.classification_window.clear()
            self.stable_classification = None
            self.train_button.configure(text="Train")

        self.recording = True
        self.record_request_time = time.monotonic()
        self.recording_label = self.class_names[self.class_index]
        self.recorded_samples = []
        self._set_status(f"Recording {self.args.segment_seconds:.1f} s for {self.recording_label}")

    def toggle_classifier(self) -> None:
        if self.classifier is not None:
            self.classifier = None
            self.classification_window.clear()
            self.stable_classification = None
            self.train_button.configure(text="Train")
            self._set_status("Collection mode")
            return

        missing = [
            label
            for label in self.class_names
            if len(self.training_data[label]) < self.args.min_samples_per_class
        ]
        if missing:
            self._set_status(
                "Need "
                f"{self.args.min_samples_per_class}+ examples for: "
                + ", ".join(missing)
            )
            return

        classifier = MLClassifier(self.deps, self.class_names, self._feature_config())
        try:
            classifier.train(self.training_data)
        except Exception as exc:
            self._set_status(str(exc))
            return

        self.classifier = classifier
        self.classification_window.clear()
        self.stable_classification = None
        self.train_button.configure(text="Collect")
        self._set_status("Training done")

    def save_model(self) -> None:
        if self.classifier is None:
            self._set_status("Train before saving.")
            return
        try:
            self.classifier.save(self.model_path)
        except Exception as exc:
            self._set_status(f"Save failed: {exc}")
            return
        self._set_status(f"Saved {self.model_path}")

    def load_model(self) -> None:
        if not self.model_path.exists():
            self._set_status(f"Missing {self.model_path}")
            return
        try:
            classifier = MLClassifier.load(self.deps, self.model_path)
        except Exception as exc:
            self._set_status(f"Load failed: {exc}")
            return

        if classifier.class_names != self.class_names:
            self._set_status(f"Model classes are {classifier.class_names}, not {self.class_names}")
            return
        if classifier.feature_config != self._feature_config():
            self._set_status("Model feature settings do not match current arguments.")
            return

        self.classifier = classifier
        self.classification_window.clear()
        self.stable_classification = None
        self.train_button.configure(text="Collect")
        self._set_status(f"Loaded {self.model_path}")

    def toggle_logger(self) -> None:
        if self.log_captured:
            print(f"Detection log: {sorted(self.detection_log)}")
            self.detection_log.clear()
            self.log_captured = False
            self._set_status("Logger cleared")
        else:
            self.log_captured = True
            self._set_status("Logger on")

    def next_class(self) -> None:
        if self.recording:
            return
        self.class_index = (self.class_index + 1) % len(self.class_names)
        self.selected_class.set(self.class_names[self.class_index])
        self._set_status(f"Class: {self.class_names[self.class_index]}")

    def _tick(self) -> None:
        samples, errors = self.imu_input.latest_samples()
        for error in errors:
            self._set_status(f"Serial error: {error}")
            print(f"Serial error: {error}", file=sys.stderr)

        for sample in samples:
            self._handle_sample(sample)

        self._classify_latest_window()
        self._redraw()
        self.root.after(max(1, int(1000 / self.args.fps)), self._tick)

    def _handle_sample(self, sample: tuple[float, float, float, float, float, float, float]) -> None:
        timestamp = sample[0]
        self.latest_sample = sample
        self.total_samples += 1
        self.history.append(sample)
        while self.history and timestamp - self.history[0][0] > self.args.max_history_seconds:
            self.history.popleft()

        if self.recording and self.record_request_time is not None and timestamp >= self.record_request_time:
            self.recorded_samples.append(sample)
            if timestamp - self.record_request_time >= self.args.segment_seconds:
                self._finish_capture()

    def _finish_capture(self) -> None:
        label = self.recording_label
        samples = self.recorded_samples
        self.recording = False
        self.record_request_time = None
        self.recording_label = None

        if label is None:
            self._set_status("Capture failed: missing label.")
            return

        try:
            instance = self.features.capture_instance(label, samples)
        except RuntimeError as exc:
            self._set_status(f"Capture failed: {exc}")
            return

        self.training_data[label].append(instance)
        self._set_status(
            f"Captured {label}: {len(self.training_data[label])} examples "
            f"({instance.sample_count} samples, {instance.duration_s:.2f} s)"
        )

    def _classify_latest_window(self) -> None:
        if self.classifier is None or self.recording:
            return
        now = time.monotonic()
        if now - self.last_classified_at < self.args.classification_interval:
            return
        self.last_classified_at = now

        try:
            instance = self.features.capture_instance(None, self._latest_window_samples())
        except RuntimeError:
            return

        guessed_label = self.classifier.classify(instance)
        self.classification_window.append(guessed_label)
        counts = collections.Counter(self.classification_window)
        majority_label = next(
            (
                label
                for label in self.class_names
                if counts[label] >= self.vote_threshold
            ),
            None,
        )
        if majority_label is not None and majority_label != self.stable_classification:
            self.stable_classification = majority_label
            print(f"Classification updated to: {self.stable_classification}")

        if self.log_captured and self.stable_classification is not None:
            self.detection_log.add(self.stable_classification)

    def _latest_window_samples(self) -> list[tuple[float, float, float, float, float, float, float]]:
        if not self.history:
            return []
        end_time = self.history[-1][0]
        start_time = end_time - self.args.segment_seconds
        return [sample for sample in self.history if sample[0] >= start_time]

    def _redraw(self) -> None:
        self._update_overlay_text()
        window_samples = self._latest_window_samples()
        self.canvas.delete("plot")
        self._draw_panel(
            window_samples,
            title="Accelerometer [g]",
            y0=118,
            y1=318,
            channel_slice=slice(1, 4),
            colors=ACCEL_COLORS,
            y_abs=self.args.accel_plot_range,
        )
        self._draw_panel(
            window_samples,
            title="Gyroscope [deg/s]",
            y0=338,
            y1=538,
            channel_slice=slice(4, 7),
            colors=GYRO_COLORS,
            y_abs=self.args.gyro_plot_range,
        )

    def _draw_panel(
        self,
        samples: list[tuple[float, float, float, float, float, float, float]],
        *,
        title: str,
        y0: int,
        y1: int,
        channel_slice: slice,
        colors: tuple[str, str, str],
        y_abs: float,
    ) -> None:
        x0 = 58
        x1 = self.width - 24
        mid_y = (y0 + y1) / 2.0
        half_h = (y1 - y0) / 2.0 - 24
        self.canvas.create_rectangle(x0, y0, x1, y1, fill="#FFFFFF", outline="#CBD5E1", tags="plot")
        self.canvas.create_text(x0 + 12, y0 + 18, anchor="w", fill="#334155", font=("Helvetica", 14, "bold"), text=title, tags="plot")

        for frac in (0.25, 0.5, 0.75):
            y = y0 + (y1 - y0) * frac
            self.canvas.create_line(x0, y, x1, y, fill="#E2E8F0", tags="plot")
        for sec in range(1, int(self.args.segment_seconds) + 1):
            x = x0 + (sec / self.args.segment_seconds) * (x1 - x0)
            self.canvas.create_line(x, y0 + 32, x, y1 - 14, fill="#EEF2F7", tags="plot")

        self.canvas.create_line(x0, mid_y, x1, mid_y, fill="#CBD5E1", tags="plot")
        self.canvas.create_text(x0 - 8, y0 + 38, anchor="e", fill="#64748B", font=("Helvetica", 10), text=f"+{y_abs:g}", tags="plot")
        self.canvas.create_text(x0 - 8, mid_y, anchor="e", fill="#64748B", font=("Helvetica", 10), text="0", tags="plot")
        self.canvas.create_text(x0 - 8, y1 - 18, anchor="e", fill="#64748B", font=("Helvetica", 10), text=f"-{y_abs:g}", tags="plot")

        suffix = ("x", "y", "z")
        legend_x = x1 - 156
        for i, axis in enumerate(suffix):
            self.canvas.create_line(legend_x + 52 * i, y0 + 18, legend_x + 52 * i + 18, y0 + 18, fill=colors[i], width=3, tags="plot")
            self.canvas.create_text(legend_x + 52 * i + 24, y0 + 18, anchor="w", fill="#475569", font=("Helvetica", 10), text=axis, tags="plot")

        if len(samples) < 2:
            self.canvas.create_text(
                (x0 + x1) / 2,
                mid_y,
                fill="#94A3B8",
                font=("Helvetica", 13),
                text="Waiting for enough IMU samples",
                tags="plot",
            )
            return

        data = self.np.asarray(samples, dtype="float32")
        rel_time = data[:, 0] - data[-1, 0] + self.args.segment_seconds
        rel_time = self.np.clip(rel_time, 0.0, self.args.segment_seconds)
        x_values = x0 + (rel_time / self.args.segment_seconds) * (x1 - x0)
        values = data[:, channel_slice]

        for channel in range(values.shape[1]):
            clipped = self.np.clip(values[:, channel], -y_abs, y_abs)
            y_values = mid_y - (clipped / y_abs) * half_h
            coords = []
            for x, y in zip(x_values, y_values):
                coords.extend((float(x), float(y)))
            if len(coords) >= 4:
                self.canvas.create_line(
                    *coords,
                    fill=colors[channel],
                    width=2,
                    smooth=True,
                    tags="plot",
                )

    def _update_overlay_text(self) -> None:
        counts_text = " | ".join(f"{label}: {len(self.training_data[label])}" for label in self.class_names)

        if self.recording and self.record_request_time is not None and self.recording_label is not None:
            elapsed = min(self.args.segment_seconds, time.monotonic() - self.record_request_time)
            progress = elapsed / self.args.segment_seconds
            self.canvas.itemconfigure(self.primary_text, text=f"Recording {self.recording_label}")
            self.canvas.itemconfigure(
                self.secondary_text,
                text=f"{elapsed:.2f}/{self.args.segment_seconds:.2f} s | {len(self.recorded_samples)} samples",
            )
            self.canvas.itemconfigure(self.progress_bg, state="normal")
            self.canvas.itemconfigure(self.progress_fg, state="normal")
            self.canvas.coords(self.progress_fg, 22, 86, 22 + (self.width - 44) * progress, 92)
            return

        self.canvas.itemconfigure(self.progress_bg, state="hidden")
        self.canvas.itemconfigure(self.progress_fg, state="hidden")

        if self.classifier is not None:
            if self.stable_classification is None:
                primary = "Classifying..."
            else:
                primary = f"Classified: {self.stable_classification}"
            if self.log_captured:
                secondary = f"Logger: {sorted(self.detection_log)} | {self.status_text}"
            else:
                secondary = self.status_text
            self.canvas.itemconfigure(self.primary_text, text=primary)
            self.canvas.itemconfigure(self.secondary_text, text=secondary)
            return

        self.canvas.itemconfigure(self.primary_text, text=self.class_names[self.class_index])
        if self.latest_sample is None:
            secondary = f"{self.status_text} | {counts_text}"
        else:
            _, ax, ay, az, gx, gy, gz = self.latest_sample
            secondary = (
                f"{counts_text} | samples {self.total_samples} | "
                f"a=({ax:.2f},{ay:.2f},{az:.2f}) g | "
                f"g=({gx:.1f},{gy:.1f},{gz:.1f}) deg/s"
            )
        self.canvas.itemconfigure(self.secondary_text, text=secondary)

    def _class_selected(self, _event: Any) -> None:
        if self.recording:
            self.selected_class.set(self.class_names[self.class_index])
            return
        self.class_index = self.class_names.index(self.selected_class.get())
        self._set_status(f"Class: {self.class_names[self.class_index]}")

    def _key_pressed(self, event: Any) -> None:
        if event.keysym == "Down":
            self.next_class()
            return

        key = event.char.lower() if event.char else ""
        if key == "t":
            self.toggle_classifier()
        elif key == "s":
            self.save_model()
        elif key == "l":
            self.load_model()
        elif key == "b":
            self.toggle_logger()
        elif key in {"c", " "} or event.keysym == "Return":
            self.start_capture()

    def _feature_config(self) -> dict[str, Any]:
        return {
            "segment_seconds": float(self.args.segment_seconds),
            "target_samples": int(self.args.target_samples),
            "min_segment_samples": int(self.args.min_segment_samples),
            "min_segment_coverage": float(self.args.min_segment_coverage),
            "feature_version": "student_no_stats_v1",
        }

    def _set_status(self, value: str) -> None:
        self.status_text = value
        self.status_label.configure(text=value)


def parse_classes(value: str) -> tuple[str, str, str]:
    names = tuple(part.strip() for part in value.split(",") if part.strip())
    if len(names) != 3:
        raise argparse.ArgumentTypeError("Provide exactly three comma-separated class names.")
    return names  # type: ignore[return-value]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Student realtime 1-second IMU segment classifier GUI.")
    parser.add_argument("--port", default=None, help="Serial port, e.g. /dev/cu.usbserial-5B1F0080901")
    parser.add_argument("--baud", type=int, default=115200, help="Serial baud rate")
    parser.add_argument("--classes", type=parse_classes, default=DEFAULT_CLASS_NAMES, help="Exactly 3 comma-separated class names")
    parser.add_argument("--segment-seconds", type=float, default=1.0, help="Seconds per IMU segment")
    parser.add_argument("--target-samples", type=int, default=20, help="Resampled samples per segment")
    parser.add_argument("--min-segment-samples", type=int, default=10, help="Minimum serial samples needed for a valid segment")
    parser.add_argument("--min-segment-coverage", type=float, default=0.8, help="Minimum fraction of segment duration required")
    parser.add_argument("--min-samples-per-class", type=int, default=2, help="Training examples required for each class")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--vote-window", type=int, default=7)
    parser.add_argument("--vote-threshold", type=int, default=5)
    parser.add_argument("--classification-interval", type=float, default=0.25, help="Seconds between live classifications")
    parser.add_argument("--fps", type=float, default=30.0, help="GUI redraw rate")
    parser.add_argument("--max-history-seconds", type=float, default=20.0, help="Seconds of IMU history kept in memory")
    parser.add_argument("--accel-plot-range", type=float, default=2.5, help="Accelerometer y-range in +/- g")
    parser.add_argument("--gyro-plot-range", type=float, default=250.0, help="Gyroscope y-range in +/- deg/s")
    parser.add_argument("--demo-signal", action="store_true", help="Use generated IMU-like data instead of serial input")
    parser.add_argument("--demo-sample-rate", type=float, default=20.0, help="Generated sample rate for --demo-signal")
    parser.add_argument("--print-unparsed", action="store_true", help="Print serial lines that are not IMU samples")
    args = parser.parse_args(argv)

    if not args.demo_signal and not args.port:
        parser.error("--port is required unless --demo-signal is used")
    if args.segment_seconds <= 0.0:
        parser.error("--segment-seconds must be positive")
    if args.target_samples < 2:
        parser.error("--target-samples must be at least 2")
    if args.min_segment_samples < 2:
        parser.error("--min-segment-samples must be at least 2")
    if not 0.0 < args.min_segment_coverage <= 1.0:
        parser.error("--min-segment-coverage must be in (0, 1]")
    if args.min_samples_per_class < 1:
        parser.error("--min-samples-per-class must be at least 1")
    if args.vote_window < 1:
        parser.error("--vote-window must be at least 1")
    if not 1 <= args.vote_threshold <= args.vote_window:
        parser.error("--vote-threshold must be between 1 and --vote-window")
    if args.classification_interval <= 0.0:
        parser.error("--classification-interval must be positive")
    if args.fps <= 0.0:
        parser.error("--fps must be positive")
    if args.max_history_seconds < args.segment_seconds:
        parser.error("--max-history-seconds must be at least --segment-seconds")
    if args.accel_plot_range <= 0.0 or args.gyro_plot_range <= 0.0:
        parser.error("Plot ranges must be positive")
    if args.demo_sample_rate <= 0.0:
        parser.error("--demo-sample-rate must be positive")

    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    missing = missing_dependencies()
    if missing:
        packages = " ".join(dict.fromkeys(missing))
        print("Missing Python dependencies:", packages, file=sys.stderr)
        print("Install them with: python3 -m pip install -r requirements-python.txt", file=sys.stderr)
        return 2

    deps = load_runtime_deps()
    app = ImuSegmentClassifierApp(deps, args)
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
