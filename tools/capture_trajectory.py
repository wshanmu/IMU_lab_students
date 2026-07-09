#!/usr/bin/env python3
"""Version 1 student script: capture IMU data and plot one raw trajectory."""

import argparse
import queue
import re
import signal
import sys
import threading
import time
from collections import deque

import numpy as np
import pyqtgraph as pg
import serial
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets


G_MPS2 = 9.80665
FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
SAMPLE_RE = re.compile(
    rf"accel\[g\]\s+x=\s*({FLOAT_RE})\s+y=\s*({FLOAT_RE})\s+z=\s*({FLOAT_RE})"
    rf"\s+\|\s+gyro\[dps\]\s+x=\s*({FLOAT_RE})\s+y=\s*({FLOAT_RE})\s+z=\s*({FLOAT_RE})"
)

AXES = ("x", "y", "z")
COLORS = {
    "x": "#4C72B0",
    "y": "#55A868",
    "z": "#C44E52",
}
PATH_START_COLOR = np.array([76, 114, 176], dtype=float)
PATH_END_COLOR = np.array([221, 132, 82], dtype=float)


def parse_sample(line):
    match = SAMPLE_RE.search(line)
    if not match:
        return None
    return tuple(float(value) for value in match.groups())


def interpolate_color(start_rgb, end_rgb, fraction):
    fraction = float(np.clip(fraction, 0.0, 1.0))
    rgb = np.rint(start_rgb + (end_rgb - start_rgb) * fraction).astype(int)
    return tuple(int(value) for value in rgb)


def configure_curve(curve):
    if hasattr(curve, "setClipToView"):
        curve.setClipToView(True)
    if hasattr(curve, "setDownsampling"):
        curve.setDownsampling(auto=True, method="peak")


def downsample_path(points, max_segments):
    max_points = max_segments + 1
    if len(points) <= max_points:
        return points

    indices = np.linspace(0, len(points) - 1, max_points)
    indices = np.unique(np.rint(indices).astype(int))
    return points[indices]


def serial_worker(port, baud, samples, stop_event, print_unparsed):
    try:
        with serial.Serial(port, baudrate=baud, timeout=0.1) as ser:
            ser.reset_input_buffer()
            while not stop_event.is_set():
                raw = ser.readline()
                if not raw:
                    continue

                line = raw.decode("utf-8", errors="replace").strip()
                sample = parse_sample(line)
                if sample is not None:
                    samples.put((time.monotonic(), *sample))
                elif print_unparsed:
                    print(line)
    except serial.SerialException as exc:
        samples.put(("error", str(exc)))


def normalized(vector):
    norm = np.linalg.norm(vector)
    if norm < 1e-12:
        return vector
    return vector / norm


def quat_normalized(q):
    return normalized(q)


def quat_multiply(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=float,
    )


def quat_from_two_vectors(source, target):
    source = normalized(source)
    target = normalized(target)
    dot = float(np.dot(source, target))

    if dot > 0.999999:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    if dot < -0.999999:
        axis = np.cross(source, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(source, np.array([0.0, 1.0, 0.0]))
        axis = normalized(axis)
        return np.array([0.0, axis[0], axis[1], axis[2]], dtype=float)

    axis = np.cross(source, target)
    return quat_normalized(np.array([1.0 + dot, axis[0], axis[1], axis[2]], dtype=float))


def quat_from_angular_velocity(omega_rad_s, dt):
    angle = float(np.linalg.norm(omega_rad_s) * dt)
    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    axis = omega_rad_s / np.linalg.norm(omega_rad_s)
    half_angle = 0.5 * angle
    return np.array([np.cos(half_angle), *(axis * np.sin(half_angle))], dtype=float)


def quat_rotate(q, vector):
    q_vec = q[1:]
    uv = np.cross(q_vec, vector)
    uuv = np.cross(q_vec, uv)
    return vector + 2.0 * (q[0] * uv + uuv)


def samples_to_arrays(samples):
    data = np.asarray(samples, dtype=float)
    if data.size == 0:
        return None
    return {
        "time": data[:, 0],
        "accel_g": data[:, 1:4],
        "gyro_dps": data[:, 4:7],
    }


def compute_trajectory(capture_samples, baseline_samples, args):
    capture = samples_to_arrays(capture_samples)
    if capture is None or len(capture["time"]) < 3:
        raise ValueError("Not enough captured samples to compute trajectory.")

    baseline = samples_to_arrays(baseline_samples)
    if baseline is None or len(baseline["time"]) < args.min_baseline_samples:
        fallback_count = min(max(args.min_baseline_samples, 8), len(capture["time"]))
        baseline_accel_g = capture["accel_g"][:fallback_count]
        baseline_gyro_dps = capture["gyro_dps"][:fallback_count]
        baseline_warning = f"baseline used first {fallback_count} captured samples"
    else:
        baseline_accel_g = baseline["accel_g"]
        baseline_gyro_dps = baseline["gyro_dps"]
        baseline_warning = None

    accel_mean_g = np.mean(baseline_accel_g, axis=0)
    gyro_bias_dps = np.mean(baseline_gyro_dps, axis=0)
    gravity_norm_g = float(np.linalg.norm(accel_mean_g))
    if gravity_norm_g < 0.2:
        raise ValueError("Calibration failed: accelerometer gravity magnitude is too small.")

    world_specific_force_g = np.array([0.0, 0.0, gravity_norm_g])
    q_body_to_world = quat_from_two_vectors(accel_mean_g, world_specific_force_g)

    times = capture["time"] - capture["time"][0]
    accel_g = capture["accel_g"]
    gyro_dps = capture["gyro_dps"]

    position_m = np.zeros((len(times), 3), dtype=float)
    velocity_mps = np.zeros(3, dtype=float)
    linear_accel_mps2 = np.zeros((len(times), 3), dtype=float)

    for i in range(1, len(times)):
        dt = float(times[i] - times[i - 1])
        if dt <= 0.0 or dt > args.max_dt:
            continue

        gyro_corrected_dps = gyro_dps[i] - gyro_bias_dps
        delta_q = quat_from_angular_velocity(np.deg2rad(gyro_corrected_dps), dt)
        q_body_to_world = quat_normalized(quat_multiply(q_body_to_world, delta_q))

        specific_force_world_g = quat_rotate(q_body_to_world, accel_g[i])
        linear_accel = (specific_force_world_g - world_specific_force_g) * G_MPS2

        if np.linalg.norm(linear_accel) < args.accel_deadband:
            linear_accel[:] = 0.0

        stationary = (
            np.linalg.norm(linear_accel) < args.stationary_accel_threshold
            and np.linalg.norm(gyro_corrected_dps) < args.stationary_gyro_threshold
        )

        old_velocity = velocity_mps.copy()
        if stationary:
            velocity_mps[:] = 0.0
            linear_accel[:] = 0.0
        else:
            velocity_mps += linear_accel * dt
            if args.velocity_damping > 0.0:
                velocity_mps *= np.exp(-args.velocity_damping * dt)

        position_m[i] = position_m[i - 1] + 0.5 * (old_velocity + velocity_mps) * dt
        linear_accel_mps2[i] = linear_accel

    return {
        "time": times,
        "accel_g": accel_g,
        "gyro_dps": gyro_dps,
        "position_m": position_m,
        "linear_accel_mps2": linear_accel_mps2,
        "baseline_warning": baseline_warning,
        "gyro_bias_dps": gyro_bias_dps,
        "gravity_norm_g": gravity_norm_g,
    }


class CaptureWindow(pg.GraphicsLayoutWidget):
    def __init__(self, capture_callback):
        super().__init__(show=True, title="ESP32 BMI270 Capture Trajectory V1")
        self.capture_callback = capture_callback
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setFocus()

    def keyPressEvent(self, event):
        if event.text().lower() == "c":
            self.capture_callback()
            event.accept()
            return
        super().keyPressEvent(event)


class CaptureTrajectoryGui:
    def __init__(self, args, samples):
        self.args = args
        self.samples = samples
        self.total_samples = 0
        self.latest_sample = None
        self.prebuffer = deque()
        self.recording = False
        self.record_request_time = None
        self.recorded_samples = []
        self.baseline_samples = []
        self.last_live_plot_time = 0.0

        pg.setConfigOptions(antialias=True, background="#F5F7FB", foreground="#273142")

        self.window = CaptureWindow(self.start_capture)
        self.window.resize(1220, 850)
        self.window.setBackground("#F5F7FB")
        self.capture_shortcut = QtGui.QShortcut(QtGui.QKeySequence("c"), self.window)
        self.capture_shortcut.activated.connect(self.start_capture)

        self.window.addLabel(
            "<span style='font-size:22px; font-weight:700; color:#1F2937;'>BMI270 Trajectory Capture V1</span>",
            row=0,
            col=0,
            colspan=2,
        )
        self.status_label = self.window.addLabel(
            "<span style='color:#475569;'>Keep the IMU still, then press <b>c</b> to capture.</span>",
            row=1,
            col=0,
            colspan=2,
        )

        self.trajectory_plot = self.window.addPlot(row=2, col=0, rowspan=2, title="Reconstructed XY Trajectory")
        self.accel_plot = self.window.addPlot(row=2, col=1, title="Captured Accelerometer")
        self.gyro_plot = self.window.addPlot(row=3, col=1, title="Captured Gyroscope")

        self._setup_trajectory_plot()
        self._setup_time_plot(self.accel_plot, "Accelerometer", "Acceleration", "g")
        self._setup_time_plot(self.gyro_plot, "Gyroscope", "Angular Velocity", "deg/s")
        self.gyro_plot.setLabel("bottom", "Capture Time", units="s")
        self.accel_plot.setXLink(self.gyro_plot)

        self.trajectory_segments = []
        self.trajectory_legend_curve = self.trajectory_plot.plot(
            [],
            [],
            pen=pg.mkPen("#4C72B0", width=3.0),
            name="path early",
        )
        self.trajectory_legend_end_curve = self.trajectory_plot.plot(
            [],
            [],
            pen=pg.mkPen("#DD8452", width=3.0),
            name="path late",
        )
        self.start_marker = self.trajectory_plot.plot(
            pen=None,
            symbol="o",
            symbolBrush="#55A868",
            symbolPen="#55A868",
            symbolSize=11,
            name="start",
        )
        self.end_marker = self.trajectory_plot.plot(
            pen=None,
            symbol="o",
            symbolBrush="#C44E52",
            symbolPen="#C44E52",
            symbolSize=11,
            name="end",
        )

        self.accel_curves = {
            axis: self.accel_plot.plot(pen=pg.mkPen(COLORS[axis], width=2.2), name=f"a{axis}")
            for axis in AXES
        }
        self.gyro_curves = {
            axis: self.gyro_plot.plot(pen=pg.mkPen(COLORS[axis], width=2.2), name=f"g{axis}")
            for axis in AXES
        }

        for curve in (
            self.trajectory_legend_curve,
            self.trajectory_legend_end_curve,
            *self.accel_curves.values(),
            *self.gyro_curves.values(),
        ):
            configure_curve(curve)

    def _setup_time_plot(self, plot, title, left_label, units):
        plot.setTitle(f"<span style='font-size:15px; color:#273142;'>{title}</span>")
        plot.setLabel("left", left_label, units=units, color="#334155")
        plot.showGrid(x=True, y=True, alpha=0.22)
        plot.setMenuEnabled(False)
        plot.hideButtons()
        plot.enableAutoRange(x=True, y=True)
        self._style_plot(plot)
        self._add_legend(plot)

    def _setup_trajectory_plot(self):
        self.trajectory_plot.setTitle("<span style='font-size:15px; color:#273142;'>Reconstructed XY Trajectory</span>")
        self.trajectory_plot.setLabel("left", "Y", units="m", color="#334155")
        self.trajectory_plot.setLabel("bottom", "X", units="m", color="#334155")
        self.trajectory_plot.showGrid(x=True, y=True, alpha=0.22)
        self.trajectory_plot.setAspectLocked(True)
        self.trajectory_plot.setMenuEnabled(False)
        self.trajectory_plot.hideButtons()
        self._style_plot(self.trajectory_plot)
        self._add_legend(self.trajectory_plot)

    def _style_plot(self, plot):
        view_box = plot.getViewBox()
        view_box.setBackgroundColor("#FFFFFF")
        view_box.setBorder(pg.mkPen("#D5DAE3", width=1))

        for axis_name in ("left", "bottom"):
            axis = plot.getAxis(axis_name)
            axis.setPen(pg.mkPen("#CBD5E1", width=1))
            axis.setTextPen(pg.mkPen("#475569"))
            axis.setStyle(tickFont=None, autoExpandTextSpace=True)

    def _add_legend(self, plot):
        legend = plot.addLegend(offset=(-12, 12))
        if hasattr(legend, "setBrush"):
            legend.setBrush(pg.mkBrush(255, 255, 255, 218))
        if hasattr(legend, "setPen"):
            legend.setPen(pg.mkPen("#D5DAE3", width=1))

    def start_capture(self):
        if self.recording:
            return

        now = time.monotonic()
        self.recording = True
        self.record_request_time = now
        self.recorded_samples = []
        self.last_live_plot_time = 0.0
        self.baseline_samples = [
            sample for sample in self.prebuffer
            if now - self.args.baseline_seconds <= sample[0] < now
        ]
        self._clear_capture_plots()

        self.status_label.setText(
            "<span style='color:#1D4ED8;'>Recording next "
            f"{self.args.capture_seconds:.1f} seconds... keep motion intentional.</span>"
        )

    def update(self):
        got_sample = False
        finished_capture = False

        while True:
            try:
                item = self.samples.get_nowait()
            except queue.Empty:
                break

            if item[0] == "error":
                self.status_label.setText(f"<span style='color:#B42318;'>Serial error: {item[1]}</span>")
                print(f"Serial error: {item[1]}", file=sys.stderr)
                continue

            timestamp, ax, ay, az, gx, gy, gz = item
            sample = (timestamp, ax, ay, az, gx, gy, gz)
            self.latest_sample = sample
            self.total_samples += 1
            got_sample = True

            self.prebuffer.append(sample)
            while self.prebuffer and timestamp - self.prebuffer[0][0] > self.args.baseline_seconds:
                self.prebuffer.popleft()

            if self.recording and timestamp >= self.record_request_time:
                self.recorded_samples.append(sample)
                elapsed = timestamp - self.record_request_time
                if elapsed >= self.args.capture_seconds:
                    self.finish_capture()
                    finished_capture = True
                    break

        if self.recording:
            now = time.monotonic()
            update_interval = 1.0 / self.args.trajectory_update_hz
            if (
                len(self.recorded_samples) >= 3
                and now - self.last_live_plot_time >= update_interval
            ):
                self.update_live_trajectory()
                self.last_live_plot_time = now

        if self.recording and self.record_request_time is not None:
            elapsed = min(self.args.capture_seconds, time.monotonic() - self.record_request_time)
            pct = 100.0 * elapsed / self.args.capture_seconds
            self.status_label.setText(
                "<span style='color:#1D4ED8;'>"
                f"Recording {elapsed:0.2f}/{self.args.capture_seconds:0.2f} s ({pct:0.0f}%) | "
                f"{len(self.recorded_samples)} samples"
                "</span>"
            )
        elif finished_capture:
            return
        elif got_sample and self.latest_sample is not None:
            _, ax, ay, az, gx, gy, gz = self.latest_sample
            self.status_label.setText(
                "<span style='color:#475569;'>"
                "Press <b>c</b> to capture the next "
                f"{self.args.capture_seconds:.1f} s window. "
                f"samples {self.total_samples} | "
                f"accel x={ax: .3f} y={ay: .3f} z={az: .3f} g | "
                f"gyro x={gx: .2f} y={gy: .2f} z={gz: .2f} deg/s"
                "</span>"
            )

    def update_live_trajectory(self):
        try:
            result = compute_trajectory(self.recorded_samples, self.baseline_samples, self.args)
        except ValueError:
            return
        self.update_plots(result)

    def finish_capture(self):
        self.recording = False
        if len(self.recorded_samples) < 3:
            self.status_label.setText("<span style='color:#B42318;'>Capture failed: not enough samples.</span>")
            return

        try:
            result = compute_trajectory(self.recorded_samples, self.baseline_samples, self.args)
        except ValueError as exc:
            self.status_label.setText(f"<span style='color:#B42318;'>{exc}</span>")
            return

        self.update_plots(result)

        pos = result["position_m"]
        displacement = float(np.linalg.norm(pos[-1] - pos[0]))
        path_len = float(np.sum(np.linalg.norm(np.diff(pos[:, :2], axis=0), axis=1)))
        warning = ""
        if result["baseline_warning"]:
            warning = f" | {result['baseline_warning']}"

        self.status_label.setText(
            "<span style='color:#166534;'>"
            f"Done: {len(self.recorded_samples)} samples, "
            f"duration {result['time'][-1]:.2f} s, "
            f"XY path {path_len:.3f} m, displacement {displacement:.3f} m"
            f"{warning}. Press <b>c</b> to capture again."
            "</span>"
        )

    def update_plots(self, result):
        t = result["time"]
        accel = result["accel_g"]
        gyro = result["gyro_dps"]
        pos = result["position_m"]

        self._set_time_colored_trajectory(pos)
        self.start_marker.setData(pos[:1, 0], pos[:1, 1])
        self.end_marker.setData(pos[-1:, 0], pos[-1:, 1])

        for i, axis in enumerate(AXES):
            self.accel_curves[axis].setData(t, accel[:, i])
            self.gyro_curves[axis].setData(t, gyro[:, i])

        self.gyro_plot.setXRange(0.0, max(self.args.capture_seconds, float(t[-1])), padding=0)
        self.accel_plot.enableAutoRange(x=False, y=True)
        self.gyro_plot.enableAutoRange(x=False, y=True)
        self.trajectory_plot.autoRange(padding=0.12)

    def _set_time_colored_trajectory(self, pos):
        for segment in self.trajectory_segments:
            self.trajectory_plot.removeItem(segment)
        self.trajectory_segments.clear()

        if len(pos) < 2:
            self.trajectory_legend_curve.setData(pos[:, 0], pos[:, 1])
            self.trajectory_legend_end_curve.setData([], [])
            return

        pos = downsample_path(pos, self.args.max_trajectory_segments)
        self.trajectory_legend_curve.setData([], [])
        self.trajectory_legend_end_curve.setData([], [])
        denom = max(1, len(pos) - 2)
        for i in range(len(pos) - 1):
            color = interpolate_color(PATH_START_COLOR, PATH_END_COLOR, i / denom)
            segment = pg.PlotCurveItem(
                pos[i:i + 2, 0],
                pos[i:i + 2, 1],
                pen=pg.mkPen(color=color, width=3.0),
            )
            configure_curve(segment)
            self.trajectory_plot.addItem(segment)
            self.trajectory_segments.append(segment)

    def _clear_capture_plots(self):
        self._set_time_colored_trajectory(np.zeros((0, 3), dtype=float))
        self.start_marker.setData([], [])
        self.end_marker.setData([], [])
        for axis in AXES:
            self.accel_curves[axis].setData([], [])
            self.gyro_curves[axis].setData([], [])


def main():
    parser = argparse.ArgumentParser(description="V1: capture BMI270 data and plot one raw trajectory.")
    parser.add_argument("--port", required=True, help="Serial port, e.g. /dev/cu.usbserial-5B1F0080901")
    parser.add_argument("--baud", type=int, default=115200, help="Serial baud rate")
    parser.add_argument("--capture-seconds", type=float, default=5.0, help="Seconds to record after pressing c")
    parser.add_argument("--baseline-seconds", type=float, default=1.0, help="Idle samples before pressing c used as bias baseline")
    parser.add_argument("--fps", type=float, default=100.0, help="GUI refresh rate")
    parser.add_argument("--trajectory-update-hz", type=float, default=15.0, help="Live trajectory redraw rate while recording")
    parser.add_argument("--max-trajectory-segments", type=int, default=180, help="Maximum colored path segments to draw")
    parser.add_argument("--print-unparsed", action="store_true", help="Print serial lines that are not IMU samples")
    parser.add_argument("--accel-deadband", type=float, default=0.08, help="Ignore linear accel below this m/s^2")
    parser.add_argument("--velocity-damping", type=float, default=0.08, help="Velocity damping coefficient in 1/s")
    parser.add_argument("--stationary-accel-threshold", type=float, default=0.18, help="Stationary accel threshold in m/s^2")
    parser.add_argument("--stationary-gyro-threshold", type=float, default=2.0, help="Stationary gyro threshold in deg/s")
    parser.add_argument("--max-dt", type=float, default=0.2, help="Ignore integration step if serial gap exceeds this many seconds")
    parser.add_argument("--min-baseline-samples", type=int, default=8, help="Minimum pre-capture samples for baseline")
    args = parser.parse_args()

    if args.capture_seconds <= 0.0:
        parser.error("--capture-seconds must be positive")
    if args.baseline_seconds <= 0.0:
        parser.error("--baseline-seconds must be positive")
    if args.fps <= 0.0:
        parser.error("--fps must be positive")
    if args.trajectory_update_hz <= 0.0:
        parser.error("--trajectory-update-hz must be positive")
    if args.max_trajectory_segments < 1:
        parser.error("--max-trajectory-segments must be at least 1")
    if args.min_baseline_samples < 1:
        parser.error("--min-baseline-samples must be at least 1")

    samples = queue.Queue()
    stop_event = threading.Event()
    reader = threading.Thread(
        target=serial_worker,
        args=(args.port, args.baud, samples, stop_event, args.print_unparsed),
        daemon=True,
    )
    reader.start()

    signal.signal(signal.SIGINT, signal.SIG_DFL)

    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(
        """
        QWidget {
            background: #F5F7FB;
            color: #273142;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
            font-size: 13px;
        }
        """
    )

    gui = CaptureTrajectoryGui(args, samples)

    timer = QtCore.QTimer()
    timer.timeout.connect(gui.update)
    timer.start(max(1, int(1000 / args.fps)))

    def cleanup():
        timer.stop()
        stop_event.set()
        reader.join(timeout=2)

    app.aboutToQuit.connect(cleanup)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
