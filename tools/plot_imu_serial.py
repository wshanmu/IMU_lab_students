#!/usr/bin/env python3
"""Read ESP32 BMI270 serial logs and plot the six IMU channels live."""

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
from pyqtgraph.Qt import QtCore, QtWidgets


FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
SAMPLE_RE = re.compile(
    rf"accel\[g\]\s+x=\s*({FLOAT_RE})\s+y=\s*({FLOAT_RE})\s+z=\s*({FLOAT_RE})"
    rf"\s+\|\s+gyro\[dps\]\s+x=\s*({FLOAT_RE})\s+y=\s*({FLOAT_RE})\s+z=\s*({FLOAT_RE})"
)

AXES = ("x", "y", "z")
CHANNELS = ("ax", "ay", "az", "gx", "gy", "gz")
SEABORN_COLORS = {
    "x": "#4C72B0",
    "y": "#55A868",
    "z": "#C44E52",
}


def parse_sample(line):
    match = SAMPLE_RE.search(line)
    if not match:
        return None
    return tuple(float(value) for value in match.groups())


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


class ImuPlotter:
    def __init__(self, args, samples):
        self.args = args
        self.samples = samples
        self.start_time = None
        self.sample_count = 0

        self.times = deque(maxlen=args.max_samples)
        self.values = {channel: deque(maxlen=args.max_samples) for channel in CHANNELS}

        pg.setConfigOptions(
            antialias=True,
            background="#F5F7FB",
            foreground="#273142",
        )

        self.window = pg.GraphicsLayoutWidget(show=True, title="ESP32 BMI270 Live Readings")
        self.window.resize(1160, 720)
        self.window.setBackground("#F5F7FB")

        self.title = self.window.addLabel(
            "<span style='font-size:22px; font-weight:700; color:#1F2937;'>ESP32 BMI270 Live Readings</span>",
            row=0,
            col=0,
        )
        self.accel_plot = self.window.addPlot(row=1, col=0, title="Accelerometer")
        self.gyro_plot = self.window.addPlot(row=2, col=0, title="Gyroscope")
        self.status_label = self.window.addLabel(
            "<span style='color:#64748B;'>Waiting for serial samples...</span>",
            row=3,
            col=0,
        )

        self._setup_plot(self.accel_plot, "Accelerometer", "Acceleration", "g")
        self._setup_plot(self.gyro_plot, "Gyroscope", "Angular Velocity", "deg/s")
        self.gyro_plot.setLabel("bottom", "Time", units="s")
        self.accel_plot.setXLink(self.gyro_plot)

        self.accel_curves = {
            axis: self.accel_plot.plot(
                pen=pg.mkPen(SEABORN_COLORS[axis], width=2.4),
                name=f"a{axis}",
            )
            for axis in AXES
        }
        self.gyro_curves = {
            axis: self.gyro_plot.plot(
                pen=pg.mkPen(SEABORN_COLORS[axis], width=2.4),
                name=f"g{axis}",
            )
            for axis in AXES
        }

        for curve in (*self.accel_curves.values(), *self.gyro_curves.values()):
            curve.setClipToView(True)
            curve.setDownsampling(auto=True, method="peak")

    def _setup_plot(self, plot, title, left_label, units):
        plot.setTitle(f"<span style='font-size:15px; color:#273142;'>{title}</span>")
        plot.setLabel("left", left_label, units=units, color="#334155")
        plot.showGrid(x=True, y=True, alpha=0.22)
        plot.setMenuEnabled(False)
        plot.hideButtons()
        plot.enableAutoRange(x=False, y=True)

        view_box = plot.getViewBox()
        view_box.setBackgroundColor("#FFFFFF")
        view_box.setBorder(pg.mkPen("#D5DAE3", width=1))

        for axis_name in ("left", "bottom"):
            axis = plot.getAxis(axis_name)
            axis.setPen(pg.mkPen("#CBD5E1", width=1))
            axis.setTextPen(pg.mkPen("#475569"))
            axis.setStyle(tickFont=None, autoExpandTextSpace=True)

        legend = plot.addLegend(offset=(-12, 12))
        if hasattr(legend, "setBrush"):
            legend.setBrush(pg.mkBrush(255, 255, 255, 218))
        if hasattr(legend, "setPen"):
            legend.setPen(pg.mkPen("#D5DAE3", width=1))

    def update(self):
        latest = None

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
            if self.start_time is None:
                self.start_time = timestamp

            self.times.append(timestamp - self.start_time)
            for channel, value in zip(CHANNELS, (ax, ay, az, gx, gy, gz)):
                self.values[channel].append(value)

            latest = (ax, ay, az, gx, gy, gz)
            self.sample_count += 1

        if latest is None:
            return

        x_data = np.fromiter(self.times, dtype=float, count=len(self.times))
        for axis in AXES:
            self.accel_curves[axis].setData(
                x_data,
                np.fromiter(self.values[f"a{axis}"], dtype=float, count=len(self.values[f"a{axis}"])),
            )
            self.gyro_curves[axis].setData(
                x_data,
                np.fromiter(self.values[f"g{axis}"], dtype=float, count=len(self.values[f"g{axis}"])),
            )

        right = max(self.args.window, float(x_data[-1]))
        self.gyro_plot.setXRange(max(0.0, right - self.args.window), right, padding=0)

        ax, ay, az, gx, gy, gz = latest
        self.status_label.setText(
            "<span style='color:#475569;'>"
            f"samples {self.sample_count} &nbsp; "
            f"accel: x={ax: .3f} y={ay: .3f} z={az: .3f} g &nbsp; "
            f"gyro: x={gx: .2f} y={gy: .2f} z={gz: .2f} deg/s"
            "</span>"
        )


def main():
    parser = argparse.ArgumentParser(description="Live plot ESP32 BMI270 serial output.")
    parser.add_argument("--port", required=True, help="Serial port, e.g. /dev/cu.usbserial-5B1F0080901")
    parser.add_argument("--baud", type=int, default=115200, help="Serial baud rate")
    parser.add_argument("--window", type=float, default=10.0, help="Seconds of data to keep visible")
    parser.add_argument("--max-samples", type=int, default=5000, help="Maximum samples kept in memory")
    parser.add_argument("--fps", type=float, default=100.0, help="Plot refresh rate")
    parser.add_argument("--print-unparsed", action="store_true", help="Print serial lines that are not IMU samples")
    args = parser.parse_args()

    if args.fps <= 0.0:
        parser.error("--fps must be positive")
    if args.window <= 0.0:
        parser.error("--window must be positive")

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

    plotter = ImuPlotter(args, samples)

    timer = QtCore.QTimer()
    timer.timeout.connect(plotter.update)
    timer.start(max(1, int(1000 / args.fps)))

    def cleanup():
        timer.stop()
        stop_event.set()
        reader.join(timeout=2)

    app.aboutToQuit.connect(cleanup)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
