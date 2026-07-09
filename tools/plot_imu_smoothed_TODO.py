#!/usr/bin/env python3
"""Live plot raw and moving-average smoothed BMI270 serial readings."""

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


class MovingAverageFilter:
    """Streaming moving average.

    TODO SECTION:
    The task is to implement update(new_value) so it returns the average of
    the most recent N samples, where N is self.window_size.

    Notes:
    - Do not recompute sum(self.window) every time.
    - Keep a running_sum.
    - When the window is full, subtract the oldest sample before appending.
    - Return running_sum / number_of_samples_currently_in_window.

    Please finished the TODO
    Useful information:
    - self.window is a deque with maxlen=self.window_size
    - Two conditions: 1. when the window is not full, 2. when the window is full.
    - More information about deque can be found here:
    https://www.geeksforgeeks.org/python/deque-in-python/
    """

    def __init__(self, window_size):
        self.window_size = window_size
        self.window = deque()
        self.running_sum = 0.0

    def update(self, new_value):
        # TODO: Implement the moving average update logic here
        pass


def configure_curve(curve):
    if hasattr(curve, "setClipToView"):
        curve.setClipToView(True)
    if hasattr(curve, "setDownsampling"):
        curve.setDownsampling(auto=True, method="peak")


class SmoothedImuPlotter:
    def __init__(self, args, samples):
        self.args = args
        self.samples = samples
        self.start_time = None
        self.sample_count = 0
        self.latest_raw = None
        self.latest_smooth = None

        self.times = deque(maxlen=args.max_samples)
        self.values = {channel: deque(maxlen=args.max_samples) for channel in CHANNELS}
        self.smoothed_values = {channel: deque(maxlen=args.max_samples) for channel in CHANNELS}
        self.filters = {
            channel: MovingAverageFilter(args.average_window)
            for channel in CHANNELS
        }

        pg.setConfigOptions(
            antialias=True,
            background="#F5F7FB",
            foreground="#273142",
        )

        self.window = pg.GraphicsLayoutWidget(show=True, title="ESP32 BMI270 Raw + Smoothed Readings")
        self.window.resize(1180, 760)
        self.window.setBackground("#F5F7FB")

        self.window.addLabel(
            "<span style='font-size:22px; font-weight:700; color:#1F2937;'>ESP32 BMI270 Raw + Moving Average</span>",
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

        self.raw_accel_curves = {}
        self.smooth_accel_curves = {}
        self.raw_gyro_curves = {}
        self.smooth_gyro_curves = {}

        for axis in AXES:
            color = SEABORN_COLORS[axis]
            raw_pen = pg.mkPen(color=color, width=1.2, style=QtCore.Qt.PenStyle.DashLine)
            smooth_pen = pg.mkPen(color=color, width=2.8)

            self.raw_accel_curves[axis] = self.accel_plot.plot(pen=raw_pen, name=f"a{axis} raw")
            self.smooth_accel_curves[axis] = self.accel_plot.plot(pen=smooth_pen, name=f"a{axis} smooth")
            self.raw_gyro_curves[axis] = self.gyro_plot.plot(pen=raw_pen, name=f"g{axis} raw")
            self.smooth_gyro_curves[axis] = self.gyro_plot.plot(pen=smooth_pen, name=f"g{axis} smooth")

        for curve in (
            *self.raw_accel_curves.values(),
            *self.smooth_accel_curves.values(),
            *self.raw_gyro_curves.values(),
            *self.smooth_gyro_curves.values(),
        ):
            configure_curve(curve)

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
        got_sample = False

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
                self.smoothed_values[channel].append(self.filters[channel].update(value))

            self.latest_raw = (ax, ay, az, gx, gy, gz)
            self.latest_smooth = tuple(self.smoothed_values[channel][-1] for channel in CHANNELS)
            self.sample_count += 1
            got_sample = True

        if not got_sample:
            return

        x_data = np.fromiter(self.times, dtype=float, count=len(self.times))

        for axis in AXES:
            raw_accel = np.fromiter(self.values[f"a{axis}"], dtype=float, count=len(self.values[f"a{axis}"]))
            raw_gyro = np.fromiter(self.values[f"g{axis}"], dtype=float, count=len(self.values[f"g{axis}"]))
            smooth_accel = np.fromiter(
                self.smoothed_values[f"a{axis}"],
                dtype=float,
                count=len(self.smoothed_values[f"a{axis}"]),
            )
            smooth_gyro = np.fromiter(
                self.smoothed_values[f"g{axis}"],
                dtype=float,
                count=len(self.smoothed_values[f"g{axis}"]),
            )

            self.raw_accel_curves[axis].setData(x_data, raw_accel)
            self.smooth_accel_curves[axis].setData(x_data, smooth_accel)
            self.raw_gyro_curves[axis].setData(x_data, raw_gyro)
            self.smooth_gyro_curves[axis].setData(x_data, smooth_gyro)

        right = max(self.args.window, float(x_data[-1]))
        self.gyro_plot.setXRange(max(0.0, right - self.args.window), right, padding=0)

        ax, ay, az, gx, gy, gz = self.latest_raw
        sax, say, saz, sgx, sgy, sgz = self.latest_smooth
        self.status_label.setText(
            "<span style='color:#475569;'>"
            f"samples {self.sample_count} | moving average window {self.args.average_window} samples<br>"
            f"raw accel: x={ax: .3f} y={ay: .3f} z={az: .3f} g &nbsp; "
            f"smooth accel: x={sax: .3f} y={say: .3f} z={saz: .3f} g<br>"
            f"raw gyro: x={gx: .2f} y={gy: .2f} z={gz: .2f} deg/s &nbsp; "
            f"smooth gyro: x={sgx: .2f} y={sgy: .2f} z={sgz: .2f} deg/s"
            "</span>"
        )


def main():
    parser = argparse.ArgumentParser(description="Live plot raw and moving-average BMI270 serial output.")
    parser.add_argument("--port", required=True, help="Serial port, e.g. /dev/cu.usbserial-5B1F0080901")
    parser.add_argument("--baud", type=int, default=115200, help="Serial baud rate")
    parser.add_argument("--window", type=float, default=10.0, help="Seconds of data to keep visible")
    parser.add_argument("--max-samples", type=int, default=5000, help="Maximum samples kept in memory")
    parser.add_argument("--fps", type=float, default=60.0, help="Plot refresh rate")
    parser.add_argument("--average-window", type=int, default=5, help="Moving average length in samples")
    parser.add_argument("--print-unparsed", action="store_true", help="Print serial lines that are not IMU samples")
    args = parser.parse_args()

    if args.fps <= 0.0:
        parser.error("--fps must be positive")
    if args.window <= 0.0:
        parser.error("--window must be positive")
    if args.average_window < 1:
        parser.error("--average-window must be at least 1")

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

    plotter = SmoothedImuPlotter(args, samples)

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
