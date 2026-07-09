# UCLA COSMOS Cluster 10 - Lab 4 IMU

This repository contains the code for the COSMOS Cluster 10 IMU lab. The lab uses an ESP32 microcontroller with a BMI270 IMU to stream 6-axis motion data over USB serial, then uses Python scripts for visualization, smoothing, trajectory estimation, and simple motion classification.

## What You Will Use

| Path | Purpose |
| --- | --- |
| `main/main.c` | ESP-IDF firmware for streaming BMI270 accelerometer and gyroscope readings. |
| `tools/plot_imu_serial.py` | Live raw accelerometer and gyroscope plot. |
| `tools/plot_imu_smoothed_TODO.py` | Student TODO for moving-average smoothing. |
| `tools/capture_trajectory.py` | Short trajectory capture demo. |
| `tools/capture_trajectory_smoothed_TODO.py` | Student TODO for smoothing before trajectory estimation. |
| `tools/IMU_Classifier/python_imu_segment_demo_student.py` | Student TODO for a 3-class motion classifier. |

The lab firmware may already be flashed on your board. If so, you can skip directly to [Python setup](#python-setup).

## Clone or Update the Repository

Clone the repository once:

```bash
git clone https://github.com/wshanmu/IMU_lab.git
cd IMU_lab
```

Before lab, pull the latest version:

```bash
git pull
```

If you edited files and `git pull` refuses to continue, ask a TA before running any reset or checkout command.

## Hardware Overview

The data path is:

```text
BMI270 IMU -> I2C -> ESP32 -> USB serial -> Python scripts
```

The firmware uses:

| Item | Value |
| --- | --- |
| ESP-IDF target | `esp32` |
| I2C SDA | GPIO 21 |
| I2C SCL | GPIO 22 |
| IMU | BMI270 |
| Accelerometer range | `+/-4 g` |
| Gyroscope range | `+/-1000 dps` |
| Approximate sample rate | 100 Hz |
| Serial baud rate | 115200 |

Expected serial output:

```text
accel[g] x= 0.012 y=-0.034 z= 0.998 | gyro[dps] x= 0.10 y=-0.20 z= 0.05
```

## Python Setup

Use the `cosmos-ds` Conda environment from Lab 1.

```bash
cd tools
conda activate cosmos-ds
python -m pip install -r requirements.txt
```

If `conda activate cosmos-ds` fails, return to the Lab 1 environment setup and create the course environment before continuing.

Find the ESP32 serial port.

macOS:

```bash
ls /dev/cu.* 2>/dev/null
```

Windows:

```text
Open Device Manager -> Ports (COM & LPT), then look for the new COM port, such as COM5.
```

## Run the Python Tools

Replace `YOUR_PORT` with your actual serial port.

Raw IMU plot:

```bash
python plot_imu_serial.py --port YOUR_PORT --baud 115200
```

Moving-average TODO:

```bash
python plot_imu_smoothed_TODO.py --port YOUR_PORT --baud 115200 --average-window 5
```

Trajectory capture:

```bash
python capture_trajectory.py --port YOUR_PORT --baud 115200 --capture-seconds 5
```

Smoothed trajectory TODO:

```bash
python capture_trajectory_smoothed_TODO.py --port YOUR_PORT --baud 115200 --capture-seconds 5 --average-window 5
```

Classifier:

```bash
cd IMU_Classifier
python -m pip install -r requirements-python.txt
./run_student.sh --port YOUR_PORT --classes Still,Shake,Turn
```

On Windows PowerShell, run the classifier directly:

```powershell
cd IMU_Classifier
python -m pip install -r requirements-python.txt
python python_imu_segment_demo_student.py --port YOUR_PORT --classes Still,Shake,Turn
```

## Flash the ESP32 Firmware

Only flash the board if the TA asks you to, or if the board is not producing the expected serial output.

### Option A: VS Code ESP-IDF Extension

ESP-IDF setup is covered in the WiFi lab. Reuse that same VS Code extension and ESP-IDF installation here.

1. Open this repository folder in VS Code.
2. Open the command palette:
   - macOS: `Command+Shift+P`
   - Windows/Linux: `Ctrl+Shift+P`
3. Run `ESP-IDF: Set Espressif Device Target` and choose `esp32`.
4. Run `ESP-IDF: Select Port to Use` and select the ESP32 serial port.
5. Run `ESP-IDF: Build your Project`.
6. Run `ESP-IDF: Flash your Project`.
7. Run `ESP-IDF: Monitor your Device` and check for the expected `accel[g] ... gyro[dps] ...` output.

### Option B: Command Line

Open an ESP-IDF terminal with the ESP-IDF environment activated.

From the repository root:

```bash
idf.py set-target esp32
idf.py build
idf.py -p YOUR_PORT flash monitor
```

Examples:

```bash
idf.py -p /dev/cu.usbserial-5B1F0080901 flash monitor
idf.py -p COM5 flash monitor
```

To leave the ESP-IDF monitor, press:

```text
Ctrl-]
```

## Flashing Troubleshooting

| Problem | Likely Cause | Fix |
| --- | --- | --- |
| Port does not appear | USB cable or driver issue | Use a data-capable USB cable, reconnect the board, or try another USB port. |
| Port is busy | Another program is using serial | Close Python scripts, PuTTY, Arduino monitor, or ESP-IDF monitor. |
| Build cannot find components | ESP-IDF component manager did not resolve dependencies | Confirm internet access and rerun `idf.py reconfigure` or `idf.py build`. |
| Permission denied on macOS/Linux | Serial device permission issue | Reconnect the board; on Linux, ask the TA about dialout/uucp permissions. |
| Output is unreadable | Wrong baud rate or wrong firmware | Use `115200` baud and reflash the lab firmware. |

## Official References

- [ESP-IDF Get Started for ESP32](https://docs.espressif.com/projects/esp-idf/en/latest/esp32/get-started/index.html)
- [ESP-IDF VS Code Extension installation](https://docs.espressif.com/projects/vscode-esp-idf-extension/en/latest/installation.html)
- [Bosch BMI270 product page](https://www.bosch-sensortec.com/en/products/motion-sensors/imus/bmi270)
