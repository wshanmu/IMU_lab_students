# IMU Classifier Student Demo

This folder contains the student version of the Lab 5 IMU classifier. It reads BMI270 data from the ESP32 serial port, records 1-second IMU windows, extracts features, and trains a simple linear SVM classifier.

Each training example is one 1-second segment containing:

- accelerometer: `ax`, `ay`, `az` in `g`
- gyroscope: `gx`, `gy`, `gz` in `deg/s`
- one label from three task classes

The default classes are `Task 1`, `Task 2`, and `Task 3`. You can rename them from the command line.

## Setup

From `tools/IMU_Classifier`:

```bash
python -m pip install -r requirements-python.txt
```

## Run With ESP32

Find the ESP32 serial port.

macOS:

```bash
ls /dev/cu.usb* /dev/cu.SLAB* /dev/cu.wch* 2>/dev/null
```

Windows:

```text
Open Device Manager -> Ports (COM & LPT).
```

Run the student TODO version:

macOS/Linux:

```bash
./run_student.sh --port /dev/cu.usbserial-5B1F0080901 --classes Still,Shake,Turn
```

Windows PowerShell:

```powershell
python python_imu_segment_demo_student.py --port COM5 --classes Still,Shake,Turn
```

Test the GUI without hardware:

```bash
python python_imu_segment_demo_student.py --demo-signal --classes Still,Shake,Turn
```

## Workflow

1. Pick three tasks, for example `Still`, `Shake`, and `Turn`.
2. Select the class in the dropdown.
3. Press `c`, `Space`, `Enter`, or the `Capture 1 s` button.
4. Perform that task for the next 1 second.
5. Collect at least five examples for each class if time allows.
6. Press `t` or `Train`.
7. Perform a task again and watch the live classification.
8. Press `d` or `Save Data` to save the captured training examples.
9. Press `s` to save the trained model, or `l` to load it later.
10. In a later run, press `o` or `Load Data` to reload saved examples, then press `t` to train again.

## Controls

- `Down`: select the next class
- `c`, `Space`, or `Enter`: record the next 1-second segment
- `t`: train the classifier, or switch back to collection mode
- `s`: save the trained model
- `l`: load the saved model
- `d`: save captured training examples
- `o`: load captured training examples
- `b`: toggle detection logging

## Student TODO

In `python_imu_segment_demo_student.py`, find the `STUDENT TODO` block inside `ImuSegmentFeatureExtractor.capture_instance`.

The student version currently uses only the resampled and centered waveform:

```python
measurements = centered.T.reshape(-1).astype("float32")
```

Improve the classifier by computing extra statistics and concatenating them:

```python
summary = self.np.concatenate([...]).astype("float32")
measurements = self.np.concatenate([centered.T.reshape(-1), summary]).astype("float32")
```

Suggested statistics:

- standard deviation of each channel
- min and max of each channel
- range of each channel
- RMS energy of each channel
- mean absolute first difference of each channel

## Notes

- Keep each task motion consistent while collecting training data.
- Make sure every class has a similar number of examples.
- If predictions are unstable, collect more examples or make the three tasks more different.
- Saved training examples go to `training_data_imu_student.joblib` by default.
- Use `--training-data my_examples.joblib` to choose a different training-data file.
- Saved models only work with the same class names and feature settings used during training.
