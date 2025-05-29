# Drowsiness Detection and Concentration Monitoring System

This project is a **real-time drowsiness detection and concentration monitoring system** using facial features such as **Head Pose**, **Gaze**, and **Blink**.  
A pre-trained **RNN model** is used to predict drowsiness, and the system also provides real-time visual feedback of the user's concentration.  
Bluetooth control for robot interaction is included based on the predicted drowsiness state.

---

## 📌 Features

- Real-time webcam or Raspberry Pi camera input
- Head Pose, Gaze, and Blink detection modules
- RNN-based drowsiness prediction model
- Real-time visualization of concentration levels
- Bluetooth communication for robot control

---

## 🧩 Components

| Module | Description |
|--------|-------------|
| `headpose/` | Head pose estimation |
| `gaze/` | Gaze tracking |
| `blink/` | Blink detection |
| `predictor.py` | RNN inference for real-time drowsiness prediction |
| `models/drowsiness_rnn_best.h5` | Pre-trained RNN model |
| `minmax_scaler.pkl` | MinMaxScaler used during model training |
| `spike_tx.py` | Bluetooth communication and robot command definitions |
| `config.py` | Window dimensions and other UI configurations |
| `main.py` | Main execution file |

---

## 🚀 How to Run

### 1. Install Requirements

```bash
pip install opencv-python-headless matplotlib joblib tensorflow scikit-learn keyboard
```

> On Raspberry Pi, you also need:
```bash
pip install "picamera[array]"
```

---

### 2. Prepare Trained Model and Scaler

Place the following files in the `models/` directory:

```
models/
├── drowsiness_rnn_best.h5
└── minmax_scaler.pkl
```

These should be exported from your training environment (e.g., Colab).

---

### 3. Run the Application

```bash
python main.py
```

- On **Windows**: uses webcam (OpenCV)
- On **Linux/Raspberry Pi**: uses PiCamera

---

## 🎮 Keyboard Controls

- Press `c` → Start calibration
- Press `ESC` → Exit the program

---

## 📡 Robot Command Mapping

Based on RNN predictions:

| Class | Meaning | Command |
|-------|---------|---------|
| 0 | Normal | `DO_NOTHING` |
| 1 | Drowsiness Level 1 | `ATTACK_1` |
| 2 | Drowsiness Level 2 | `ATTACK_2` |

Commands are sent to the robot via Bluetooth using `spike_tx.py`.

---

## 📊 Concentration Visualization

Real-time graphs are shown using **Matplotlib** embedded into **OpenCV** windows.  
It visualizes Gaze, Blink, HeadPose, and an overall attention metric.

---

## 📁 Project Structure

```
project/
├── main.py
├── predictor.py
├── config.py
├── spike_tx.py
├── headpose/
├── gaze/
├── blink/
├── models/
│   ├── drowsiness_rnn_best.h5
│   └── minmax_scaler.pkl
```

---

## ⚠️ Notes

- `keyboard` module may require administrator privileges.
- Make sure the model and scaler were trained using the same scaling method (e.g., MinMaxScaler).
- Ensure your camera is compatible with OpenCV or PiCamera.
- Use `joblib` to export/import the trained scaler consistently between training and inference.

---

## 📬 Contact

For questions or contributions, feel free to open an Issue or Pull Request!
