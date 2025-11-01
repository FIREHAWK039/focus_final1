# Focus Detector

A real-time focus percentage estimator using MediaPipe FaceMesh and Pose.

Features:
- Gaze, head orientation, eye openness, and posture are combined into a single Focus percentage (0-100).
- Calibration mode: press `c` to calibrate baseline for 5 seconds.
- Toggle logging with `l` to write per-frame CSV logs to `focus_log.csv`.
- Alert when focus < 40% for > 10 seconds.

Requirements
- Python 3.8+
- See `requirements.txt` for packages (mediapipe, opencv-python, numpy)

Install

```bash
python -m pip install -r requirements.txt
```

Run

```bash
python focus_detector.py --camera 0 --log
```

Run the web UI (serves at http://localhost:5000):

```bash
python app.py
```

Controls
- `c` : Calibrate baseline (5 seconds)
- `l` : Toggle logging (writes to `focus_log.csv`)
- `q` or `Esc` : Quit

How focus is computed
- Sub-scores (0..1) are computed each frame for:
  - gaze: how centered the iris is in each eye
  - head: small yaw/pitch/roll increase score
  - eye: normalized eye openness (EAR-like)
  - posture: vertical alignment of shoulders and hips
  - face_presence: 1 if face detected else 0
- Combined with weights: gaze 0.35, head 0.25, eye 0.15, posture 0.15, face 0.10
- Final value multiplied by 100 to give percentage and smoothed with a 5-frame moving average.

Calibration & Personalization
- Press `c` and look at the camera normally for 5 seconds. A baseline JSON is saved to `baseline_focus_baseline.json`.
- Baseline values are used to normalize future frames, improving per-user accuracy.

Notes & Future improvements
- Replace heuristic head pose with solvePnP for more accurate 3D head rotation.
- Use MediaPipe Iris landmarks more robustly or train a small model to map raw features to subjective labels.
- Add a Streamlit dashboard for visualization and controls.
- Add optional sound/vibration alerts and throttling to avoid false positives.

License: MIT
# focus_final1
# focus2o
