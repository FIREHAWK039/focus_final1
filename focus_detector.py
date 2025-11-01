"""
Focus Detector

Real-time focus percentage estimator using MediaPipe FaceMesh and Pose.

Features computed per frame:
- gaze_score: how centered the iris is relative to eye center (0-1)
- head_score: small yaw/pitch/roll => high score (0-1)
- eye_score: eye openness (EAR-like) (0-1)
- posture_score: uprightness computed from shoulder-hip vertical angle (0-1)
- face_presence: whether face landmarks are detected (0 or 1)

Combined: focus = 0.35*gaze + 0.25*head + 0.15*eye + 0.15*posture + 0.10*face
Smoothed by moving average (window=5 frames) and output as percentage.

Calibration: press 'c' to calibrate for 5 seconds while looking at the camera. Baseline saved to baseline.json
Toggle logging with 'l'. Quit with 'q' or ESC.

Requires: mediapipe, opencv-python, numpy
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import json
import os
from collections import deque
import argparse

# ------------------------- Configuration -------------------------
WEIGHTS = {
    "gaze": 0.35,
    "head": 0.25,
    "eye": 0.15,
    "posture": 0.15,
    "face": 0.10,
}
SMOOTHING_WINDOW = 5  # frames for moving average
CALIBRATE_SECONDS = 5
ALERT_THRESHOLD = 40.0  # percent
ALERT_SECONDS = 10.0
BASELINE_FILE = "baseline_focus_baseline.json"
LOG_FILE = "focus_log.csv"

# Landmark indices (MediaPipe FaceMesh)
# These are approximate common indices used for quick heuristics.
L_EYE = {"left":33, "right":133, "top":159, "bottom":145}
R_EYE = {"left":362, "right":263, "top":386, "bottom":374}
# Iris landmarks (MediaPipe uses 468-473 typically)
L_IRIS = [468, 469, 470, 471]
R_IRIS = [472, 473, 474, 475]

# Pose indices (MediaPipe Pose)
POSE_L_SHOULDER = 11
POSE_R_SHOULDER = 12
POSE_L_HIP = 23
POSE_R_HIP = 24

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose

# ------------------------- Utility functions -------------------------

def norm(v):
    return np.linalg.norm(v)


def moving_average(deq):
    if not deq:
        return 0.0
    return float(sum(deq) / len(deq))


# Compute eye openness similar to EAR but using MediaPipe indices
def eye_openness(landmarks, eye_indices, image_w, image_h):
    try:
        left = landmarks[eye_indices['left']]
        right = landmarks[eye_indices['right']]
        top = landmarks[eye_indices['top']]
        bottom = landmarks[eye_indices['bottom']]
    except Exception:
        return 0.0

    l = np.array([left.x * image_w, left.y * image_h])
    r = np.array([right.x * image_w, right.y * image_h])
    t = np.array([top.x * image_w, top.y * image_h])
    b = np.array([bottom.x * image_w, bottom.y * image_h])

    hor = norm(r - l)
    ver = norm(t - b)
    if hor <= 1e-6:
        return 0.0
    openness = ver / hor  # higher when eyes are open
    # Typical open ratio observed ~0.18-0.30 for open eyes depending on face size
    # We'll normalize using a soft mapping: 0.0..0.4 -> 0..1
    return np.clip((openness - 0.08) / (0.3 - 0.08), 0.0, 1.0)


# Approx gaze score using iris center relative to eye center.
def gaze_score(landmarks, image_w, image_h):
    # If iris landmarks missing, fall back to center of eye corners
    try:
        # left iris center
        l_iris_pts = [landmarks[i] for i in L_IRIS]
        r_iris_pts = [landmarks[i] for i in R_IRIS]
        l_center = np.mean([[p.x * image_w, p.y * image_h] for p in l_iris_pts], axis=0)
        r_center = np.mean([[p.x * image_w, p.y * image_h] for p in r_iris_pts], axis=0)

        l_eye_left = np.array([landmarks[L_EYE['left']].x * image_w, landmarks[L_EYE['left']].y * image_h])
        l_eye_right = np.array([landmarks[L_EYE['right']].x * image_w, landmarks[L_EYE['right']].y * image_h])
        r_eye_left = np.array([landmarks[R_EYE['left']].x * image_w, landmarks[R_EYE['left']].y * image_h])
        r_eye_right = np.array([landmarks[R_EYE['right']].x * image_w, landmarks[R_EYE['right']].y * image_h])

        # normalize horizontal offset within each eye box
        def hor_pos(center, left, right):
            width = norm(right - left)
            if width <= 1e-6:
                return 0.5
            return np.clip((center[0] - left[0]) / width, 0.0, 1.0)

        l_pos = hor_pos(l_center, l_eye_left, l_eye_right)
        r_pos = hor_pos(r_center, r_eye_left, r_eye_right)

        # If both eyes look roughly centered (pos ~0.35-0.65), good.
        # Compute distance from 0.5 (center) and map to score
        avg_dist = (abs(l_pos - 0.5) + abs(r_pos - 0.5)) / 2.0
        score = 1.0 - np.clip(avg_dist / 0.5, 0.0, 1.0)
        return float(score)
    except Exception:
        # fallback: neutral score
        return 0.5


# Head orientation approximations using face landmarks (eyes and nose)
def head_orientation_score(landmarks, image_w, image_h):
    try:
        left_eye = np.array([landmarks[L_EYE['left']].x * image_w, landmarks[L_EYE['left']].y * image_h])
        right_eye = np.array([landmarks[R_EYE['right']].x * image_w, landmarks[R_EYE['right']].y * image_h])
        # nose tip commonly at index 1 or 4; MediaPipe typical nose tip index is 1 or 4, but we'll use 1
        nose = np.array([landmarks[1].x * image_w, landmarks[1].y * image_h])

        # roll: angle between eyes
        eye_vec = right_eye - left_eye
        roll = np.degrees(np.arctan2(eye_vec[1], eye_vec[0]))

        # yaw: nose x relative to eye midpoint: large lateral offset -> looking sideways
        eye_mid = (left_eye + right_eye) / 2.0
        yaw = np.degrees(np.arctan2(nose[0] - eye_mid[0], max(1e-6, norm(eye_vec))))

        # pitch: nose y relative to eye midpoint: looking up/down
        pitch = np.degrees(np.arctan2(nose[1] - eye_mid[1], max(1e-6, norm(eye_vec))))

        # Map absolute angles to scores: smaller absolute angles -> higher score
        # Roll typical small +/-10 deg, yaw +/- 40, pitch +/- 30
        roll_score = 1.0 - np.clip(abs(roll) / 25.0, 0.0, 1.0)
        yaw_score = 1.0 - np.clip(abs(yaw) / 45.0, 0.0, 1.0)
        pitch_score = 1.0 - np.clip(abs(pitch) / 30.0, 0.0, 1.0)

        combined = (0.5 * yaw_score) + (0.3 * pitch_score) + (0.2 * roll_score)
        return float(np.clip(combined, 0.0, 1.0))
    except Exception:
        return 0.5


# Posture uprightness score from pose landmarks (shoulder-hip vertical alignment)
def posture_score(pose_landmarks, image_w, image_h):
    try:
        ls = pose_landmarks.landmark[POSE_L_SHOULDER]
        rs = pose_landmarks.landmark[POSE_R_SHOULDER]
        lh = pose_landmarks.landmark[POSE_L_HIP]
        rh = pose_landmarks.landmark[POSE_R_HIP]

        shoulder_mid = np.array([(ls.x + rs.x) / 2 * image_w, (ls.y + rs.y) / 2 * image_h])
        hip_mid = np.array([(lh.x + rh.x) / 2 * image_w, (lh.y + rh.y) / 2 * image_h])

        spine_vec = hip_mid - shoulder_mid
        # angle between spine vector and positive Y axis (downwards). If upright, spine vec is mostly down => angle ~0
        vertical = np.array([0.0, 1.0])
        spine_norm = norm(spine_vec)
        if spine_norm <= 1e-6:
            return 0.5
        cos_ang = np.dot(spine_vec / spine_norm, vertical)  # in [-1,1]
        angle_deg = np.degrees(np.arccos(np.clip(cos_ang, -1.0, 1.0)))
        # map 0..45 deg -> 1..0
        score = 1.0 - np.clip(angle_deg / 45.0, 0.0, 1.0)
        return float(score)
    except Exception:
        return 0.5


# ------------------------- Main Detector Class -------------------------
class FocusDetector:
    def __init__(self, camera_idx=0, enable_logging=False, baseline_file=BASELINE_FILE):
        self.cap = cv2.VideoCapture(camera_idx)
        self.face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.pose = mp_pose.Pose(static_image_mode=False, model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.scores_deque = deque(maxlen=SMOOTHING_WINDOW)
        self.raw_deque = deque(maxlen=SMOOTHING_WINDOW)
        self.baseline = None
        self.baseline_file = baseline_file
        self.enable_logging = enable_logging
        self.log_file = LOG_FILE
        self.alert_state = False
        self.alert_start_time = None
        self.last_frame_time = None

        # load baseline if exists
        if os.path.exists(self.baseline_file):
            try:
                with open(self.baseline_file, 'r') as f:
                    self.baseline = json.load(f)
                    print(f"Loaded baseline from {self.baseline_file}")
            except Exception:
                self.baseline = None

        if self.enable_logging:
            # write header
            with open(self.log_file, 'w') as f:
                f.write('timestamp,focus,gaze,head,eye,posture,face_presence\n')

    def normalize_with_baseline(self, features):
        # features is a dict of raw scores in 0..1; baseline similar
        if not self.baseline:
            return features
        normed = {}
        for k, v in features.items():
            base = self.baseline.get(k, 0.5)
            # normalize by baseline: ratio or difference depending on the feature
            # We'll use simple offset normalization: centered so baseline -> 0.5
            normed[k] = float(np.clip((v / max(1e-6, base)) * 0.5, 0.0, 1.0))
        return normed

    def compute_focus(self, features):
        # features: gaze, head, eye, posture, face_presence (0/1)
        weighted = (WEIGHTS['gaze'] * features['gaze'] +
                    WEIGHTS['head'] * features['head'] +
                    WEIGHTS['eye'] * features['eye'] +
                    WEIGHTS['posture'] * features['posture'] +
                    WEIGHTS['face'] * features['face_presence'])
        return weighted * 100.0

    def calibrate(self, seconds=CALIBRATE_SECONDS):
        print(f"Calibration: please look at the camera for {seconds} seconds...")
        t0 = time.time()
        counts = 0
        accum = {'gaze':0.0, 'head':0.0, 'eye':0.0, 'posture':0.0}
        while time.time() - t0 < seconds:
            ret, frame = self.cap.read()
            if not ret:
                continue
            image_h, image_w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_results = self.face_mesh.process(rgb)
            pose_results = self.pose.process(rgb)
            if face_results.multi_face_landmarks and pose_results.pose_landmarks:
                landmarks = face_results.multi_face_landmarks[0].landmark
                gaze = gaze_score(landmarks, image_w, image_h)
                head = head_orientation_score(landmarks, image_w, image_h)
                eye = (eye_openness(landmarks, L_EYE, image_w, image_h) + eye_openness(landmarks, R_EYE, image_w, image_h)) / 2.0
                posture = posture_score(pose_results.pose_landmarks, image_w, image_h)
                accum['gaze'] += gaze
                accum['head'] += head
                accum['eye'] += eye
                accum['posture'] += posture
                counts += 1
            cv2.putText(frame, f"Calibrating... {int(time.time()-t0)}s", (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
            cv2.imshow('Calibration', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyWindow('Calibration')
        if counts == 0:
            print("Calibration failed: no valid frames captured.")
            return False
        baseline = {k: (accum[k] / counts) for k in accum}
        # ensure face presence baseline included
        baseline['face_presence'] = 1.0
        self.baseline = baseline
        with open(self.baseline_file, 'w') as f:
            json.dump(self.baseline, f, indent=2)
        print(f"Calibration complete. Baseline saved to {self.baseline_file}")
        return True

    def run(self, headless: bool = False, max_frames: int = None):
        print("Starting Focus Detector. Press 'c' to calibrate, 'l' to toggle logging, 'q' to quit.")
        frame_counter = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            image_h, image_w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_results = self.face_mesh.process(rgb)
            pose_results = self.pose.process(rgb)

            features = {'gaze':0.5, 'head':0.5, 'eye':0.5, 'posture':0.5, 'face_presence':0.0}

            if face_results.multi_face_landmarks:
                landmarks = face_results.multi_face_landmarks[0].landmark
                features['gaze'] = gaze_score(landmarks, image_w, image_h)
                features['head'] = head_orientation_score(landmarks, image_w, image_h)
                l_eye = eye_openness(landmarks, L_EYE, image_w, image_h)
                r_eye = eye_openness(landmarks, R_EYE, image_w, image_h)
                features['eye'] = (l_eye + r_eye) / 2.0
                features['face_presence'] = 1.0

                # draw face mesh
                mp_drawing.draw_landmarks(frame, face_results.multi_face_landmarks[0], mp_face_mesh.FACEMESH_TESSELATION,
                                          mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1),
                                          mp_drawing.DrawingSpec(color=(0,0,255), thickness=1))

            if pose_results.pose_landmarks:
                features['posture'] = posture_score(pose_results.pose_landmarks, image_w, image_h)
                mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # apply baseline normalization
            norm_features = self.normalize_with_baseline(features)

            focus = self.compute_focus(norm_features)
            # smoothing
            self.scores_deque.append(focus)
            smooth_focus = moving_average(self.scores_deque)

            # write logging
            if self.enable_logging:
                with open(self.log_file, 'a') as f:
                    ts = time.time()
                    f.write(f"{ts},{smooth_focus},{norm_features['gaze']},{norm_features['head']},{norm_features['eye']},{norm_features['posture']},{norm_features['face_presence']}\n")

            # Alert handling
            if smooth_focus < ALERT_THRESHOLD:
                if self.alert_start_time is None:
                    self.alert_start_time = time.time()
                elif time.time() - self.alert_start_time > ALERT_SECONDS:
                    self.alert_state = True
            else:
                self.alert_start_time = None
                self.alert_state = False

            # overlay (only if not headless)
            if not headless:
                overlay_text = f"Focus: {smooth_focus:.1f}%"
                color = (0, 200, 0) if smooth_focus >= 60 else (0, 200, 255) if smooth_focus >= 40 else (0, 0, 255)
                cv2.putText(frame, overlay_text, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)

                # sub-scores
                cv2.putText(frame, f"Gaze: {norm_features['gaze']:.2f}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                cv2.putText(frame, f"Head: {norm_features['head']:.2f}", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                cv2.putText(frame, f"Eye: {norm_features['eye']:.2f}", (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                cv2.putText(frame, f"Posture: {norm_features['posture']:.2f}", (30, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

                if self.alert_state:
                    cv2.putText(frame, "ALERT: Low focus!", (30, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 3)

                cv2.imshow('Focus Detector', frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break
                elif key == ord('c'):
                    # calibrate
                    self.calibrate(CALIBRATE_SECONDS)
                elif key == ord('l'):
                    self.enable_logging = not self.enable_logging
                    print(f"Logging {'enabled' if self.enable_logging else 'disabled'}")
            else:
                # headless: print a compact per-frame summary and optionally stop after max_frames
                print(f"frame={frame_counter} focus={smooth_focus:.2f}% gaze={norm_features['gaze']:.2f} head={norm_features['head']:.2f} eye={norm_features['eye']:.2f} posture={norm_features['posture']:.2f}")

                frame_counter += 1
                if max_frames is not None and frame_counter >= max_frames:
                    break

        self.cap.release()
        cv2.destroyAllWindows()

    # New helper for external apps (e.g., Flask) to fetch a processed frame and focus data
    def get_frame(self):
        """Read one frame from the camera, process it and return a JPEG-encoded image plus focus data.

        Returns:
            (jpg_bytes, smooth_focus, norm_features) or (None, None, None) if frame read failed.
        """
        ret, frame = self.cap.read()
        if not ret:
            return None, None, None
        image_h, image_w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = self.face_mesh.process(rgb)
        pose_results = self.pose.process(rgb)

        features = {'gaze':0.5, 'head':0.5, 'eye':0.5, 'posture':0.5, 'face_presence':0.0}

        if face_results.multi_face_landmarks:
            landmarks = face_results.multi_face_landmarks[0].landmark
            features['gaze'] = gaze_score(landmarks, image_w, image_h)
            features['head'] = head_orientation_score(landmarks, image_w, image_h)
            l_eye = eye_openness(landmarks, L_EYE, image_w, image_h)
            r_eye = eye_openness(landmarks, R_EYE, image_w, image_h)
            features['eye'] = (l_eye + r_eye) / 2.0
            features['face_presence'] = 1.0

            # draw face mesh on frame for visualization
            mp_drawing.draw_landmarks(frame, face_results.multi_face_landmarks[0], mp_face_mesh.FACEMESH_TESSELATION,
                                      mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1),
                                      mp_drawing.DrawingSpec(color=(0,0,255), thickness=1))

        if pose_results.pose_landmarks:
            features['posture'] = posture_score(pose_results.pose_landmarks, image_w, image_h)
            mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        norm_features = self.normalize_with_baseline(features)
        focus = self.compute_focus(norm_features)
        # smoothing
        self.scores_deque.append(focus)
        smooth_focus = moving_average(self.scores_deque)

        # overlay minimal text
        overlay_text = f"Focus: {smooth_focus:.1f}%"
        color = (0, 200, 0) if smooth_focus >= 60 else (0, 200, 255) if smooth_focus >= 40 else (0, 0, 255)
        cv2.putText(frame, overlay_text, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        # encode as JPEG
        success, jpg = cv2.imencode('.jpg', frame)
        if not success:
            return None, None, None
        return jpg.tobytes(), smooth_focus, norm_features

    def close(self):
        """Release camera and mediapipe resources."""
        try:
            if self.cap:
                self.cap.release()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


# ------------------------- CLI -------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Real-time Focus Detector')
    parser.add_argument('--camera', type=int, default=0, help='Camera index')
    parser.add_argument('--log', action='store_true', help='Enable CSV logging')
    parser.add_argument('--baseline', type=str, default=BASELINE_FILE, help='Baseline JSON file path')
    args = parser.parse_args()

    fd = FocusDetector(camera_idx=args.camera, enable_logging=args.log, baseline_file=args.baseline)
    fd.run()