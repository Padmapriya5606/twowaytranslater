"""
Stage 1-2 of the pipeline: capture frames and extract MediaPipe Holistic
landmarks (hands + a compact facial subset for non-manual grammar).

Usage as a library:
    from extract_landmarks import LandmarkExtractor
    ext = LandmarkExtractor()
    vec = ext.extract_from_frame(bgr_frame)   # -> np.ndarray shape (258,)

Usage as a script (extract every frame of a video to a .npy sequence):
    python extract_landmarks.py --video path/to/clip.mp4 --out out.npy
"""
import argparse
import numpy as np
import cv2
import mediapipe as mp

mp_holistic = mp.solutions.holistic

# A compact, high-signal subset of the 468 FaceMesh points that carry ISL
# non-manual grammar: eyebrows, eyes, mouth corners, nose tip, cheeks.
# (Using all 468 raw points is unnecessary and bloats the sequence model —
# these ~20 points capture eyebrow raise, mouth shape, and head tilt cues.)
FACE_SUBSET_IDX = [
    70, 63, 105, 66, 107,      # left eyebrow
    336, 296, 334, 293, 300,   # right eyebrow
    33, 133, 160, 158, 153,    # left eye
    362, 263, 387, 385, 380,   # right eye
    1,                          # nose tip
    61, 291, 78, 308, 13, 14,  # mouth corners / lips
]

NUM_HAND_LANDMARKS = 21
HAND_DIM = NUM_HAND_LANDMARKS * 3          # x, y, z
FACE_DIM = len(FACE_SUBSET_IDX) * 3
TOTAL_DIM = HAND_DIM * 2 + FACE_DIM        # left hand + right hand + face subset


class LandmarkExtractor:
    def __init__(self, static_image_mode=False, model_complexity=1,
                 min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.holistic = mp_holistic.Holistic(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def _hand_to_vec(self, hand_landmarks):
        if hand_landmarks is None:
            return np.zeros(HAND_DIM, dtype=np.float32)
        pts = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
            dtype=np.float32,
        )
        return pts.flatten()

    def _face_to_vec(self, face_landmarks):
        if face_landmarks is None:
            return np.zeros(FACE_DIM, dtype=np.float32)
        pts = np.array(
            [[face_landmarks.landmark[i].x,
              face_landmarks.landmark[i].y,
              face_landmarks.landmark[i].z] for i in FACE_SUBSET_IDX],
            dtype=np.float32,
        )
        return pts.flatten()

    def extract_from_frame(self, bgr_frame: np.ndarray) -> np.ndarray:
        """Returns a (258,) float32 vector: [left_hand(63) | right_hand(63) | face(69)]."""
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.holistic.process(rgb)

        left = self._hand_to_vec(results.left_hand_landmarks)
        right = self._hand_to_vec(results.right_hand_landmarks)
        face = self._face_to_vec(results.face_landmarks)
        return np.concatenate([left, right, face], axis=0)

    def extract_from_video(self, video_path: str, max_frames: int = 90) -> np.ndarray:
        """Returns (T, 258) array, T <= max_frames. Pads short clips with zeros."""
        cap = cv2.VideoCapture(video_path)
        seq = []
        while cap.isOpened() and len(seq) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            seq.append(self.extract_from_frame(frame))
        cap.release()

        if len(seq) == 0:
            return np.zeros((max_frames, TOTAL_DIM), dtype=np.float32)

        arr = np.stack(seq, axis=0)
        if arr.shape[0] < max_frames:
            pad = np.zeros((max_frames - arr.shape[0], TOTAL_DIM), dtype=np.float32)
            arr = np.concatenate([arr, pad], axis=0)
        return arr[:max_frames]

    def close(self):
        self.holistic.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max_frames", type=int, default=90)
    args = parser.parse_args()

    ext = LandmarkExtractor()
    seq = ext.extract_from_video(args.video, max_frames=args.max_frames)
    np.save(args.out, seq)
    ext.close()
    print(f"Saved landmark sequence {seq.shape} -> {args.out}")
