"""
Geometric & Spatial-Temporal Feature Extraction Module for High-Accuracy ISL Sign Recognition.

Converts raw (N, T, 207) MediaPipe landmark sequences into (N, T, 328) spatial-temporal invariant features:
1. Wrist-centered normalized 3D landmark coordinates (63 left + 63 right = 126 dims)
2. Spatial wrist displacement relative to face reference (3 left + 3 right = 6 dims)
3. Hand presence & inter-hand spatial metrics (2 dims)
4. Hand flex angles (15 left + 15 right = 30 dims)
5. 1st-order temporal velocity vectors d(features)/dt (164 dims)

Total output shape per frame: (328,) invariant features.
"""
import numpy as np


FINGER_INDICES = [
    [0, 1, 2, 3, 4],     # Thumb
    [0, 5, 6, 7, 8],     # Index
    [0, 9, 10, 11, 12],  # Middle
    [0, 13, 14, 15, 16], # Ring
    [0, 17, 18, 19, 20], # Pinky
]


def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """Computes angle in radians between two 3D vectors."""
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cos_angle = np.dot(v1, v2) / (n1 * n2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.arccos(cos_angle))


def extract_hand_part(hand_pts: np.ndarray):
    """Extracts normalized coordinates (63), wrist position (3), active flag (1), and joint angles (15)."""
    is_active = 0.0 if np.all(hand_pts == 0) else 1.0
    if is_active == 0.0:
        norm_coords = np.zeros(63, dtype=np.float32)
        wrist_pos = np.zeros(3, dtype=np.float32)
        angles = np.zeros(15, dtype=np.float32)
        return norm_coords, wrist_pos, is_active, angles

    wrist = hand_pts[0]
    wrist_pos = wrist.astype(np.float32)
    palm_len = np.linalg.norm(hand_pts[9] - wrist)
    if palm_len < 1e-6:
        palm_len = 1.0
    norm_coords = ((hand_pts - wrist) / palm_len).flatten().astype(np.float32)

    angles = []
    for finger in FINGER_INDICES:
        for i in range(len(finger) - 2):
            p1 = hand_pts[finger[i]]
            p2 = hand_pts[finger[i + 1]]
            p3 = hand_pts[finger[i + 2]]
            v1 = p1 - p2
            v2 = p3 - p2
            angles.append(_angle_between(v1, v2))
    angles = np.array(angles, dtype=np.float32)
    return norm_coords, wrist_pos, is_active, angles


def extract_frame_geometric_features(frame_207: np.ndarray) -> np.ndarray:
    """Converts a single (207,) landmark frame into a (164,) static geometric vector."""
    if np.all(frame_207 == 0):
        return np.zeros(164, dtype=np.float32)

    left_hand = frame_207[0:63].reshape(21, 3)
    right_hand = frame_207[63:126].reshape(21, 3)

    l_norm, l_w, l_act, l_ang = extract_hand_part(left_hand)
    r_norm, r_w, r_act, r_ang = extract_hand_part(right_hand)

    return np.concatenate([l_norm, r_norm, l_w, r_w, [l_act, r_act], l_ang, r_ang], axis=0)


def convert_sequence_to_geometric(X: np.ndarray) -> np.ndarray:
    """Converts (N, T, 207) raw landmark sequences into (N, T, 328) static + temporal velocity feature sequences."""
    N, T = X.shape[0], X.shape[1]
    out = np.zeros((N, T, 328), dtype=np.float32)
    for i in range(N):
        seq_static = np.zeros((T, 164), dtype=np.float32)
        for t in range(T):
            seq_static[t] = extract_frame_geometric_features(X[i, t])
        # Compute 1st order velocity (gradient across time)
        vel = np.gradient(seq_static, axis=0)
        out[i] = np.concatenate([seq_static, vel], axis=1)
    return out


