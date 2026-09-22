"""
Renders the animated sign avatar the deaf-mute person watches (reverse
direction: hearing person speaks -> gloss -> avatar signs it back).

This is a 2D skeletal stick-figure renderer for the Python-side demo/
prototyping — enough to validate the gloss->animation pipeline before
investing in a full rigged 3D avatar for the Android app (which would
typically use a Unity or Filament/SceneView render target, driven by the
same per-frame joint-angle keyframes this module produces).

Each regional sign asset (see regional_dialect/dialect_selector.py) is a
JSON file of keyframes: a list of frames, each frame a dict of joint name
-> (x, y) in normalized [0,1] coordinates. build_starter_sign_assets()
below creates a few illustrative placeholder keyframe files so the pipeline
runs end-to-end; replace these with real keyframes captured from your
regional video dataset (extract_landmarks.py output can be converted into
these same keyframes).
"""
import json
import time
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt

JOINTS = [
    "head", "neck", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_hand", "right_hand", "torso",
]

BONES = [
    ("head", "neck"), ("neck", "torso"),
    ("neck", "left_shoulder"), ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"), ("left_wrist", "left_hand"),
    ("neck", "right_shoulder"), ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"), ("right_wrist", "right_hand"),
]

ASSETS_DIR = Path(__file__).resolve().parents[2] / "data" / "avatar_assets"


def load_keyframes(asset_path: str) -> List[dict]:
    full_path = Path(__file__).resolve().parents[2] / asset_path
    if not full_path.exists():
        raise FileNotFoundError(f"Avatar asset not found: {full_path}")
    with open(full_path) as f:
        return json.load(f)["frames"]


def play_sign_sequence(asset_paths: List[str], fps: int = 12):
    """Plays a sequence of gloss animations back-to-back using matplotlib."""
    fig, ax = plt.subplots(figsize=(4, 5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.invert_yaxis()
    ax.axis("off")

    for asset_path in asset_paths:
        frames = load_keyframes(asset_path)
        for frame in frames:
            ax.clear()
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.invert_yaxis()
            ax.axis("off")
            for j1, j2 in BONES:
                if j1 in frame and j2 in frame:
                    x1, y1 = frame[j1]
                    x2, y2 = frame[j2]
                    ax.plot([x1, x2], [y1, y2], "o-", color="royalblue", linewidth=3)
            plt.pause(1.0 / fps)
    plt.close(fig)


def build_starter_sign_assets():
    """Generates a handful of illustrative placeholder keyframe JSONs so the
    speech->avatar pipeline is runnable before you've captured real regional
    keyframes from your dataset."""
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    base_pose = {
        "head": (0.5, 0.15), "neck": (0.5, 0.25), "torso": (0.5, 0.55),
        "left_shoulder": (0.4, 0.28), "right_shoulder": (0.6, 0.28),
        "left_elbow": (0.32, 0.4), "right_elbow": (0.68, 0.4),
        "left_wrist": (0.3, 0.55), "right_wrist": (0.7, 0.55),
        "left_hand": (0.28, 0.6), "right_hand": (0.72, 0.6),
    }

    words = {
        "hello": [{"right_hand": (0.72, 0.2)}, {"right_hand": (0.55, 0.15)}],
        "water": [{"right_hand": (0.55, 0.3)}, {"right_hand": (0.55, 0.4)}],
        "school": [{"left_hand": (0.35, 0.35)}, {"right_hand": (0.65, 0.35)}],
        "help": [{"right_hand": (0.5, 0.45)}, {"left_hand": (0.5, 0.45)}],
    }

    for word, deltas in words.items():
        frames = []
        for delta in deltas:
            frame = dict(base_pose)
            frame.update(delta)
            frames.append(frame)
        # hold the last pose briefly
        frames.append(frames[-1])

        for region_dir in ["generic", "south", "north", "maharashtra"]:
            out_dir = ASSETS_DIR.parent / "avatar_assets" / region_dir
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{word}.json"
            if not out_path.exists():
                with open(out_path, "w") as f:
                    json.dump({"word": word, "region": region_dir, "frames": frames}, f, indent=2)
    print(f"Wrote starter avatar keyframe assets under {ASSETS_DIR}")


if __name__ == "__main__":
    build_starter_sign_assets()
    print("Playing demo sequence: hello -> water (close the plot window to finish)...")
    time.sleep(0.5)
    play_sign_sequence([
        "data/avatar_assets/generic/hello.json",
        "data/avatar_assets/generic/water.json",
    ])
