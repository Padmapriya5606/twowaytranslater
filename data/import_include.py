"""
Importer for Dataset #1 — INCLUDE (kaggle.com/datasets/kaushikyh/indian-sign-language-words-with-landmarks)

This is your PRIMARY word-level sign recognition dataset (263 word classes)
and it ALREADY has MediaPipe hand + pose landmarks extracted — so unlike
the generic data/prepare_dataset.py path, this importer does NOT re-run
MediaPipe. It reshapes whatever landmark files Kaggle gives you into the
project's standard (T, 258) format used by src/models/cnn_lstm_sign_model.py
and writes them straight into data/processed/landmarks/.

IMPORTANT: Kaggle dataset internal layouts change between uploads, and this
particular dataset's description says "Only 80 word signs are uploaded for
now, more will be updated" — so the exact file layout you get may differ
from what's below. Run this FIRST:

    python data/inspect_dataset.py --path data/raw/include_raw

...and compare what it prints against the assumptions in `find_landmark_files()`
below. If the class-folder-per-word layout doesn't match, that's the one
function to edit — everything downstream (reshape_to_258, npy writing,
labels.csv, splits) stays the same.

Expected layout this script assumes (edit find_landmark_files if different):
    data/raw/include_raw/<word_class>/<video_id>.npy   (or .csv/.json)
    each file: hand landmarks (21 pts x 2 hands x 3) + pose landmarks

Usage:
    python data/import_include.py --src data/raw/include_raw --out data/processed
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "src" / "landmark_extraction"))
from extract_landmarks import TOTAL_DIM, HAND_DIM, FACE_DIM  # noqa: E402


def find_landmark_files(src_root: Path):
    """Yields (file_path, class_name, video_id). Assumes one folder per
    word class, one landmark file per video inside it. Adjust this function
    if your unzip has a different structure (check with inspect_dataset.py)."""
    for class_dir in sorted(src_root.iterdir()):
        if not class_dir.is_dir():
            continue
        for f in sorted(class_dir.iterdir()):
            if f.suffix.lower() in (".npy", ".csv", ".json"):
                yield f, class_dir.name, f.stem


def load_any(path: Path) -> np.ndarray:
    """Loads a landmark file regardless of format and returns a (T, D) array
    where D is whatever the source dataset used (commonly 21*3*2 for hands
    only, or larger if pose is included)."""
    if path.suffix == ".npy":
        arr = np.load(path, allow_pickle=True)
        return np.asarray(arr, dtype=np.float32)
    if path.suffix == ".csv":
        import pandas as pd
        df = pd.read_csv(path)
        return df.values.astype(np.float32)
    if path.suffix == ".json":
        import json
        with open(path) as f:
            data = json.load(f)
        # Common shape: list of frames, each frame a list/dict of landmark coords
        if isinstance(data, list):
            frames = []
            for frame in data:
                if isinstance(frame, dict):
                    vals = []
                    for v in frame.values():
                        vals.extend(v if isinstance(v, (list, tuple)) else [v])
                    frames.append(vals)
                else:
                    frames.append(frame)
            return np.array(frames, dtype=np.float32)
        raise ValueError(f"Unrecognized JSON landmark structure in {path}")
    raise ValueError(f"Unsupported file type: {path}")


def reshape_to_project_format(raw: np.ndarray, max_frames: int = 90) -> np.ndarray:
    """Maps whatever dimensionality the source landmarks have onto our
    standard TOTAL_DIM (258 = 63 left hand + 63 right hand + 69 face
    subset). If the source only has hands (no face), we zero-pad the face
    slice — the model still trains fine on hands-only signal, and you can
    backfill facial-grammar cues later from the FER2013-trained model
    running live at inference time instead."""
    T = raw.shape[0]
    D = raw.shape[1] if raw.ndim > 1 else 0

    out = np.zeros((max_frames, TOTAL_DIM), dtype=np.float32)
    usable_T = min(T, max_frames)
    usable_D = min(D, TOTAL_DIM)
    if usable_T > 0 and usable_D > 0:
        out[:usable_T, :usable_D] = raw[:usable_T, :usable_D]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="data/raw/include_raw")
    parser.add_argument("--out", default="data/processed")
    parser.add_argument("--max_frames", type=int, default=90)
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--test_frac", type=float, default=0.15)
    args = parser.parse_args()

    src_root = Path(args.src)
    out_dir = Path(args.out)
    landmarks_dir = out_dir / "landmarks"
    splits_dir = out_dir / "splits"
    landmarks_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    if not src_root.exists():
        print(f"{src_root} not found. Unzip the INCLUDE dataset there first, "
              f"then run: python data/inspect_dataset.py --path {src_root}")
        return

    files = list(find_landmark_files(src_root))
    if not files:
        print(f"No landmark files found under {src_root}. Run "
              f"'python data/inspect_dataset.py --path {src_root}' and check "
              f"the actual folder layout — you likely need to tweak "
              f"find_landmark_files() in this script to match it.")
        return

    rows = []
    skipped = 0
    for i, (fpath, class_name, video_id) in enumerate(files):
        try:
            raw = load_any(fpath)
        except Exception as e:
            print(f"[skip] {fpath}: {e}")
            skipped += 1
            continue

        seq = reshape_to_project_format(raw, max_frames=args.max_frames)

        class_out_dir = landmarks_dir / class_name
        class_out_dir.mkdir(parents=True, exist_ok=True)
        out_id = f"{class_name}_include_{video_id}"
        npy_path = class_out_dir / f"{out_id}.npy"
        np.save(npy_path, seq)

        rows.append({
            "video_id": out_id,
            "class": class_name,
            "region": "generic",
            "landmark_path": str(npy_path.relative_to(out_dir)),
        })
        if (i + 1) % 200 == 0:
            print(f"Processed {i + 1}/{len(files)} clips...")

    print(f"Imported {len(rows)} clips ({skipped} skipped due to load errors).")

    # Append to existing labels.csv if present (so this can run alongside
    # data/prepare_dataset.py output rather than overwriting it), else create.
    labels_csv = out_dir / "labels.csv"
    write_header = not labels_csv.exists()
    with open(labels_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video_id", "class", "region", "landmark_path"])
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    print(f"Appended to {labels_csv}")

    # Re-split (regenerates train/val/test from the FULL current labels.csv
    # each time an importer runs, so INCLUDE clips get mixed properly across
    # splits rather than all landing in one split).
    import random
    random.seed(42)
    all_rows = list(csv.DictReader(open(labels_csv)))
    by_class = {}
    for r in all_rows:
        by_class.setdefault(r["class"], []).append(r)

    train_rows, val_rows, test_rows = [], [], []
    for cls, items in by_class.items():
        random.shuffle(items)
        n = len(items)
        n_val = max(1, int(n * args.val_frac)) if n > 2 else 0
        n_test = max(1, int(n * args.test_frac)) if n > 2 else 0
        val_rows.extend(items[:n_val])
        test_rows.extend(items[n_val:n_val + n_test])
        train_rows.extend(items[n_val + n_test:])

    for name, split_rows in [("train", train_rows), ("val", val_rows), ("test", test_rows)]:
        path = splits_dir / f"{name}.csv"
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["video_id", "class", "region", "landmark_path"])
            writer.writeheader()
            writer.writerows(split_rows)
        print(f"{name}: {len(split_rows)} clips -> {path}")


if __name__ == "__main__":
    main()
