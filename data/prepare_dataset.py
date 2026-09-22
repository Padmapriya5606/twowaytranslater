"""
Walks data/raw/sign_videos/<class>/<video>.mp4 (and data/raw/regional/<region>/<class>/<video>.mp4),
extracts MediaPipe landmark sequences for every clip, and writes:

    data/processed/landmarks/<class>/<video_id>.npy
    data/processed/labels.csv                (video_id, class, region, path)
    data/processed/splits/{train,val,test}.csv

Usage:
    python data/prepare_dataset.py --dataset_dir data/raw --out data/processed
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src" / "landmark_extraction"))
from extract_landmarks import LandmarkExtractor  # noqa: E402

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}


def find_clips(sign_videos_dir: Path):
    """Yields (video_path, class_name, region) for the generic + regional trees."""
    if sign_videos_dir.exists():
        for class_dir in sorted(sign_videos_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            for clip in sorted(class_dir.iterdir()):
                if clip.suffix.lower() in VIDEO_EXTS:
                    yield clip, class_dir.name, "generic"

    regional_dir = sign_videos_dir.parent / "regional"
    if regional_dir.exists():
        for region_dir in sorted(regional_dir.iterdir()):
            if not region_dir.is_dir():
                continue
            for class_dir in sorted(region_dir.iterdir()):
                if not class_dir.is_dir():
                    continue
                for clip in sorted(class_dir.iterdir()):
                    if clip.suffix.lower() in VIDEO_EXTS:
                        yield clip, class_dir.name, region_dir.name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default="data/raw")
    parser.add_argument("--out", default="data/processed")
    parser.add_argument("--max_frames", type=int, default=90)
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--test_frac", type=float, default=0.15)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    out_dir = Path(args.out)
    landmarks_dir = out_dir / "landmarks"
    splits_dir = out_dir / "splits"
    landmarks_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    sign_videos_dir = dataset_dir / "sign_videos"
    clips = list(find_clips(sign_videos_dir))
    if not clips:
        print(f"No video clips found under {sign_videos_dir} or {dataset_dir / 'regional'}. "
              f"Check data/README.md for the expected layout.")
        return

    extractor = LandmarkExtractor()
    rows = []
    for i, (clip_path, class_name, region) in enumerate(clips):
        video_id = f"{class_name}_{region}_{clip_path.stem}"
        class_out_dir = landmarks_dir / class_name
        class_out_dir.mkdir(parents=True, exist_ok=True)
        npy_path = class_out_dir / f"{video_id}.npy"

        seq = extractor.extract_from_video(str(clip_path), max_frames=args.max_frames)
        import numpy as np
        np.save(npy_path, seq)

        rows.append({
            "video_id": video_id,
            "class": class_name,
            "region": region,
            "landmark_path": str(npy_path.relative_to(out_dir)),
        })
        if (i + 1) % 25 == 0:
            print(f"Processed {i + 1}/{len(clips)} clips...")

    extractor.close()

    labels_csv = out_dir / "labels.csv"
    with open(labels_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video_id", "class", "region", "landmark_path"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {labels_csv}")

    # Simple stratified-ish split by shuffling within each class
    import random
    random.seed(42)
    by_class = {}
    for r in rows:
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
