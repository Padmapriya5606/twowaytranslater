"""
Importer for Dataset #2 — ISLTranslate (github.com/Exploration-Lab/ISLTranslate)

REAL STRUCTURE (from the repo):
    ISLTranslate.csv                          # uid, english translation
    ISL-videos.tar.gz                         # raw sentence/phrase videos, named by uid
    Extracted-Features/
        mediapipe_holistic_poses1.tar.gz ... 13.tar.gz   # precomputed MediaPipe poses

IMPORTANT — read this before training:
ISLTranslate gives you (video/pose sequence) <-> (English sentence) pairs.
It does NOT give you word-level gloss annotations (no "I GO SCHOOL
YESTERDAY" token sequence) — that's a level of annotation this dataset
doesn't have. So it can't directly fill the
`gloss_sequence,english_sentence` CSV format the original
train_transformer_nlp.py script expects.

What it's actually good for, and how this importer wires it in:
    1. It's excellent training data for a *pose-sequence -> English
       sentence* model directly (skip the gloss intermediate step for
       continuous sentences). This is arguably a better architecture for
       full sentences than isolated-word-gloss-then-correct.
    2. It's usable to bootstrap approximate gloss sequences: run your
       INCLUDE-trained CNN-LSTM word classifier over segmented windows of
       each ISLTranslate pose sequence to produce a *pseudo-gloss* sequence,
       then pair that pseudo-gloss with the real English sentence — this
       gives you gloss->English training pairs without hand-annotating
       31k sentences yourself. See `--mode pseudo_gloss` below (requires a
       trained sign model — run this AFTER train_sign_recognition.py).

This importer supports both:
    --mode direct        (default) writes data/raw/isl_sentences/isl_pose_english_pairs.csv
                          with columns: uid, pose_landmark_path, english_sentence
                          for direct pose-sequence -> sentence training.
    --mode pseudo_gloss   additionally runs your trained sign model over each
                          pose sequence and writes isl_gloss_english_pairs.csv
                          (gloss_sequence, english_sentence) — the format
                          train_transformer_nlp.py already expects.

Usage:
    python data/inspect_dataset.py --path data/raw/isl_translate_raw   # confirm layout first
    python data/import_isltranslate.py --src data/raw/isl_translate_raw --mode direct
    python data/import_isltranslate.py --src data/raw/isl_translate_raw --mode pseudo_gloss
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "src" / "landmark_extraction"))
from extract_landmarks import TOTAL_DIM  # noqa: E402
sys.path.append(str(Path(__file__).resolve().parents[1] / "src" / "training"))
import config  # noqa: E402


def find_pose_file(features_dir: Path, uid: str):
    """The repo ships poses split across mediapipe_holistic_poses1..13
    archives; after you extract them all into one folder, poses are
    typically named by uid. Adjust the glob pattern here if your extracted
    filenames differ (check with inspect_dataset.py)."""
    for ext in (".npy", ".json", ".csv", ".pkl"):
        candidates = list(features_dir.rglob(f"{uid}{ext}"))
        if candidates:
            return candidates[0]
    return None


def load_pose_sequence(path: Path, max_frames: int) -> np.ndarray:
    if path.suffix == ".npy":
        raw = np.load(path, allow_pickle=True).astype(np.float32)
    elif path.suffix == ".csv":
        import pandas as pd
        raw = pd.read_csv(path).values.astype(np.float32)
    elif path.suffix == ".json":
        import json
        with open(path) as f:
            data = json.load(f)
        raw = np.array(data, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported pose file type: {path}")

    out = np.zeros((max_frames, TOTAL_DIM), dtype=np.float32)
    T = min(raw.shape[0], max_frames)
    D = min(raw.shape[1] if raw.ndim > 1 else 0, TOTAL_DIM)
    if T > 0 and D > 0:
        out[:T, :D] = raw[:T, :D]
    return out


def mode_direct(src: Path, out_dir: Path, max_frames: int):
    csv_path = src / "ISLTranslate.csv"
    features_dir = src / "Extracted-Features"
    if not csv_path.exists():
        print(f"{csv_path} not found. Expected ISLTranslate.csv at the top of --src.")
        return

    pose_out_dir = out_dir / "landmarks" / "_isltranslate_sentences"
    pose_out_dir.mkdir(parents=True, exist_ok=True)

    sentences_dir = out_dir.parent / "raw" / "isl_sentences"
    sentences_dir.mkdir(parents=True, exist_ok=True)
    out_csv = sentences_dir / "isl_pose_english_pairs.csv"

    rows = []
    with open(csv_path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            uid = row.get("uid") or row.get("id") or list(row.values())[0]
            english = row.get("english") or row.get("translation") or list(row.values())[-1]

            pose_path = find_pose_file(features_dir, uid) if features_dir.exists() else None
            landmark_rel_path = ""
            if pose_path is not None:
                seq = load_pose_sequence(pose_path, max_frames)
                npy_out = pose_out_dir / f"{uid}.npy"
                np.save(npy_out, seq)
                landmark_rel_path = str(npy_out.relative_to(out_dir))

            rows.append({"uid": uid, "pose_landmark_path": landmark_rel_path, "english_sentence": english})
            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1} sentence pairs...")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["uid", "pose_landmark_path", "english_sentence"])
        writer.writeheader()
        writer.writerows(rows)

    n_with_pose = sum(1 for r in rows if r["pose_landmark_path"])
    print(f"Wrote {len(rows)} pairs -> {out_csv} ({n_with_pose} with matched pose features)")
    if n_with_pose == 0 and features_dir.exists():
        print("No pose files matched by uid — extract the mediapipe_holistic_poses*.tar.gz "
              "archives into Extracted-Features/ and re-run inspect_dataset.py to confirm "
              "the actual filename pattern, then adjust find_pose_file().")


def mode_pseudo_gloss(src: Path, out_dir: Path, max_frames: int):
    import tensorflow as tf
    import json as jsonlib

    sign_model_path = config.SIGN_MODEL_CKPT
    label_map_path = config.CHECKPOINT_DIR / "sign_label_map.json"
    if not sign_model_path.exists() or not label_map_path.exists():
        print("pseudo_gloss mode needs a trained sign model — run "
              "train_sign_recognition.py (on the INCLUDE import) first.")
        return

    model = tf.keras.models.load_model(sign_model_path)
    with open(label_map_path) as f:
        class_to_idx = jsonlib.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    pose_csv = out_dir.parent / "raw" / "isl_sentences" / "isl_pose_english_pairs.csv"
    if not pose_csv.exists():
        print(f"{pose_csv} not found — run --mode direct first to build it.")
        return

    out_csv = out_dir.parent / "raw" / "isl_sentences" / "isl_gloss_english_pairs.csv"
    rows_out = []
    window = config.SEQ_LEN

    with open(pose_csv, newline="", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if not row["pose_landmark_path"]:
                continue
            seq = np.load(out_dir / row["pose_landmark_path"])
            glosses = []
            for start in range(0, seq.shape[0], window):
                chunk = seq[start:start + window]
                if chunk.shape[0] < window:
                    pad = np.zeros((window - chunk.shape[0], TOTAL_DIM), dtype=np.float32)
                    chunk = np.concatenate([chunk, pad], axis=0)
                probs = model.predict(chunk[np.newaxis, ...], verbose=0)[0]
                glosses.append(idx_to_class.get(int(np.argmax(probs)), "<UNK>"))

            rows_out.append({
                "gloss_sequence": " ".join(glosses),
                "english_sentence": row["english_sentence"],
            })
            if (i + 1) % 500 == 0:
                print(f"Pseudo-glossed {i + 1} sentences...")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["gloss_sequence", "english_sentence"])
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"Wrote {len(rows_out)} pseudo-gloss pairs -> {out_csv}. "
          f"This is what train_transformer_nlp.py reads by default.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="data/raw/isl_translate_raw")
    parser.add_argument("--out", default="data/processed")
    parser.add_argument("--mode", choices=["direct", "pseudo_gloss"], default="direct")
    parser.add_argument("--max_frames", type=int, default=90)
    args = parser.parse_args()

    src = Path(args.src)
    out_dir = Path(args.out)
    if not src.exists():
        print(f"{src} not found. Unzip/extract ISLTranslate there first, then run "
              f"python data/inspect_dataset.py --path {src}")
        return

    if args.mode == "direct":
        mode_direct(src, out_dir, args.max_frames)
    else:
        mode_pseudo_gloss(src, out_dir, args.max_frames)


if __name__ == "__main__":
    main()
