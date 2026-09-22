"""
Importer for Dataset #3 — ISL-CSLTR (kaggle.com/datasets/drblack00/isl-csltr-indian-sign-language-dataset)

700 fully annotated sentence-level videos (100 spoken-language sentences,
7 signers) + word-level images. Unlike INCLUDE, this one does NOT ship
pre-extracted landmarks — you get raw video/image files with a
sentence-per-video annotation, so this importer DOES run MediaPipe
(reuses LandmarkExtractor, same as the generic data/prepare_dataset.py).

Two things this dataset is genuinely useful for in your pipeline:
    1. Continuous/coarticulated signing practice for the CNN-LSTM model
       (signs flowing into each other, not isolated) — treat each sentence
       video as one training clip if you want the model to also learn
       continuous motion, in addition to INCLUDE's isolated-word clips.
    2. A second, independently-sourced source of gloss/sentence pairs for
       the transformer NLP stage (via the same pseudo-gloss approach used
       for ISLTranslate in import_isltranslate.py, OR directly if the
       Kaggle annotation CSV already gives you a word-by-word gloss column
       — check with inspect_dataset.py, annotation formats vary by upload).

Exact internal folder names vary between the Kaggle upload and the
Mendeley source archive. Run this FIRST:
    python data/inspect_dataset.py --path data/raw/isl_csltr_raw

This script assumes (edit `find_sentence_videos()` if your layout differs):
    data/raw/isl_csltr_raw/
        <some_annotation>.csv         # video filename/id -> sentence text
        <video files somewhere under here, .mp4/.avi>

Usage:
    python data/import_isl_csltr.py --src data/raw/isl_csltr_raw --out data/processed
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "src" / "landmark_extraction"))
from extract_landmarks import LandmarkExtractor  # noqa: E402

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def find_annotation_csv(src_root: Path):
    """Looks for the sentence-annotation CSV/Excel wherever it is in the
    tree. If your download has multiple CSVs (e.g. separate word-level and
    sentence-level annotation files), check inspect_dataset.py's output and
    hardcode the right one here."""
    candidates = list(src_root.rglob("*.csv"))
    if not candidates:
        return None
    # Prefer one with "sentence" in the name if present
    for c in candidates:
        if "sentence" in c.name.lower():
            return c
    return candidates[0]


def find_video_for_row(src_root: Path, video_ref: str):
    """video_ref might be a bare filename, a stem, or a relative path —
    try a few reasonable matches."""
    direct = src_root / video_ref
    if direct.exists():
        return direct

    stem = Path(video_ref).stem
    for ext in VIDEO_EXTS:
        matches = list(src_root.rglob(f"{stem}{ext}"))
        if matches:
            return matches[0]
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="data/raw/isl_csltr_raw")
    parser.add_argument("--out", default="data/processed")
    parser.add_argument("--max_frames", type=int, default=150)  # sentences run longer than single words
    parser.add_argument("--video_col", default=None,
                         help="Column name holding the video filename/id, if auto-detect guesses wrong")
    parser.add_argument("--sentence_col", default=None,
                         help="Column name holding the English sentence text, if auto-detect guesses wrong")
    args = parser.parse_args()

    src_root = Path(args.src)
    out_dir = Path(args.out)
    if not src_root.exists():
        print(f"{src_root} not found. Unzip ISL-CSLTR there first, then run "
              f"python data/inspect_dataset.py --path {src_root}")
        return

    ann_csv = find_annotation_csv(src_root)
    if ann_csv is None:
        print(f"No annotation CSV found under {src_root}. Check "
              f"inspect_dataset.py output — the annotation might be an .xlsx "
              f"instead (convert to .csv first) or nested inside a subfolder "
              f"this glob didn't reach.")
        return
    print(f"Using annotation file: {ann_csv}")

    with open(ann_csv, newline="", errors="replace") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    video_col = args.video_col or next(
        (c for c in fieldnames if "video" in c.lower() or "file" in c.lower() or "name" in c.lower()),
        fieldnames[0] if fieldnames else None,
    )
    sentence_col = args.sentence_col or next(
        (c for c in fieldnames if "sentence" in c.lower() or "text" in c.lower() or "label" in c.lower()),
        fieldnames[-1] if fieldnames else None,
    )
    print(f"Guessed columns -> video: '{video_col}', sentence: '{sentence_col}'. "
          f"If wrong, re-run with --video_col / --sentence_col.")

    sentences_dir = out_dir.parent / "raw" / "isl_sentences"
    sentences_dir.mkdir(parents=True, exist_ok=True)
    sentence_pairs_out = sentences_dir / "isl_csltr_sentence_pairs.csv"

    landmarks_out_dir = out_dir / "landmarks" / "_isl_csltr_sentences"
    landmarks_out_dir.mkdir(parents=True, exist_ok=True)

    extractor = LandmarkExtractor()
    pair_rows = []
    n_ok, n_missing = 0, 0

    for i, row in enumerate(rows):
        video_ref = row.get(video_col, "")
        sentence = row.get(sentence_col, "")
        if not video_ref or not sentence:
            continue

        video_path = find_video_for_row(src_root, video_ref)
        if video_path is None:
            n_missing += 1
            continue

        seq = extractor.extract_from_video(str(video_path), max_frames=args.max_frames)
        video_id = Path(video_ref).stem
        npy_path = landmarks_out_dir / f"{video_id}.npy"
        np.save(npy_path, seq)

        pair_rows.append({
            "video_id": video_id,
            "landmark_path": str(npy_path.relative_to(out_dir)),
            "english_sentence": sentence,
        })
        n_ok += 1
        if n_ok % 50 == 0:
            print(f"Processed {n_ok} sentence videos...")

    extractor.close()

    with open(sentence_pairs_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video_id", "landmark_path", "english_sentence"])
        writer.writeheader()
        writer.writerows(pair_rows)

    print(f"\nDone. {n_ok} videos processed, {n_missing} rows had no matching video file.")
    print(f"Sentence pairs -> {sentence_pairs_out}")
    print("Next: run this through the same pseudo-gloss step as ISLTranslate "
          "(see import_isltranslate.py --mode pseudo_gloss) once your sign "
          "model is trained, to turn these into gloss->English training pairs "
          "for the transformer.")


if __name__ == "__main__":
    main()
