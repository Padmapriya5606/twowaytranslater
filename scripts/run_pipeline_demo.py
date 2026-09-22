"""
One-command sanity check that every stage of the pipeline imports and runs
without a trained model yet (models run in "stub mode" and print warnings
instead of crashing). Good first thing to run after `pip install -r
requirements.txt` to confirm your environment is set up correctly, before
you start training.

Usage:
    python scripts/run_pipeline_demo.py
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402


def check_landmark_extraction():
    from landmark_extraction.extract_landmarks import LandmarkExtractor, TOTAL_DIM
    ext = LandmarkExtractor()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    vec = ext.extract_from_frame(dummy_frame)
    assert vec.shape == (TOTAL_DIM,), f"expected ({TOTAL_DIM},), got {vec.shape}"
    ext.close()
    print(f"[OK] landmark extraction -> vector shape {vec.shape}")


def check_model_architectures():
    from models.cnn_lstm_sign_model import build_cnn_lstm_model
    from models.facial_expression_model import build_facial_expression_model
    from models.transformer_nlp import build_custom_transformer

    m1 = build_cnn_lstm_model(num_classes=10)
    m2 = build_facial_expression_model()
    m3 = build_custom_transformer()
    print(f"[OK] cnn_lstm_sign_model params: {m1.count_params():,}")
    print(f"[OK] facial_expression_model params: {m2.count_params():,}")
    print(f"[OK] custom transformer params: {m3.count_params():,}")


def check_dialect_selector():
    from regional_dialect.dialect_selector import DialectSelector, build_starter_vocab_files
    build_starter_vocab_files()
    sel = DialectSelector()
    sel.set_region("south")
    asset = sel.gloss_to_asset("WATER")
    print(f"[OK] dialect selector -> {asset}")


def check_avatar_assets():
    from avatar.avatar_renderer import build_starter_sign_assets
    build_starter_sign_assets()
    print("[OK] avatar starter assets generated")


if __name__ == "__main__":
    print("Running pipeline sanity checks...\n")
    check_landmark_extraction()
    check_model_architectures()
    check_dialect_selector()
    check_avatar_assets()
    print("\nAll stages import and run correctly. Next: add your dataset "
          "under data/raw/ and run data/prepare_dataset.py, then the "
          "training scripts in src/training/.")
