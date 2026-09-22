"""Central config so every training script agrees on paths/hyperparams.
Edit this file to point at wherever your downloaded dataset actually lives.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# --- Data paths -------------------------------------------------------
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
LANDMARKS_DIR = DATA_PROCESSED_DIR / "landmarks"
LABELS_CSV = DATA_PROCESSED_DIR / "labels.csv"
SPLITS_DIR = DATA_PROCESSED_DIR / "splits"

FER2013_CSV = DATA_RAW_DIR / "fer2013" / "fer2013.csv"
ISL_SENTENCE_PAIRS_CSV = DATA_RAW_DIR / "isl_sentences" / "isl_gloss_english_pairs.csv"

# --- Output / checkpoint paths -----------------------------------------
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
SIGN_MODEL_CKPT = CHECKPOINT_DIR / "cnn_lstm_sign_model.h5"
FACE_MODEL_CKPT = CHECKPOINT_DIR / "facial_expression_model.h5"
NLP_MODEL_DIR = CHECKPOINT_DIR / "transformer_nlp"

TFLITE_DIR = PROJECT_ROOT / "tflite_models"

# --- Sign recognition hyperparams --------------------------------------
SEQ_LEN = 90              # frames per clip (~3s @ 30fps), matches extract_landmarks default
LANDMARK_DIM = 164
BATCH_SIZE = 32
EPOCHS_SIGN = 150
LR_SIGN = 1e-3

# --- Facial expression hyperparams -------------------------------------
EPOCHS_FACE = 80
LR_FACE = 5e-4
BATCH_SIZE_FACE = 64

# --- Transformer NLP hyperparams ----------------------------------------
EPOCHS_NLP = 15
LR_NLP = 5e-5              # for fine-tuning t5-small
MAX_GLOSS_LEN = 32

# --- Regions supported by the dialect selector --------------------------
REGIONS = ["south", "north", "maharashtra", "generic"]

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
TFLITE_DIR.mkdir(parents=True, exist_ok=True)
