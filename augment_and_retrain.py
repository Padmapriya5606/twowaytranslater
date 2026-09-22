"""
SignWave — Data Augmentation + Retrain
Multiplies existing landmark data 5x using augmentation techniques.
Expected accuracy improvement: 5-10% -> 40-60%
"""

import numpy as np
import pandas as pd
import keras
import json
from pathlib import Path

print("="*60)
print("  SignWave — Augmentation + Retrain")
print("="*60)

DATA_DIR = Path('data/processed')

# ── Load label map ────────────────────────────────────
with open('checkpoints/sign_label_map.json') as f:
    label_map = json.load(f)
num_classes = len(label_map)
print(f"Classes: {num_classes}")

# ── Load data ─────────────────────────────────────────
def load_split(name):
    df = pd.read_csv(DATA_DIR / 'splits' / f'{name}.csv')
    X, y = [], []
    for row in df.itertuples():
        try:
            arr = np.load(DATA_DIR / row.landmark_path)
            X.append(arr)
            y.append(label_map[row.landmark_path.split('/')[1]
                               if '/' in row.landmark_path
                               else row._asdict().get('class', '')])
        except:
            pass
    return np.array(X), np.array(y)

print("Loading splits...")
df_train = pd.read_csv(DATA_DIR / 'splits' / 'train.csv')
df_val   = pd.read_csv(DATA_DIR / 'splits' / 'val.csv')
df_test  = pd.read_csv(DATA_DIR / 'splits' / 'test.csv')

def load_df(df):
    X, y = [], []
    for row in df.itertuples():
        try:
            arr = np.load(DATA_DIR / row.landmark_path)
            X.append(arr)
            y.append(label_map[row.__dict__.get('class',
                     getattr(row, 'class', None))])
        except:
            pass
    return np.array(X), np.array(y)

# Load using class column directly
def load_df2(df):
    X, y = [], []
    for _, row in df.iterrows():
        try:
            arr = np.load(DATA_DIR / row['landmark_path'])
            X.append(arr)
            y.append(label_map[row['class']])
        except:
            pass
    return np.array(X), np.array(y)

X_train, y_train = load_df2(df_train)
X_val,   y_val   = load_df2(df_val)
X_test,  y_test  = load_df2(df_test)

print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
print(f"Input shape: {X_train.shape}")

SEQ_LEN     = X_train.shape[1]
FEATURE_DIM = X_train.shape[2]

# ── Augmentation functions ────────────────────────────
def add_noise(seq, sigma=0.01):
    """Add small gaussian noise to landmarks."""
    return seq + np.random.normal(0, sigma, seq.shape)

def time_warp(seq, factor=0.1):
    """Slightly speed up or slow down the sequence."""
    T = seq.shape[0]
    stretch = int(T * (1 + np.random.uniform(-factor, factor)))
    stretch = max(10, stretch)
    indices = np.linspace(0, T-1, stretch).astype(int)
    warped  = seq[indices]
    # Resize back to original length
    out = np.zeros_like(seq)
    idx = np.linspace(0, len(warped)-1, T).astype(int)
    out = warped[idx]
    return out

def scale_landmarks(seq, factor_range=(0.9, 1.1)):
    """Slightly scale the landmark values."""
    factor = np.random.uniform(*factor_range)
    return seq * factor

def mirror_hands(seq):
    """Flip left/right hand landmarks."""
    flipped = seq.copy()
    # First 63 = left hand, next 63 = right hand
    flipped[:, :63]  = seq[:, 63:126]
    flipped[:, 63:126] = seq[:, :63]
    # Mirror x coordinates (negate x which is first of each triplet)
    for i in range(0, 63, 3):
        flipped[:, i]    = 1.0 - flipped[:, i]
        flipped[:, i+63] = 1.0 - flipped[:, i+63]
    return flipped

def dropout_frames(seq, drop_rate=0.1):
    """Randomly zero out some frames."""
    mask = np.random.random(seq.shape[0]) > drop_rate
    result = seq.copy()
    result[~mask] = 0
    return result

def augment_sequence(seq):
    """Apply random combination of augmentations."""
    aug = seq.copy()
    if np.random.random() > 0.3:
        aug = add_noise(aug)
    if np.random.random() > 0.5:
        aug = time_warp(aug)
    if np.random.random() > 0.4:
        aug = scale_landmarks(aug)
    if np.random.random() > 0.6:
        aug = mirror_hands(aug)
    if np.random.random() > 0.7:
        aug = dropout_frames(aug)
    return aug

# ── Augment training data ─────────────────────────────
print("\nAugmenting training data (5x)...")
X_aug = [X_train]
y_aug = [y_train]

for i in range(4):  # 4 augmented copies + original = 5x
    print(f"  Augmentation pass {i+1}/4...")
    X_new = np.array([augment_sequence(x) for x in X_train])
    X_aug.append(X_new)
    y_aug.append(y_train)

X_train_aug = np.concatenate(X_aug, axis=0)
y_train_aug = np.concatenate(y_aug, axis=0)

# Shuffle
idx = np.random.permutation(len(X_train_aug))
X_train_aug = X_train_aug[idx]
y_train_aug = y_train_aug[idx]

print(f"Augmented train size: {len(X_train_aug)} (was {len(X_train)})")

# ── Build improved model ──────────────────────────────
print("\nBuilding model...")

model = keras.Sequential([
    keras.layers.Input(shape=(SEQ_LEN, FEATURE_DIM)),

    # CNN block
    keras.layers.Conv1D(64, 3, activation='relu', padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.Conv1D(64, 3, activation='relu', padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling1D(2),
    keras.layers.Dropout(0.25),

    keras.layers.Conv1D(128, 3, activation='relu', padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.Conv1D(128, 3, activation='relu', padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling1D(2),
    keras.layers.Dropout(0.25),

    # LSTM block
    keras.layers.LSTM(256, return_sequences=True, dropout=0.3),
    keras.layers.LSTM(128, return_sequences=True,  dropout=0.3),
    keras.layers.LSTM(64,  return_sequences=False, dropout=0.3),

    # Classifier
    keras.layers.Dense(256, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint(
        'checkpoints/cnn_lstm_sign_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
]

print("\nStarting training...")
model.fit(
    X_train_aug, y_train_aug,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=callbacks
)

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nFinal Test Accuracy : {acc*100:.2f}%")
print(f"Final Test Loss     : {loss:.4f}")
print(f"Total classes       : {num_classes}")
print("Saved to checkpoints/cnn_lstm_sign_model.h5")