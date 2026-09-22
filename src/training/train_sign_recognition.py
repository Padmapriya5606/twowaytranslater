"""
Trains the upgraded Spatial-Temporal Sign Recognition Model on feature-engineered landmark sequences.

Usage:
    python src/training/train_sign_recognition.py
    python src/training/train_sign_recognition.py --epochs 80 --batch_size 32
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import keras

sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parents[1] / "landmark_extraction"))
from models.cnn_lstm_sign_model import build_cnn_lstm_model  # noqa: E402
from geometric_features import convert_sequence_to_geometric  # noqa: E402
import config  # noqa: E402


def normalize_sequence(X: np.ndarray) -> np.ndarray:
    """Converts raw (N, T, 207) landmark sequences into invariant spatial-temporal feature sequences (N, T, 164)."""
    return convert_sequence_to_geometric(X)


def load_split(split_name: str):
    csv_path = config.SPLITS_DIR / f"{split_name}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Run data/prepare_dataset.py first."
        )
    df = pd.read_csv(csv_path)
    X = np.stack([
        np.load(config.DATA_PROCESSED_DIR / row.landmark_path)
        for row in df.itertuples()
    ])
    X = normalize_sequence(X)
    return X, df["class"].tolist(), df["region"].tolist()


def augment_sequences(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Applies multi-faceted data augmentation (speed scaling, spatial jitter, Gaussian noise)."""
    N, T, D = X.shape
    X_aug_list = [X]
    y_aug_list = [y]

    # 1. Gaussian noise augmentation
    noise = np.random.normal(0, 0.015, X.shape).astype(np.float32)
    X_aug_list.append(X + noise)
    y_aug_list.append(y)

    # 2. Spatial scaling / amplitude jitter augmentation
    scale = np.random.uniform(0.85, 1.15, (N, 1, D)).astype(np.float32)
    X_aug_list.append(X * scale)
    y_aug_list.append(y)

    # 3. Temporal speed jittering (subsampling & linear interpolation)
    X_speed = np.zeros_like(X)
    for i in range(N):
        speed_factor = np.random.uniform(0.75, 1.25)
        indices = np.linspace(0, T - 1, int(T * speed_factor))
        indices = np.clip(indices, 0, T - 1).astype(int)
        resized = X[i, indices]
        if len(resized) < T:
            pad = np.zeros((T - len(resized), D), dtype=np.float32)
            resized = np.concatenate([resized, pad], axis=0)
        else:
            resized = resized[:T]
        X_speed[i] = resized

    X_aug_list.append(X_speed)
    y_aug_list.append(y)

    X_final = np.concatenate(X_aug_list, axis=0)
    y_final = np.concatenate(y_aug_list, axis=0)

    # Shuffle augmented dataset
    perm = np.random.permutation(len(X_final))
    return X_final[perm], y_final[perm]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=config.EPOCHS_SIGN)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LR_SIGN)
    args = parser.parse_args()

    X_train, y_train_labels, _ = load_split("train")
    X_val, y_val_labels, _ = load_split("val")

    classes = sorted(set(y_train_labels) | set(y_val_labels))
    class_to_idx = {c: i for i, c in enumerate(classes)}

    y_train = np.array([class_to_idx[c] for c in y_train_labels])
    y_val = np.array([class_to_idx[c] for c in y_val_labels])

    label_map_path = config.CHECKPOINT_DIR / "sign_label_map.json"
    with open(label_map_path, "w") as f:
        json.dump(class_to_idx, f, indent=2)
    print(f"Saved label map ({len(classes)} classes) -> {label_map_path}")

    # Perform multi-faceted data augmentation
    X_train_aug, y_train_aug = augment_sequences(X_train, y_train)
    print(f"Augmented training set: {X_train.shape[0]} -> {X_train_aug.shape[0]} samples")

    model = build_cnn_lstm_model(
        seq_len=config.SEQ_LEN,
        landmark_dim=X_train.shape[2],
        num_classes=len(classes),
    )
    
    # Use AdamW optimizer with learning rate & label smoothing
    initial_lr = args.lr
    optimizer = keras.optimizers.AdamW(learning_rate=initial_lr, weight_decay=1e-4)
    loss = keras.losses.CategoricalCrossentropy(label_smoothing=0.1, from_logits=False)
    
    # One-hot encode targets for label smoothing
    y_train_aug_onehot = keras.utils.to_categorical(y_train_aug, num_classes=len(classes))
    y_val_onehot = keras.utils.to_categorical(y_val, num_classes=len(classes))
    
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=["accuracy"],
    )
    model.summary()

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(config.SIGN_MODEL_CKPT), save_best_only=True, monitor="val_accuracy"
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=25, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy", factor=0.5, patience=8, min_lr=1e-5, verbose=1
        )
    ]

    history = model.fit(
        X_train_aug, y_train_aug_onehot,
        validation_data=(X_val, y_val_onehot),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
    )

    best_val_acc = max(history.history["val_accuracy"])
    print(f"\nBest Validation Accuracy Reached: {best_val_acc * 100:.2f}%")

    model.save(config.SIGN_MODEL_CKPT)
    print(f"Saved trained model -> {config.SIGN_MODEL_CKPT}")

    X_test, y_test_labels, _ = load_split("test")
    test_indices = [i for i, c in enumerate(y_test_labels) if c in class_to_idx]
    if test_indices:
        X_test_eval = X_test[test_indices]
        y_test_eval = np.array([class_to_idx[y_test_labels[i]] for i in test_indices])
        y_test_eval_onehot = keras.utils.to_categorical(y_test_eval, num_classes=len(classes))
        test_loss, test_acc = model.evaluate(X_test_eval, y_test_eval_onehot, verbose=0)
        print(f"\n" + "=" * 60)
        print(f"UPGRADED SIGN RECOGNITION MODEL TEST ACCURACY: {test_acc * 100:.2f}%")
        print("=" * 60)
        print(f"Test Loss: {test_loss:.4f}")


if __name__ == "__main__":
    main()


