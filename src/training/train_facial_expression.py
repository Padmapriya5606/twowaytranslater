"""
Trains the FER2013 facial-expression CNN used for ISL non-manual grammar
(question marker, negation, emphasis, ...).

Usage:
    python src/training/train_facial_expression.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import keras
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

sys.path.append(str(Path(__file__).resolve().parents[1]))
from models.facial_expression_model import build_facial_expression_model, load_fer2013, EMOTION_LABELS  # noqa: E402
import config  # noqa: E402


def random_cutout(img, mask_size=8):
    """Applies random rectangular cutout to augment face images."""
    if np.random.rand() > 0.5:
        return img
    h, w, c = img.shape
    y = np.random.randint(0, h - mask_size)
    x = np.random.randint(0, w - mask_size)
    img_cut = img.copy()
    img_cut[y:y+mask_size, x:x+mask_size, :] = 0.0
    return img_cut


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=config.EPOCHS_FACE)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE_FACE)
    parser.add_argument("--lr", type=float, default=config.LR_FACE)
    parser.add_argument("--csv", default=str(config.FER2013_CSV))
    parser.add_argument("--max_samples", type=int, default=5000)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists() and not (csv_path.parent / "train").exists():
        raise FileNotFoundError(
            f"Dataset not found at {args.csv} or directory {csv_path.parent / 'train'}. "
            f"Place fer2013.csv or train/ test/ folders at data/raw/fer2013/."
        )

    X_train, y_train, X_val, y_val, X_test, y_test = load_fer2013(args.csv)
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # Compute class weights for handling class imbalance (especially disgust)
    y_train_labels = np.argmax(y_train, axis=1)
    classes = np.unique(y_train_labels)
    class_weights = compute_class_weight("balanced", classes=classes, y=y_train_labels)
    class_weight_dict = dict(zip(classes, class_weights))
    print("Class weights:", class_weight_dict)

    model = build_facial_expression_model()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"],
    )
    model.summary()

    datagen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.15,
        brightness_range=(0.8, 1.2),
        horizontal_flip=True,
        fill_mode="nearest",
        preprocessing_function=random_cutout,
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(config.FACE_MODEL_CKPT), save_best_only=True, monitor="val_accuracy"
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=12, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", patience=4, factor=0.5, min_lr=1e-6
        ),
    ]

    model.fit(
        datagen.flow(X_train, y_train, batch_size=args.batch_size),
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        callbacks=callbacks,
        class_weight=class_weight_dict,
    )

    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"\nTest Accuracy: {test_acc:.4f} (Test Loss: {test_loss:.4f})")

    # Detailed evaluation: Classification report and confusion matrix
    y_test_true = np.argmax(y_test, axis=1)
    y_test_pred_prob = model.predict(X_test)
    y_test_pred = np.argmax(y_test_pred_prob, axis=1)

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT (Per-class precision, recall, F1)")
    print("=" * 60)
    print(classification_report(y_test_true, y_test_pred, target_names=EMOTION_LABELS))

    print("=" * 60)
    print("CONFUSION MATRIX")
    print("=" * 60)
    print(confusion_matrix(y_test_true, y_test_pred))

    model.save(config.FACE_MODEL_CKPT)
    print(f"\nSaved trained model -> {config.FACE_MODEL_CKPT}")


if __name__ == "__main__":
    main()

