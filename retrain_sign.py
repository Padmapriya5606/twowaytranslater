import numpy as np
import pandas as pd
import keras
import json
from pathlib import Path

print("Loading data...")
DATA_DIR = Path('data/processed')

with open('checkpoints/sign_label_map.json') as f:
    label_map = json.load(f)
num_classes = len(label_map)

def load_split(name):
    df = pd.read_csv(DATA_DIR / 'splits' / f'{name}.csv')
    X = np.stack([np.load(DATA_DIR / r.landmark_path) for r in df.itertuples()])
    y = np.array([label_map[c] for c in df['class']])
    return X, y

X_train, y_train = load_split('train')
X_val,   y_val   = load_split('val')
X_test,  y_test  = load_split('test')

print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
print(f"Input shape: {X_train.shape}")
print(f"Classes: {num_classes}")

SEQ_LEN     = X_train.shape[1]
FEATURE_DIM = X_train.shape[2]

model = keras.Sequential([
    keras.layers.Input(shape=(SEQ_LEN, FEATURE_DIM)),

    keras.layers.Conv1D(64, 3, activation='relu', padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.Conv1D(128, 3, activation='relu', padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling1D(2),
    keras.layers.Dropout(0.3),

    keras.layers.LSTM(256, return_sequences=True, dropout=0.3),
    keras.layers.LSTM(128, return_sequences=False, dropout=0.3),

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
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=callbacks
)

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nFinal Test Accuracy: {acc*100:.2f}%")
print(f"Final Test Loss:     {loss:.4f}")
print("Saved to checkpoints/cnn_lstm_sign_model.h5")