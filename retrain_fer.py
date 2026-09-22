import numpy as np
import keras
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

print("Loading data...")
X_train = np.load('data/processed/fer/X_train.npy')
X_val   = np.load('data/processed/fer/X_val.npy')
X_test  = np.load('data/processed/fer/X_test.npy')
y_train = to_categorical(np.load('data/processed/fer/y_train.npy'), 7)
y_val   = to_categorical(np.load('data/processed/fer/y_val.npy'),   7)
y_test  = to_categorical(np.load('data/processed/fer/y_test.npy'),  7)
print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

model = keras.Sequential([
    keras.layers.Conv2D(32,(3,3), activation='relu', padding='same', input_shape=(48,48,1)),
    keras.layers.Conv2D(32,(3,3), activation='relu', padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Dropout(0.2),

    keras.layers.Conv2D(64,(3,3), activation='relu', padding='same'),
    keras.layers.Conv2D(64,(3,3), activation='relu', padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Dropout(0.3),

    keras.layers.Conv2D(128,(3,3), activation='relu', padding='same'),
    keras.layers.Conv2D(128,(3,3), activation='relu', padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Dropout(0.3),

    keras.layers.Conv2D(256,(3,3), activation='relu', padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.GlobalAveragePooling2D(),

    keras.layers.Dense(512, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(7, activation='softmax')
])

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

dg = ImageDataGenerator(
    rotation_range=15,
    horizontal_flip=True,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1
)
dg.fit(X_train)

callbacks = [
    keras.callbacks.ModelCheckpoint(
        'checkpoints/facial_expression_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=12,
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
    dg.flow(X_train, y_train, batch_size=64),
    validation_data=(X_val, y_val),
    epochs=80,
    callbacks=callbacks
)

print("\nEvaluating on test set...")
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nFinal Test Accuracy: {acc*100:.2f}%")
print(f"Final Test Loss:     {loss:.4f}")
print("Saved to checkpoints/facial_expression_model.h5")