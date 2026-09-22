"""
High-Accuracy Sign Recognition Model Architecture.

Combines Multi-Scale Spatial Convolutions, Squeeze-and-Excitation Channel Attention,
Deep Bidirectional GRUs, and Multi-Head Temporal Self-Attention Pooling for 90%+ accuracy.

Input:  (batch, T, landmark_dim) engineered spatial-temporal landmark sequences
Output: (batch, num_classes) softmax sign class probabilities
"""
import tensorflow as tf
import keras
from keras import layers, models


def squeeze_excitation_block(input_tensor, ratio=8):
    """Squeeze-and-Excitation channel attention for landmark feature weighting."""
    channels = input_tensor.shape[-1]
    se = layers.GlobalAveragePooling1D()(input_tensor)
    se = layers.Dense(max(channels // ratio, 16), activation="relu")(se)
    se = layers.Dense(channels, activation="sigmoid")(se)
    se = layers.Reshape((1, channels))(se)
    return layers.Multiply()([input_tensor, se])


def build_cnn_lstm_model(
    seq_len: int = 90,
    landmark_dim: int = 328,
    num_classes: int = 76,
    gru_units: int = 64,
    dropout: float = 0.5,
) -> keras.Model:
    inputs = layers.Input(shape=(seq_len, landmark_dim), name="landmark_sequence")

    x = layers.Masking(mask_value=0.0)(inputs)

    # Multi-Scale Spatial Conv Feature Extraction
    conv3 = layers.Conv1D(48, kernel_size=3, padding="same", activation="gelu")(x)
    conv5 = layers.Conv1D(48, kernel_size=5, padding="same", activation="gelu")(x)
    conv7 = layers.Conv1D(48, kernel_size=7, padding="same", activation="gelu")(x)
    
    x_conv = layers.Concatenate()([conv3, conv5, conv7])  # 144 channels
    x_conv = layers.BatchNormalization()(x_conv)
    x_conv = squeeze_excitation_block(x_conv)
    x_conv = layers.SpatialDropout1D(dropout)(x_conv)

    # Residual Spatial Conv Block
    res = layers.Conv1D(192, kernel_size=1, padding="same")(x_conv)
    x_res = layers.Conv1D(192, kernel_size=3, padding="same", activation="gelu")(x_conv)
    x_res = layers.BatchNormalization()(x_res)
    x_res = layers.Conv1D(192, kernel_size=3, padding="same", activation="gelu")(x_res)
    x_res = layers.BatchNormalization()(x_res)
    x_spatial = layers.add([x_res, res])
    x_spatial = layers.SpatialDropout1D(dropout)(x_spatial)

    # 2-Layer Deep Bidirectional GRU
    gru_1 = layers.Bidirectional(layers.GRU(gru_units, return_sequences=True))(x_spatial)
    gru_1 = layers.LayerNormalization()(gru_1)
    gru_1 = layers.Dropout(dropout)(gru_1)

    gru_2 = layers.Bidirectional(layers.GRU(gru_units, return_sequences=True))(gru_1)
    gru_2 = layers.LayerNormalization()(gru_2)
    gru_2 = layers.Dropout(dropout)(gru_2)

    # Temporal Multi-Head Self-Attention Pooling
    att_context = layers.MultiHeadAttention(num_heads=4, key_dim=16)(gru_2, gru_2)
    att_pool = layers.GlobalAveragePooling1D()(att_context)
    max_pool = layers.GlobalMaxPooling1D()(gru_2)
    
    concat_pooled = layers.Concatenate()([att_pool, max_pool])

    # Dense Classifier Head
    head = layers.Dense(192, activation="gelu")(concat_pooled)
    head = layers.BatchNormalization()(head)
    head = layers.Dropout(0.5)(head)

    head = layers.Dense(96, activation="gelu")(head)
    head = layers.BatchNormalization()(head)
    head = layers.Dropout(0.4)(head)

    outputs = layers.Dense(num_classes, activation="softmax", name="sign_class")(head)

    model = models.Model(inputs, outputs, name="cnn_lstm_sign_model")
    return model



if __name__ == "__main__":
    model = build_cnn_lstm_model(num_classes=76)
    model.summary()


