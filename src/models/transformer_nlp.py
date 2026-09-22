"""
Stage 5: Transformer NLP correction.

Takes a broken sign-token sequence, e.g. ["I", "go", "school", "yesterday"],
plus the facial-grammar tag (question / negation / statement / ...), and
produces a fluent sentence: "I went to school yesterday."

Two modes are provided:

1. `TransformerGlossCorrector` (recommended to start): fine-tunes a small
   pretrained seq2seq model (e.g. `t5-small`) on your ISL gloss->English
   pairs. This gets you a working, trainable correction model quickly
   without designing a transformer from scratch.

2. `build_custom_transformer()`: a from-scratch, lightweight
   encoder-decoder transformer in Keras, sized to be TFLite-convertible for
   fully on-device inference (the pretrained HF model is heavier and is
   best used server-side / for prototyping, or distilled down before
   shipping on-device).

For the reverse direction (speech -> ISL gloss), reuse the same corrector
architecture trained on the inverse pairs (english_sentence -> gloss_sequence).
"""
from dataclasses import dataclass
from typing import List

import tensorflow as tf
import keras
from keras import layers, models


# ---------------------------------------------------------------------------
# Option 1: fine-tune a small pretrained seq2seq model (fastest path to a
# working demo; use this first while you're still training the other models)
# ---------------------------------------------------------------------------
class TransformerGlossCorrector:
    """Wraps a HuggingFace seq2seq model fine-tuned on gloss<->sentence pairs."""

    def __init__(self, model_name: str = "t5-small"):
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def correct(self, gloss_tokens: List[str], grammar_tag: str = "statement") -> str:
        prompt = f"correct isl gloss ({grammar_tag}): " + " ".join(gloss_tokens)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        out_ids = self.model.generate(**inputs, max_new_tokens=32)
        return self.tokenizer.decode(out_ids[0], skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Option 2: lightweight from-scratch transformer, TFLite-friendly
# ---------------------------------------------------------------------------
@dataclass
class TransformerConfig:
    vocab_size: int = 8000
    max_len: int = 32
    d_model: int = 128
    num_heads: int = 4
    ff_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.1


class PositionalEmbedding(layers.Layer):
    def __init__(self, max_len, d_model, vocab_size, **kwargs):
        super().__init__(**kwargs)
        self.token_emb = layers.Embedding(vocab_size, d_model)
        self.pos_emb = layers.Embedding(max_len, d_model)

    def call(self, x):
        length = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        return self.token_emb(x) + self.pos_emb(positions)


def _encoder_block(x, cfg: TransformerConfig):
    attn = layers.MultiHeadAttention(num_heads=cfg.num_heads, key_dim=cfg.d_model // cfg.num_heads)(x, x)
    x = layers.LayerNormalization(epsilon=1e-6)(x + layers.Dropout(cfg.dropout)(attn))
    ff = layers.Dense(cfg.ff_dim, activation="relu")(x)
    ff = layers.Dense(cfg.d_model)(ff)
    x = layers.LayerNormalization(epsilon=1e-6)(x + layers.Dropout(cfg.dropout)(ff))
    return x


def _decoder_block(x, enc_out, cfg: TransformerConfig):
    causal_attn = layers.MultiHeadAttention(
        num_heads=cfg.num_heads, key_dim=cfg.d_model // cfg.num_heads
    )(x, x, use_causal_mask=True)
    x = layers.LayerNormalization(epsilon=1e-6)(x + layers.Dropout(cfg.dropout)(causal_attn))

    cross_attn = layers.MultiHeadAttention(
        num_heads=cfg.num_heads, key_dim=cfg.d_model // cfg.num_heads
    )(x, enc_out)
    x = layers.LayerNormalization(epsilon=1e-6)(x + layers.Dropout(cfg.dropout)(cross_attn))

    ff = layers.Dense(cfg.ff_dim, activation="relu")(x)
    ff = layers.Dense(cfg.d_model)(ff)
    x = layers.LayerNormalization(epsilon=1e-6)(x + layers.Dropout(cfg.dropout)(ff))
    return x


def build_custom_transformer(cfg: TransformerConfig = TransformerConfig()) -> keras.Model:
    """Small encoder-decoder transformer: gloss token ids -> corrected sentence token ids."""
    enc_inputs = layers.Input(shape=(cfg.max_len,), dtype="int32", name="gloss_tokens")
    dec_inputs = layers.Input(shape=(cfg.max_len,), dtype="int32", name="target_tokens_shifted")

    enc_embed = PositionalEmbedding(cfg.max_len, cfg.d_model, cfg.vocab_size)(enc_inputs)
    x = enc_embed
    for _ in range(cfg.num_layers):
        x = _encoder_block(x, cfg)
    enc_out = x

    dec_embed = PositionalEmbedding(cfg.max_len, cfg.d_model, cfg.vocab_size)(dec_inputs)
    y = dec_embed
    for _ in range(cfg.num_layers):
        y = _decoder_block(y, enc_out, cfg)

    outputs = layers.Dense(cfg.vocab_size, activation="softmax", name="token_logits")(y)
    return models.Model([enc_inputs, dec_inputs], outputs, name="isl_gloss_transformer")


if __name__ == "__main__":
    model = build_custom_transformer()
    model.summary()
