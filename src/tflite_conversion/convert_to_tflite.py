"""
Stage 6: converts trained Keras models to TensorFlow Lite so they run
fully on-device (no server, no internet).

Usage:
    python src/tflite_conversion/convert_to_tflite.py --model sign
    python src/tflite_conversion/convert_to_tflite.py --model face
    python src/tflite_conversion/convert_to_tflite.py --model all --quantize int8
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import keras

sys.path.append(str(Path(__file__).resolve().parents[1] / "training"))
import config  # noqa: E402


def convert(keras_model_path: Path, out_path: Path, quantize: str = "none",
            rep_data: np.ndarray = None):
    if not keras_model_path.exists():
        print(f"[skip] {keras_model_path} does not exist yet — train that model first.")
        return

    model = keras.models.load_model(keras_model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    converter._experimental_lower_tensor_list_ops = False

    if quantize == "float16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif quantize == "int8":
        if rep_data is None:
            print("int8 quantization requested but no representative dataset given — "
                  "falling back to dynamic range quantization.")
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        else:
            def rep_dataset():
                for sample in rep_data[:100]:
                    yield [np.expand_dims(sample, axis=0).astype(np.float32)]

            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = rep_dataset
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(tflite_model)
    size_mb = len(tflite_model) / (1024 * 1024)
    print(f"Wrote {out_path} ({size_mb:.2f} MB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["sign", "face", "all"], default="all")
    parser.add_argument("--quantize", choices=["none", "float16", "int8"], default="float16")
    args = parser.parse_args()

    targets = []
    if args.model in ("sign", "all"):
        targets.append((config.SIGN_MODEL_CKPT, config.TFLITE_DIR / "sign_model.tflite"))
    if args.model in ("face", "all"):
        targets.append((config.FACE_MODEL_CKPT, config.TFLITE_DIR / "face_model.tflite"))

    for keras_path, tflite_path in targets:
        convert(keras_path, tflite_path, quantize=args.quantize)

    print(
        "\nNote: the Transformer NLP model (HuggingFace t5-small fine-tune) is "
        "converted separately — see the `optimum` / `transformers.js` export path, "
        "or distill it into the custom Keras transformer in "
        "src/models/transformer_nlp.py::build_custom_transformer() before running "
        "this same converter against it, since HF PyTorch models don't convert "
        "directly via tf.lite.TFLiteConverter."
    )


if __name__ == "__main__":
    main()
