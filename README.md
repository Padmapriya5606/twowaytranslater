# ISL Two-Way Translator — Offline Android App

A real-time, two-way Indian Sign Language (ISL) ↔ Speech translator that runs
fully offline on a basic Android phone. Deaf-mute person signs → app speaks.
Hearing person speaks → app shows an animated avatar signing back.

This repo is the **full project scaffold**: dataset pipeline, model
architectures, training scripts, TFLite conversion, inference pipeline,
regional dialect layer, and an Android app skeleton ready to receive the
trained `.tflite` models. You said you've already downloaded the dataset —
next step is `python src/training/train_sign_recognition.py` once you point
it at your data (see `data/README.md`).

## Project layout

```
isl_two_way_translator/
├── data/                       # put your downloaded dataset here
│   ├── README.md               # expected folder structure per dataset
│   └── prepare_dataset.py      # converts raw videos -> landmark .npy sequences
├── src/
│   ├── landmark_extraction/
│   │   └── extract_landmarks.py    # MediaPipe Holistic: 21 pts/hand + 468 face pts
│   ├── models/
│   │   ├── cnn_lstm_sign_model.py  # CNN (spatial) + LSTM (temporal) sign classifier
│   │   ├── facial_expression_model.py  # CNN on FER2013 for ISL non-manual grammar
│   │   ├── transformer_nlp.py      # BERT-style gloss -> fluent English/ISL-gloss
│   │   └── model_utils.py
│   ├── training/
│   │   ├── config.py
│   │   ├── train_sign_recognition.py
│   │   ├── train_facial_expression.py
│   │   └── train_transformer_nlp.py
│   ├── inference/
│   │   ├── realtime_pipeline.py    # stage 1-6 wired together, webcam demo
│   │   ├── sign_to_speech.py
│   │   └── speech_to_sign.py
│   ├── tflite_conversion/
│   │   └── convert_to_tflite.py    # Keras -> TFLite (+ int8 quantization)
│   ├── regional_dialect/
│   │   └── dialect_selector.py     # South / North / Maharashtra vocab routing
│   └── avatar/
│       └── avatar_renderer.py      # skeleton-based 2D sign avatar (matplotlib/pygame)
├── android_app/                # Android Studio project skeleton
│   ├── README.md               # how to drop in the .tflite models
│   └── app/...
├── scripts/
│   └── run_pipeline_demo.py    # one-command end-to-end demo
└── requirements.txt
```

## Pipeline (matches the 6-stage design)

1. **Camera capture** → raw frames (handled in `realtime_pipeline.py`)
2. **MediaPipe Holistic** → 21 hand landmarks × 2 hands + 468 face landmarks
   (`landmark_extraction/extract_landmarks.py`)
3. **CNN (spatial) + LSTM (temporal)** → sign token sequence, plus a
   **FER2013-trained CNN** → facial expression / non-manual grammar
   (`models/cnn_lstm_sign_model.py`, `models/facial_expression_model.py`)
4. **Regional dialect layer** → picks the right vocab table for
   South / North / Maharashtra before decoding tokens
   (`regional_dialect/dialect_selector.py`)
5. **Transformer NLP** → corrects broken token sequence into a fluent sentence
   (`models/transformer_nlp.py`)
6. **TTS** → speaks the sentence (`inference/sign_to_speech.py`)

Reverse direction: **STT → NLP-to-gloss mapper → avatar renderer**
(`inference/speech_to_sign.py`, `avatar/avatar_renderer.py`).

Everything is exported to **TensorFlow Lite** (`tflite_conversion/`) so it
runs on-device with no internet, matching stage 6 of the design.

## What's real vs. what's a stub right now

- **Real, runnable code**: landmark extraction, model architectures, training
  loops, data loaders, TFLite conversion, regional vocab routing, TTS/STT
  wiring (via `pyttsx3` / `SpeechRecognition` for the Python-side demo).
- **Stub / placeholder**: actual trained weights (you train these — that's
  the next step), the avatar is a simple 2D skeletal stick-figure renderer
  (swap in a proper rigged 3D avatar later), and the Android app is a
  skeleton — Kotlin activity + Gradle files wired to load your `.tflite`
  files via the TFLite Android Support Library, but the UI is minimal.

## Next steps (once you're ready to train)

1. Drop your dataset into `data/` per the layout in `data/README.md`.
2. `pip install -r requirements.txt`
3. `python data/prepare_dataset.py --dataset_dir data/raw --out data/processed`
4. `python src/training/train_sign_recognition.py`
5. `python src/training/train_facial_expression.py` (only if using FER2013 directly)
6. `python src/training/train_transformer_nlp.py`
7. `python src/tflite_conversion/convert_to_tflite.py`
8. Copy the resulting `.tflite` files into
   `android_app/app/src/main/assets/` and build the Android app.

See each file's docstring for exact CLI args.
