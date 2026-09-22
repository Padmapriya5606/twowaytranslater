"""
Forward direction: deaf-mute person signs -> app speaks for the hearing person.

Wires together: LandmarkExtractor -> CNN-LSTM sign model -> facial-grammar
tag -> Transformer NLP correction -> TTS.
"""
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np
import tensorflow as tf
import keras

sys.path.append(str(Path(__file__).resolve().parents[1]))
from landmark_extraction.extract_landmarks import LandmarkExtractor  # noqa: E402
from training import config as training_config  # from src/training  # noqa: E402


class SignToSpeechPipeline:
    def __init__(self, sign_model_path=None, label_map_path=None,
                 nlp_model_dir=None, tts_engine="pyttsx3"):
        sign_model_path = sign_model_path or training_config.SIGN_MODEL_CKPT
        label_map_path = label_map_path or training_config.CHECKPOINT_DIR / "sign_label_map.json"
        nlp_model_dir = nlp_model_dir or training_config.NLP_MODEL_DIR

        self.extractor = LandmarkExtractor()
        self.sign_model = self._load_sign_model(sign_model_path)
        self.idx_to_class = self._load_label_map(label_map_path)
        self.nlp_corrector = self._load_nlp_model(nlp_model_dir)
        self.tts = self._init_tts(tts_engine)

        self.frame_buffer = deque(maxlen=training_config.SEQ_LEN)

    def _load_sign_model(self, path):
        path = Path(path)
        if not path.exists():
            print(f"[warning] sign model not found at {path} — train it first "
                  f"(train_sign_recognition.py). Running in stub mode.")
            return None
        return keras.models.load_model(path)

    def _load_label_map(self, path):
        path = Path(path)
        if not path.exists():
            return {}
        with open(path) as f:
            class_to_idx = json.load(f)
        return {v: k for k, v in class_to_idx.items()}

    def _load_nlp_model(self, model_dir):
        model_dir = Path(model_dir)
        if not model_dir.exists():
            print(f"[warning] NLP model not found at {model_dir} — train it first "
                  f"(train_transformer_nlp.py). Falling back to raw token join.")
            return None
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
        return tokenizer, model

    def _init_tts(self, engine_name):
        if engine_name == "pyttsx3":
            import pyttsx3
            return pyttsx3.init()
        raise ValueError(f"Unsupported TTS engine: {engine_name}")

    def push_frame(self, bgr_frame: np.ndarray):
        """Feed one camera frame; call flush_and_speak() once the person
        finishes signing (e.g. on a pause/silence detector, or a fixed
        window of SEQ_LEN frames)."""
        vec = self.extractor.extract_from_frame(bgr_frame)
        self.frame_buffer.append(vec)

    def _predict_gloss_tokens(self) -> list:
        if self.sign_model is None or len(self.frame_buffer) == 0:
            return []

        seq = np.stack(list(self.frame_buffer), axis=0)
        if seq.shape[0] < training_config.SEQ_LEN:
            pad = np.zeros(
                (training_config.SEQ_LEN - seq.shape[0], training_config.LANDMARK_DIM),
                dtype=np.float32,
            )
            seq = np.concatenate([seq, pad], axis=0)
        seq = seq[np.newaxis, ...]  # batch dim

        probs = self.sign_model.predict(seq, verbose=0)[0]
        top_idx = int(np.argmax(probs))
        gloss = self.idx_to_class.get(top_idx, "<UNK>")
        return [gloss]

    def _correct_sentence(self, gloss_tokens: list) -> str:
        if not gloss_tokens:
            return ""
        if self.nlp_corrector is None:
            return " ".join(gloss_tokens).capitalize()

        tokenizer, model = self.nlp_corrector
        prompt = "correct isl gloss: " + " ".join(gloss_tokens)
        inputs = tokenizer(prompt, return_tensors="pt")
        out_ids = model.generate(**inputs, max_new_tokens=32)
        return tokenizer.decode(out_ids[0], skip_special_tokens=True)

    def flush_and_speak(self):
        gloss_tokens = self._predict_gloss_tokens()
        sentence = self._correct_sentence(gloss_tokens)
        if sentence:
            print(f"[sign->speech] gloss={gloss_tokens} -> \"{sentence}\"")
            self.tts.say(sentence)
            self.tts.runAndWait()
        self.frame_buffer.clear()
        return sentence

    def close(self):
        self.extractor.close()
