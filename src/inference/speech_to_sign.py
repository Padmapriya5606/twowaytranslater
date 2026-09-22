"""
Reverse direction: hearing person speaks -> app shows the animated avatar
signing back for the deaf-mute person.

Wires together: SpeechRecognition (STT) -> simple NLP-to-gloss mapper
-> regional dialect selector -> avatar renderer.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from regional_dialect.dialect_selector import DialectSelector  # noqa: E402
from avatar.avatar_renderer import play_sign_sequence  # noqa: E402

# Minimal stopword list for a first-pass "sentence -> gloss keywords" mapper.
# Replace with the trained reverse-direction transformer
# (train with english_sentence -> gloss_sequence pairs, same architecture as
# transformer_nlp.py) once you have enough parallel data.
_STOPWORDS = {
    "a", "an", "the", "is", "are", "am", "was", "were", "to", "of", "in",
    "on", "at", "for", "and", "do", "does", "did", "will", "would", "can",
    "could", "please", "you",
}


def sentence_to_gloss(sentence: str) -> list:
    words = [w.strip(".,!?").upper() for w in sentence.split()]
    gloss = [w for w in words if w.lower() not in _STOPWORDS and w]
    return gloss or [w for w in words if w]


class SpeechToSignPipeline:
    def __init__(self, region: str = "generic"):
        self.dialect = DialectSelector()
        self.dialect.set_region(region)
        self.recognizer = None

    def _init_stt(self):
        import speech_recognition as sr
        self.recognizer = sr.Recognizer()
        self.mic = sr.Microphone()

    def listen_and_sign(self, timeout: float = 5.0):
        """Listens on the mic, converts to text, maps to gloss, and plays
        the avatar animation. Requires a working microphone + internet OR a
        local STT model (see note below) — the sign side of this app is
        fully offline, but off-the-shelf `speech_recognition` defaults to
        Google's free API for STT; swap in an offline engine such as
        Vosk (https://alphacephei.com/vosk/) for a fully offline build."""
        if self.recognizer is None:
            self._init_stt()

        import speech_recognition as sr
        with self.mic as source:
            self.recognizer.adjust_for_ambient_noise(source)
            print("Listening...")
            audio = self.recognizer.listen(source, timeout=timeout)

        try:
            text = self.recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            print("Could not understand audio.")
            return None
        except sr.RequestError as e:
            print(f"STT service error: {e}. Consider swapping in an offline "
                  f"engine (e.g. Vosk) for a no-internet build.")
            return None

        print(f"[speech->sign] heard: \"{text}\"")
        return self.sign_from_text(text)

    def sign_from_text(self, text: str):
        gloss_tokens = sentence_to_gloss(text)
        asset_paths = [self.dialect.gloss_to_asset(g) for g in gloss_tokens]
        print(f"[speech->sign] gloss={gloss_tokens} region={self.dialect.region}")
        try:
            play_sign_sequence(asset_paths)
        except FileNotFoundError as e:
            print(f"[warning] {e}. Run avatar_renderer.build_starter_sign_assets() "
                  f"or add real regional keyframes.")
        return gloss_tokens
