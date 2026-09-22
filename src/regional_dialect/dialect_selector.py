"""
Stage 4: Regional ISL dialect layer.

Same sign class (e.g. "water") can be signed differently in South India,
North India, and Maharashtra. Rather than train one giant undifferentiated
model, we train the CNN-LSTM sign model on landmark sequences that carry a
`region` tag (see data/prepare_dataset.py + train_sign_recognition.py), and
at inference time this module:

  1. Holds the user's selected region (persisted on first app launch).
  2. Loads/filters the correct regional vocabulary table for gloss->sign and
     sign->gloss mapping.
  3. Optionally re-weights model predictions toward the user's region if the
     underlying classifier was trained with a joint (class, region) head.

For the MVP, the simplest correct implementation is: train one shared
CNN-LSTM backbone, but maintain a per-region *lookup table* mapping
canonical gloss -> the region-specific video/animation asset used by the
avatar renderer for the reverse (speech->sign) direction. That's what's
implemented below, since region mainly affects sign *production*
(avatar output) and sign *recognition* is handled by including regional
samples in training data (see data/raw/regional/... and prepare_dataset.py).
"""
import json
from pathlib import Path
from typing import Optional

SUPPORTED_REGIONS = ["south", "north", "maharashtra", "generic"]

DEFAULT_VOCAB_DIR = Path(__file__).resolve().parents[2] / "data" / "regional_vocab"


class DialectSelector:
    def __init__(self, vocab_dir: Path = DEFAULT_VOCAB_DIR):
        self.vocab_dir = Path(vocab_dir)
        self.region: str = "generic"
        self._vocab_cache = {}

    def set_region(self, region: str):
        region = region.lower()
        if region not in SUPPORTED_REGIONS:
            raise ValueError(f"Unsupported region '{region}'. Choose from {SUPPORTED_REGIONS}.")
        self.region = region

    def _load_vocab(self, region: str) -> dict:
        if region in self._vocab_cache:
            return self._vocab_cache[region]

        path = self.vocab_dir / f"{region}.json"
        if not path.exists():
            # Fall back to generic vocab if a region-specific table hasn't
            # been built yet for every gloss.
            path = self.vocab_dir / "generic.json"
        if not path.exists():
            self._vocab_cache[region] = {}
            return {}

        with open(path) as f:
            vocab = json.load(f)
        self._vocab_cache[region] = vocab
        return vocab

    def gloss_to_asset(self, gloss: str, region: Optional[str] = None) -> str:
        """Returns the avatar animation asset path for a gloss word, in the
        style of the given (or currently selected) region, falling back to
        the generic asset if no regional variant has been recorded yet."""
        region = region or self.region
        vocab = self._load_vocab(region)
        if gloss.upper() in vocab:
            return vocab[gloss.upper()]

        generic_vocab = self._load_vocab("generic")
        return generic_vocab.get(gloss.upper(), f"assets/signs/generic/{gloss.lower()}.json")


def build_starter_vocab_files():
    """Creates placeholder regional_vocab/*.json files with a handful of
    example entries so the pipeline is runnable end-to-end before you've
    finished cataloguing every regional variant. Extend these as you label
    your regional dataset."""
    DEFAULT_VOCAB_DIR.mkdir(parents=True, exist_ok=True)
    example_words = ["WATER", "HELLO", "SCHOOL", "DOCTOR", "HELP", "YES", "NO"]

    for region in SUPPORTED_REGIONS:
        out_path = DEFAULT_VOCAB_DIR / f"{region}.json"
        if out_path.exists():
            continue
        vocab = {
            w: f"assets/signs/{region}/{w.lower()}.json" for w in example_words
        }
        with open(out_path, "w") as f:
            json.dump(vocab, f, indent=2)
        print(f"Wrote starter vocab -> {out_path}")


if __name__ == "__main__":
    build_starter_vocab_files()
    selector = DialectSelector()
    selector.set_region("south")
    print(selector.gloss_to_asset("WATER"))
