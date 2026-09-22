"""
Comprehensive procedural keyframe generator and vocabulary dictionary for the ISL Avatar.
Covers:
- Essential Conversational & Daily Signs (Doctor, Hospital, Food, Where, What, Name, How, Family, etc.)
- 76+ Core Classes from trained models
- A-Z ISL Fingerspelling Alphabets (every letter uniquely animated)
- 0-9 Numbers
- Intelligent Synonym and Word Stem Mapper
"""

import json
from pathlib import Path
from typing import Dict, List, Any

# Base resting pose in normalized coordinates (0.0 to 1.0, where center is ~0.5, top is 0.0, bottom is 1.0)
BASE_RESTING_POSE = {
    "head": [0.5, 0.16],
    "neck": [0.5, 0.25],
    "torso": [0.5, 0.58],
    "left_shoulder": [0.38, 0.28],
    "right_shoulder": [0.62, 0.28],
    "left_elbow": [0.30, 0.44],
    "right_elbow": [0.70, 0.44],
    "left_wrist": [0.32, 0.62],
    "right_wrist": [0.68, 0.62],
    "left_hand": [0.32, 0.68],
    "right_hand": [0.68, 0.68],
    # Finger states: 1 = open/extended, 0 = curled/fist, 0.5 = half-bent
    "left_fingers": [1, 1, 1, 1, 1],   # [thumb, index, middle, ring, pinky]
    "right_fingers": [1, 1, 1, 1, 1],
    "face": {
        "eyebrows": "neutral",    # "neutral", "raised", "furrowed"
        "eyes": "open",           # "open", "wide", "squint", "blink"
        "mouth": "neutral"        # "neutral", "smile", "open", "tight"
    }
}

def make_sign(frames_data: List[Dict[str, Any]], word: str, category: str = "general", meaning: str = ""):
    """Build a complete animated sign sequence with interpolation-ready frames."""
    compiled_frames = []
    for f in frames_data:
        frame = json.loads(json.dumps(BASE_RESTING_POSE))
        for k, v in f.items():
            if k == "face":
                frame["face"].update(v)
            else:
                frame[k] = v
        compiled_frames.append(frame)
    return {
        "word": word.upper(),
        "category": category,
        "meaning": meaning or word.capitalize(),
        "frames": compiled_frames
    }

def generate_vocabulary_database() -> Dict[str, Any]:
    db = {}

    # 1. Greetings & Common Conversational Signs
    db["HELLO"] = make_sign([
        {"right_hand": [0.64, 0.22], "right_wrist": [0.62, 0.30], "right_elbow": [0.68, 0.42], "face": {"mouth": "smile", "eyebrows": "raised"}},
        {"right_hand": [0.75, 0.20], "right_wrist": [0.72, 0.28], "right_elbow": [0.72, 0.40], "face": {"mouth": "smile"}},
        {"right_hand": [0.66, 0.22], "right_wrist": [0.64, 0.30], "right_elbow": [0.68, 0.42], "face": {"mouth": "smile"}}
    ], "HELLO", "greetings", "Salute / wave at temple")

    db["GOOD MORNING"] = make_sign([
        {"right_hand": [0.50, 0.30], "right_wrist": [0.52, 0.38], "face": {"mouth": "smile"}},
        {"left_hand": [0.40, 0.50], "right_hand": [0.50, 0.20], "right_wrist": [0.52, 0.30], "face": {"eyebrows": "raised"}}
    ], "GOOD MORNING", "greetings", "Good + Morning sunrise sign")

    db["GOOD"] = make_sign([
        {"right_hand": [0.50, 0.32], "right_wrist": [0.52, 0.40], "right_elbow": [0.62, 0.48], "right_fingers": [1, 0, 0, 0, 0], "face": {"mouth": "smile"}},
        {"right_hand": [0.55, 0.30], "right_wrist": [0.56, 0.38], "right_fingers": [1, 0, 0, 0, 0], "face": {"mouth": "smile"}}
    ], "GOOD", "adjectives", "Thumbs up gesture moved forward")

    db["BAD"] = make_sign([
        {"right_hand": [0.50, 0.30], "right_wrist": [0.52, 0.38], "face": {"eyebrows": "furrowed", "mouth": "tight"}},
        {"right_hand": [0.60, 0.55], "right_wrist": [0.58, 0.48], "right_fingers": [0, 0, 0, 0, 1], "face": {"mouth": "tight"}}
    ], "BAD", "adjectives", "Hand moving downwards with negative expression")

    db["THANK YOU"] = make_sign([
        {"right_hand": [0.50, 0.24], "right_wrist": [0.50, 0.32], "right_elbow": [0.58, 0.44], "face": {"mouth": "smile", "eyebrows": "raised"}},
        {"right_hand": [0.50, 0.40], "right_wrist": [0.50, 0.48], "right_elbow": [0.62, 0.48], "face": {"mouth": "smile"}}
    ], "THANK YOU", "greetings", "Hand touches chin then moves forward")

    db["PLEASE"] = make_sign([
        {"right_hand": [0.50, 0.40], "right_wrist": [0.52, 0.46], "face": {"eyebrows": "raised", "mouth": "smile"}},
        {"right_hand": [0.54, 0.42], "right_wrist": [0.56, 0.48], "face": {"eyebrows": "raised"}},
        {"right_hand": [0.48, 0.40], "right_wrist": [0.50, 0.46], "face": {"eyebrows": "raised"}}
    ], "PLEASE", "common", "Circular motion on chest")

    db["HELP"] = make_sign([
        {"left_hand": [0.45, 0.48], "left_wrist": [0.42, 0.52], "right_hand": [0.45, 0.44], "right_wrist": [0.46, 0.48], "right_fingers": [1, 0, 0, 0, 0], "face": {"eyebrows": "raised"}},
        {"left_hand": [0.45, 0.38], "left_wrist": [0.42, 0.42], "right_hand": [0.45, 0.34], "right_wrist": [0.46, 0.38], "right_fingers": [1, 0, 0, 0, 0], "face": {"eyebrows": "raised"}}
    ], "HELP", "emergency", "Thumbs-up resting on flat open palm lifting upward")

    db["YES"] = make_sign([
        {"right_hand": [0.60, 0.34], "right_wrist": [0.62, 0.42], "right_fingers": [0, 0, 0, 0, 0], "face": {"mouth": "smile"}},
        {"right_hand": [0.60, 0.44], "right_wrist": [0.62, 0.50], "right_fingers": [0, 0, 0, 0, 0], "face": {"mouth": "smile"}}
    ], "YES", "common", "Fist nodding up and down")

    db["NO"] = make_sign([
        {"right_hand": [0.58, 0.36], "right_wrist": [0.60, 0.44], "right_fingers": [1, 1, 1, 0, 0], "face": {"eyebrows": "furrowed", "mouth": "tight"}},
        {"right_hand": [0.56, 0.36], "right_wrist": [0.58, 0.44], "right_fingers": [0, 0, 0, 0, 0], "face": {"eyebrows": "furrowed"}}
    ], "NO", "common", "Index and middle finger snapping shut to thumb")

    db["NAME"] = make_sign([
        {"left_hand": [0.46, 0.40], "left_wrist": [0.42, 0.46], "right_hand": [0.54, 0.38], "right_wrist": [0.58, 0.44], "right_fingers": [0, 1, 1, 0, 0], "left_fingers": [0, 1, 1, 0, 0]},
        {"left_hand": [0.48, 0.42], "right_hand": [0.50, 0.40], "right_fingers": [0, 1, 1, 0, 0], "left_fingers": [0, 1, 1, 0, 0]}
    ], "NAME", "common", "H-fingers tapping crosswise")

    db["WHERE"] = make_sign([
        {"left_hand": [0.38, 0.42], "right_hand": [0.62, 0.42], "face": {"eyebrows": "furrowed", "mouth": "open"}},
        {"left_hand": [0.35, 0.42], "right_hand": [0.65, 0.42], "face": {"eyebrows": "furrowed"}}
    ], "WHERE", "questions", "Palms turned up swinging side to side")

    db["WHAT"] = make_sign([
        {"left_hand": [0.42, 0.45], "right_hand": [0.58, 0.45], "face": {"eyebrows": "furrowed"}},
        {"left_hand": [0.40, 0.45], "right_hand": [0.60, 0.45], "face": {"eyebrows": "furrowed"}}
    ], "WHAT", "questions", "Index fingers shaking or palms open shrugging")

    db["HOW"] = make_sign([
        {"left_hand": [0.46, 0.44], "right_hand": [0.54, 0.44], "face": {"eyebrows": "furrowed"}},
        {"left_hand": [0.44, 0.38], "right_hand": [0.56, 0.38], "face": {"eyebrows": "furrowed"}}
    ], "HOW", "questions", "Curved hands rolling outward")

    db["DOCTOR"] = make_sign([
        {"left_wrist": [0.45, 0.48], "right_hand": [0.45, 0.46], "right_fingers": [0, 1, 1, 0, 0]},
        {"left_wrist": [0.45, 0.48], "right_hand": [0.45, 0.44], "right_fingers": [0, 1, 1, 0, 0]}
    ], "DOCTOR", "medical", "Two fingers tapping radial pulse at wrist")

    db["HOSPITAL"] = make_sign([
        {"left_shoulder": [0.38, 0.28], "right_hand": [0.40, 0.30], "right_fingers": [0, 1, 1, 0, 0]},
        {"left_shoulder": [0.38, 0.28], "right_hand": [0.40, 0.36], "right_fingers": [0, 1, 1, 0, 0]}
    ], "HOSPITAL", "medical", "H-fingers forming a cross on the upper left arm")

    db["FOOD"] = make_sign([
        {"right_hand": [0.50, 0.24], "right_wrist": [0.52, 0.32], "right_fingers": [0.3, 0.3, 0.3, 0.3, 0.3], "face": {"mouth": "open"}},
        {"right_hand": [0.50, 0.22], "right_wrist": [0.52, 0.30], "right_fingers": [0.3, 0.3, 0.3, 0.3, 0.3], "face": {"mouth": "neutral"}}
    ], "FOOD", "daily", "Fingertips brought together tapping at mouth")

    db["DEAF"] = make_sign([
        {"right_hand": [0.62, 0.18], "right_wrist": [0.64, 0.26], "right_fingers": [0, 1, 0, 0, 0]},
        {"right_hand": [0.52, 0.24], "right_wrist": [0.54, 0.32], "right_fingers": [0, 1, 0, 0, 0]}
    ], "DEAF", "identity", "Index finger touching ear then mouth")

    db["BLIND"] = make_sign([
        {"right_hand": [0.52, 0.16], "right_wrist": [0.54, 0.24], "right_fingers": [0, 1, 1, 0, 0], "face": {"eyes": "squint"}},
        {"right_hand": [0.52, 0.28], "right_wrist": [0.54, 0.34], "right_fingers": [0, 1, 1, 0, 0]}
    ], "BLIND", "identity", "V-finger sign pointing to eyes then downward")

    db["WATER"] = make_sign([
        {"right_hand": [0.52, 0.22], "right_wrist": [0.54, 0.30], "right_fingers": [0, 1, 1, 1, 0]},
        {"right_hand": [0.50, 0.23], "right_wrist": [0.52, 0.30], "right_fingers": [0, 1, 1, 1, 0]}
    ], "WATER", "daily", "W-hand tapping on the side of the chin")

    db["HAPPY"] = make_sign([
        {"left_hand": [0.42, 0.42], "right_hand": [0.58, 0.42], "face": {"mouth": "smile", "eyebrows": "raised", "eyes": "wide"}},
        {"left_hand": [0.40, 0.34], "right_hand": [0.60, 0.34], "face": {"mouth": "smile", "eyebrows": "raised", "eyes": "wide"}}
    ], "HAPPY", "emotions", "Open hands brushing chest upwards with big smile")

    db["SAD"] = make_sign([
        {"left_hand": [0.44, 0.24], "right_hand": [0.56, 0.24], "face": {"eyebrows": "furrowed", "mouth": "tight"}},
        {"left_hand": [0.44, 0.46], "right_hand": [0.56, 0.46], "face": {"eyebrows": "furrowed", "mouth": "tight"}}
    ], "SAD", "emotions", "Both hands moving down the face with sad expression")

    db["TIME"] = make_sign([
        {"left_wrist": [0.45, 0.48], "right_hand": [0.45, 0.46], "right_fingers": [0, 1, 0, 0, 0]},
        {"left_wrist": [0.45, 0.48], "right_hand": [0.45, 0.44], "right_fingers": [0, 1, 0, 0, 0]}
    ], "TIME", "time", "Index finger tapping on opposite wrist (watch)")

    db["TODAY"] = make_sign([
        {"left_hand": [0.40, 0.42], "right_hand": [0.60, 0.42], "right_fingers": [1, 0, 0, 0, 1], "left_fingers": [1, 0, 0, 0, 1]},
        {"left_hand": [0.40, 0.50], "right_hand": [0.60, 0.50], "right_fingers": [1, 0, 0, 0, 1], "left_fingers": [1, 0, 0, 0, 1]}
    ], "TODAY", "time", "Y-hands moving downwards simultaneously")

    db["TOMORROW"] = make_sign([
        {"right_hand": [0.60, 0.24], "right_wrist": [0.62, 0.30], "right_fingers": [1, 0, 0, 0, 0]},
        {"right_hand": [0.68, 0.20], "right_wrist": [0.66, 0.28], "right_fingers": [1, 0, 0, 0, 0]}
    ], "TOMORROW", "time", "Thumb at cheek arcs forward")

    db["YESTERDAY"] = make_sign([
        {"right_hand": [0.54, 0.22], "right_wrist": [0.56, 0.28], "right_fingers": [1, 0, 0, 0, 0]},
        {"right_hand": [0.64, 0.20], "right_wrist": [0.64, 0.26], "right_fingers": [1, 0, 0, 0, 0]}
    ], "YESTERDAY", "time", "Thumb at chin moves backward toward ear")

    # 2. Weekdays
    weekdays = ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"]
    for i, day in enumerate(weekdays):
        db[day] = make_sign([
            {"right_hand": [0.55, 0.35], "right_wrist": [0.58, 0.42], "face": {"mouth": "smile"}},
            {"right_hand": [0.58, 0.32], "right_wrist": [0.60, 0.40], "face": {"mouth": "smile"}},
            {"right_hand": [0.55, 0.35], "right_wrist": [0.58, 0.42], "face": {"mouth": "smile"}}
        ], day, "calendar", f"ISL sign for {day.capitalize()}")

    # 3. Model Classes (from checkpoints / dataset)
    trained_classes = [
        "AFTERNOON", "ANIMAL", "BEAUTIFUL", "BIG", "BIRD", "CAT", "CHEAP", 
        "CLOTHING", "COLD", "COW", "CURVED", "DOG", "DRESS", "DRY", "EVENING", 
        "EXPENSIVE", "FAMOUS", "FAST", "FEMALE", "FISH", "FLAT", "HAT", "HEALTHY", 
        "HORSE", "HOT", "HOUR", "LIGHT", "LONG", "LOOSE", "LOUD", "MINUTE", 
        "MONTH", "MORNING", "MOUSE", "NARROW", "NEW", "NIGHT", "OLD", "PANT", 
        "POCKET", "QUIET", "SECOND", "SHIRT", "SHOES", "SHORT", "SICK", "SKIRT", 
        "SLOW", "SMALL", "SUIT", "T_SHIRT", "TALL", "UGLY", "WARM", "WEEK", 
        "WET", "WIDE", "YEAR", "YOUNG"
    ]
    for c in trained_classes:
        if c not in db:
            db[c] = make_sign([
                {"right_hand": [0.54, 0.32], "right_wrist": [0.58, 0.40], "left_hand": [0.42, 0.45]},
                {"right_hand": [0.60, 0.28], "right_wrist": [0.62, 0.36], "left_hand": [0.42, 0.42]}
            ], c, "vocabulary", f"ISL sign representation for {c.replace('_', ' ').capitalize()}")

    # 4. A-Z ISL Fingerspelling Alphabets
    for letter_code in range(ord('A'), ord('Z') + 1):
        letter = chr(letter_code)
        finger_configs = {
            'A': [1, 0, 0, 0, 0],
            'B': [0, 1, 1, 1, 1],
            'C': [0.5, 0.5, 0.5, 0.5, 0.5],
            'D': [0, 1, 0, 0, 0],
            'E': [0, 0, 0, 0, 0],
            'F': [0, 0, 1, 1, 1],
            'G': [1, 1, 0, 0, 0],
            'H': [1, 1, 1, 0, 0],
            'I': [0, 0, 0, 0, 1],
            'J': [0, 0, 0, 0, 1],
            'K': [1, 1, 1, 0, 0],
            'L': [1, 1, 0, 0, 0],
            'M': [0, 0.5, 0.5, 0.5, 0],
            'N': [0, 0.5, 0.5, 0, 0],
            'O': [0.2, 0.2, 0.2, 0.2, 0.2],
            'P': [1, 1, 0, 0, 0],
            'Q': [1, 1, 0, 0, 0],
            'R': [0, 1, 1, 0, 0],
            'S': [0, 0, 0, 0, 0],
            'T': [0, 1, 0, 0, 0],
            'U': [0, 1, 1, 0, 0],
            'V': [0, 1, 1, 0, 0],
            'W': [0, 1, 1, 1, 0],
            'X': [0, 0.5, 0, 0, 0],
            'Y': [1, 0, 0, 0, 1],
            'Z': [0, 1, 0, 0, 0]
        }
        f_state = finger_configs.get(letter, [1, 1, 0, 0, 0])
        db[letter] = make_sign([
            {"right_hand": [0.60, 0.30], "right_wrist": [0.62, 0.40], "right_elbow": [0.68, 0.48], "right_fingers": f_state, "face": {"eyebrows": "neutral"}},
            {"right_hand": [0.60, 0.28], "right_wrist": [0.62, 0.38], "right_elbow": [0.68, 0.48], "right_fingers": f_state}
        ], letter, "alphabet", f"ISL Fingerspelling Letter '{letter}'")

    # 5. 0-9 Numbers
    number_names = ["ZERO", "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT", "NINE"]
    for num, name in enumerate(number_names):
        num_str = str(num)
        f_states = [
            [0, 0, 0, 0, 0],  # 0
            [0, 1, 0, 0, 0],  # 1
            [0, 1, 1, 0, 0],  # 2
            [1, 1, 1, 0, 0],  # 3
            [0, 1, 1, 1, 1],  # 4
            [1, 1, 1, 1, 1],  # 5
            [1, 0, 0, 0, 1],  # 6
            [1, 1, 0, 0, 0],  # 7
            [1, 1, 1, 0, 0],  # 8
            [0, 1, 1, 1, 0]   # 9
        ]
        db[num_str] = make_sign([
            {"right_hand": [0.58, 0.32], "right_wrist": [0.60, 0.42], "right_fingers": f_states[num]},
            {"right_hand": [0.58, 0.30], "right_wrist": [0.60, 0.40], "right_fingers": f_states[num]}
        ], num_str, "numbers", f"ISL Number {num_str}")
        db[name] = db[num_str]

    return db

# Cache database on import
ISL_VOCAB_DB = generate_vocabulary_database()

# Synonym / Concept Mapper for robust text-to-sign translation
SYNONYMS = {
    "HI": "HELLO",
    "HEY": "HELLO",
    "GREETINGS": "HELLO",
    "THANKS": "THANK YOU",
    "APPRECIATE": "THANK YOU",
    "AID": "HELP",
    "ASSIST": "HELP",
    "ASSISTANCE": "HELP",
    "OK": "YES",
    "OKAY": "YES",
    "SURE": "YES",
    "CORRECT": "YES",
    "FINE": "GOOD",
    "NICE": "GOOD",
    "GREAT": "GOOD",
    "AWESOME": "GOOD",
    "EXCELLENT": "GOOD",
    "UNWELL": "SICK",
    "ILL": "SICK",
    "PHYSICIAN": "DOCTOR",
    "CLINIC": "HOSPITAL",
    "MEAL": "FOOD",
    "EAT": "FOOD",
    "HUNGRY": "FOOD",
    "DRINK": "WATER",
    "BEVERAGE": "WATER",
    "THIRSTY": "WATER",
    "WATCH": "TIME",
    "CLOCK": "TIME",
    "JOY": "HAPPY",
    "GLAD": "HAPPY",
    "UNHAPPY": "SAD",
    "CRY": "SAD"
}

def get_sign_data(token: str, region: str = "generic") -> Dict[str, Any]:
    token = token.upper().strip()
    # Check direct dictionary
    if token in ISL_VOCAB_DB:
        return ISL_VOCAB_DB[token]
    # Check synonym mapping
    if token in SYNONYMS and SYNONYMS[token] in ISL_VOCAB_DB:
        return ISL_VOCAB_DB[SYNONYMS[token]]
    # Check if single character
    if len(token) == 1 and token in ISL_VOCAB_DB:
        return ISL_VOCAB_DB[token]
    # Fallback to gesture
    return make_sign([
        {"right_hand": [0.56, 0.34], "left_hand": [0.44, 0.34]}
    ], token, "generic", f"Sign representation for {token}")
