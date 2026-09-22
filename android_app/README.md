# Android app skeleton

This is a minimal, buildable Android Studio project skeleton wired to load
the `.tflite` models you produce in `src/tflite_conversion/`. It is **not**
a finished production UI — it's the scaffold: Gradle config, permissions,
CameraX preview, TFLite model loading, and a bare-bones activity with two
buttons (Sign→Speech / Speech→Sign) so you have a working starting point to
build the real UI/avatar renderer on top of.

## Setup

1. Open `android_app/` in Android Studio (Arctic Fox or newer).
2. Let Gradle sync — it will pull CameraX, TFLite, and MediaPipe Android deps
   declared in `app/build.gradle`.
3. Copy your converted models here:
   ```
   android_app/app/src/main/assets/sign_model.tflite
   android_app/app/src/main/assets/face_model.tflite
   ```
   (produced by `python src/tflite_conversion/convert_to_tflite.py`)
4. Copy your `sign_label_map.json` (from `checkpoints/`) into
   `app/src/main/assets/` too — the app needs it to turn model output
   indices back into gloss words.
5. Build & run on a device (camera + mic permissions are declared in the
   manifest; the app requests them at runtime).

## What's implemented in this skeleton

- `MainActivity.kt` — permission requests, mode toggle (sign→speech /
  speech→sign), wiring to `TFLiteSignClassifier`.
- `TFLiteSignClassifier.kt` — loads `sign_model.tflite` via the TFLite
  Android Support Library and runs inference on a landmark sequence buffer.
- `build.gradle` (app-level) — CameraX, TFLite, MediaPipe, and Android
  SpeechRecognizer / TextToSpeech dependencies pre-declared.
- `AndroidManifest.xml` — camera + microphone permissions.

## What you still need to build

- MediaPipe Holistic landmark extraction on the Android side (mirror
  `src/landmark_extraction/extract_landmarks.py` using the
  `com.google.mediapipe:tasks-vision` Android library — the Python and
  Android landmark feature vectors must match dimension/order exactly).
- The regional dialect avatar renderer (start from the JSON keyframe format
  in `src/avatar/avatar_renderer.py`; render with a `Canvas`/`SurfaceView`
  skeleton first, then swap in a proper rigged 3D avatar via Filament or
  Unity-as-a-library once the pipeline is validated).
- Wiring Android's built-in `TextToSpeech` and `SpeechRecognizer` APIs
  (both work fully offline on most modern Android devices if the user has
  downloaded offline language packs in Settings — flag this to end users).
- Polishing the UI (this skeleton is intentionally bare so you can design
  it — see `frontend-design` conventions if you want it to look polished).
