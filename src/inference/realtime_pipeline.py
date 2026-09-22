"""
Stage 1-6 wired end to end: opens the webcam, buffers frames, and on a
keypress runs sign->speech; press another key to switch to speech->sign.

This is the Python-side desktop demo used to validate the pipeline before
porting the trained TFLite models into the Android app.

Controls:
    s  - capture the current buffer and speak it (sign -> speech)
    l  - listen on the mic and show the avatar (speech -> sign)
    r  - change region (south/north/maharashtra/generic)
    q  - quit
"""
import sys
from pathlib import Path

import cv2

sys.path.append(str(Path(__file__).resolve().parents[1]))
from inference.sign_to_speech import SignToSpeechPipeline  # noqa: E402
from inference.speech_to_sign import SpeechToSignPipeline  # noqa: E402


def main():
    sign_pipeline = SignToSpeechPipeline()
    speech_pipeline = SpeechToSignPipeline(region="generic")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam. If running headless, use the training/"
              "conversion scripts directly instead of this demo.")
        return

    print("Controls: [s]=sign->speech  [l]=speech->sign  [r]=change region  [q]=quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        sign_pipeline.push_frame(frame)
        cv2.imshow("ISL Two-Way Translator - press q to quit", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            sign_pipeline.flush_and_speak()
        elif key == ord("l"):
            speech_pipeline.listen_and_sign()
        elif key == ord("r"):
            region = input("Enter region (south/north/maharashtra/generic): ").strip()
            try:
                speech_pipeline.dialect.set_region(region)
                print(f"Region set to: {region}")
            except ValueError as e:
                print(e)

    cap.release()
    cv2.destroyAllWindows()
    sign_pipeline.close()


if __name__ == "__main__":
    main()
