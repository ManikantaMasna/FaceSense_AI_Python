
"""
FaceSense AI - Live Facial Expression Recognition
Python + OpenCV + DeepFace

Run:
    python main.py

Press Q or ESC to quit.
Press S to save a screenshot.
"""

import cv2
import time
from datetime import datetime
from pathlib import Path

try:
    from deepface import DeepFace
except ImportError as exc:
    raise SystemExit(
        "DeepFace is not installed. Run: pip install deepface"
    ) from exc


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

WINDOW_NAME = "FaceSense AI - Live Expression Recognition"

CAMERA_INDEX = 0
ANALYZE_EVERY_N_FRAMES = 8

DETECTOR_BACKEND = "retinaface"

EMOTION_LABELS = {
    "angry": "Angry",
    "disgust": "Disgusted",
    "fear": "Fearful",
    "happy": "Happy",
    "sad": "Sad",
    "surprise": "Surprised",
    "neutral": "Neutral",
}

EMOTION_COLORS = {
    "angry": (70, 70, 240),
    "disgust": (80, 180, 80),
    "fear": (190, 100, 220),
    "happy": (70, 210, 120),
    "sad": (220, 150, 70),
    "surprise": (70, 200, 240),
    "neutral": (190, 190, 190),
}


# --------------------------------------------------
# TEXT DISPLAY
# --------------------------------------------------

def draw_text(
    image,
    text,
    position,
    scale=0.65,
    color=(245, 245, 245),
    thickness=2
):
    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA
    )


# --------------------------------------------------
# FACE AND EMOTION ANALYSIS
# --------------------------------------------------

def analyze_frame(frame):

    results = DeepFace.analyze(
        img_path=frame,
        actions=["emotion"],
        detector_backend=DETECTOR_BACKEND,
        enforce_detection=True,
        silent=True,
    )

    # DeepFace can return either one dictionary
    # or a list of dictionaries for multiple faces.

    if isinstance(results, dict):
        results = [results]

    return results


# --------------------------------------------------
# MAIN APPLICATION
# --------------------------------------------------

def main():

    print("=" * 58)
    print("                 FACESENSE AI")
    print("       Live Facial Expression Recognition")
    print("=" * 58)

    print("Loading AI models. The first run may take longer...")
    print("Camera access will be requested by your operating system.")

    # Open webcam using Windows DirectShow.

    camera = cv2.VideoCapture(
        CAMERA_INDEX,
        cv2.CAP_DSHOW
    )

    if not camera.isOpened():
        raise SystemExit(
            "Could not open webcam. Try CAMERA_INDEX = 1."
        )

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL
    )

    frame_count = 0
    cached_results = []

    fps_value = 0.0
    previous_time = time.time()

    screenshot_dir = Path("screenshots")
    screenshot_dir.mkdir(exist_ok=True)

    print("\nFaceSense AI is running.")
    print("Controls: Q / ESC = quit | S = save screenshot\n")

    try:

        while True:

            success, frame = camera.read()

            if not success:
                print("Unable to read from webcam.")
                break

            # Mirror webcam image.

            frame = cv2.flip(frame, 1)

            frame_count += 1

            # Run AI analysis periodically.

            if frame_count % ANALYZE_EVERY_N_FRAMES == 0:

                try:

                    cached_results = analyze_frame(frame)

                    print(
                        "Faces detected:",
                        len(cached_results)
                    )

                    for result in cached_results:

                        print(
                            "Expression:",
                            result.get("dominant_emotion", "Unknown")
                        )

                except Exception as error:

                    # No face found is a normal possibility.
                    # Keep the camera preview running.

                    error_message = str(error)

                    if "Face could not be detected" not in error_message:
                        print(f"Analysis warning: {error}")

                    cached_results = []

            # --------------------------------------------------
            # DRAW FACE DETECTION AND EMOTIONS
            # --------------------------------------------------

            face_count = 0

            for result in cached_results:

                region = result.get("region", {})

                x = int(region.get("x", 0))
                y = int(region.get("y", 0))
                w = int(region.get("w", 0))
                h = int(region.get("h", 0))

                emotions = result.get("emotion", {})

                dominant = result.get(
                    "dominant_emotion",
                    "neutral"
                )

                confidence = float(
                    emotions.get(dominant, 0.0)
                )

                color = EMOTION_COLORS.get(
                    dominant,
                    (180, 180, 180)
                )

                label = EMOTION_LABELS.get(
                    dominant,
                    dominant.title()
                )

                if w > 0 and h > 0:

                    face_count += 1

                    # Face rectangle.

                    cv2.rectangle(
                        frame,
                        (x, y),
                        (x + w, y + h),
                        color,
                        2
                    )

                    # Expression label.

                    label_text = (
                        f"{label} | {confidence:.1f}%"
                    )

                    label_y = max(y - 12, 25)

                    (text_w, text_h), _ = cv2.getTextSize(
                        label_text,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.62,
                        2
                    )

                    cv2.rectangle(
                        frame,
                        (x, label_y - text_h - 10),
                        (x + text_w + 10, label_y + 4),
                        color,
                        -1
                    )

                    draw_text(
                        frame,
                        label_text,
                        (x + 5, label_y - 2),
                        scale=0.62,
                        color=(15, 18, 25),
                        thickness=2
                    )

                    # Top three emotion scores.

                    top_emotions = sorted(
                        emotions.items(),
                        key=lambda item: item[1],
                        reverse=True
                    )[:3]

                    for index, (emotion, score) in enumerate(
                        top_emotions
                    ):

                        detail = (
                            f"{EMOTION_LABELS.get(emotion, emotion.title())}: "
                            f"{float(score):.1f}%"
                        )

                        draw_text(
                            frame,
                            detail,
                            (
                                x,
                                min(
                                    y + h + 24 + index * 23,
                                    frame.shape[0] - 12
                                )
                            ),
                            scale=0.52,
                            color=color,
                            thickness=1
                        )

            # --------------------------------------------------
            # FPS CALCULATION
            # --------------------------------------------------

            current_time = time.time()

            elapsed = current_time - previous_time

            if elapsed > 0:

                instant_fps = 1.0 / elapsed

                fps_value = (
                    instant_fps
                    if fps_value == 0
                    else (
                        0.85 * fps_value
                        + 0.15 * instant_fps
                    )
                )

            previous_time = current_time

            # --------------------------------------------------
            # DASHBOARD HEADER
            # --------------------------------------------------

            cv2.rectangle(
                frame,
                (0, 0),
                (frame.shape[1], 74),
                (16, 21, 35),
                -1
            )

            draw_text(
                frame,
                "FACESENSE AI",
                (24, 31),
                0.8,
                (190, 165, 255),
                2
            )

            draw_text(
                frame,
                "LIVE FACIAL EXPRESSION RECOGNITION",
                (24, 56),
                0.42,
                (205, 210, 225),
                1
            )

            right_text = (
                f"FACES: {face_count}   FPS: {fps_value:.1f}"
            )

            draw_text(
                frame,
                right_text,
                (max(20, frame.shape[1] - 300), 42),
                0.55,
                (90, 230, 170),
                2
            )

            # --------------------------------------------------
            # FOOTER
            # --------------------------------------------------

            footer_y = frame.shape[0] - 18

            draw_text(
                frame,
                "Q / ESC: QUIT     S: SAVE SCREENSHOT",
                (20, footer_y),
                0.48,
                (210, 215, 230),
                1
            )

            if not cached_results:

                draw_text(
                    frame,
                    "Position your face in front of the camera",
                    (24, 110),
                    0.62,
                    (190, 190, 210),
                    1
                )

            # Display camera window.

            cv2.imshow(
                WINDOW_NAME,
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            # Quit application.

            if key in (ord("q"), ord("Q"), 27):
                break

            # Save screenshot.

            if key in (ord("s"), ord("S")):

                filename = screenshot_dir / (
                    f"facesense_{datetime.now():%Y%m%d_%H%M%S}.jpg"
                )

                cv2.imwrite(
                    str(filename),
                    frame
                )

                print(
                    f"Screenshot saved: {filename}"
                )

    finally:

        camera.release()

        cv2.destroyAllWindows()

        print("Camera released. FaceSense AI stopped.")


# --------------------------------------------------
# APPLICATION ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":
    main()
