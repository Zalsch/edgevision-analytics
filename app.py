import cv2
import time
from ultralytics import YOLO


MODEL_PATH = "yolo26n.pt"
SOURCE = 0


def main():
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(SOURCE)

    if not cap.isOpened():
        print("Camera could not be opened.")
        return

    previous_time = time.time()

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        results = model(
        frame,
        classes=[0],
        conf=0.4,
        device="mps",
        verbose=False
        )

        annotated_frame = results[0].plot()

        current_time = time.time()
        fps = 1 / max(current_time - previous_time, 1e-6)
        previous_time = current_time

        cv2.putText(
            annotated_frame,
            f"FPS: {fps:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow("EdgeVision Analytics", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()