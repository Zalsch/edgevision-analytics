import cv2
import time
import numpy as np
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "yolo11n.pt"
SOURCE = 0
DEVICE = "mps"

# COCO:
# 0  = person
# 67 = cell phone
TARGET_CLASSES = [67]

CONFIDENCE_THRESHOLD = 0.40


# Polygon zone
ZONE_POLYGON = np.array([
    [0, 0],
    [350, 0],
    [350, 350],
    [0, 350]
], dtype=np.int32)


# ============================================================
# MAIN
# ============================================================

def main():

    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(SOURCE)

    if not cap.isOpened():
        print("Camera could not be opened.")
        return


    previous_time = time.time()

    track_states = {}

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        cv2.polylines(
            frame,
            [ZONE_POLYGON],
            isClosed=True,
            color=(255, 255, 0),
            thickness=3
        )

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=TARGET_CLASSES,
            conf=CONFIDENCE_THRESHOLD,
            device=DEVICE,
            verbose=False
        )

        result = results[0]
        boxes = result.boxes

        occupancy = 0

        if boxes is not None and boxes.id is not None:

            # Bounding box koordinatları
            coordinates = boxes.xyxy.cpu().numpy()

            # ByteTrack ID'leri
            track_ids = boxes.id.int().cpu().tolist()

            # YOLO confidence değerleri
            confidences = boxes.conf.cpu().tolist()

            # YOLO class ID'leri
            class_ids = boxes.cls.int().cpu().tolist()

            for box, track_id, confidence, class_id in zip(
                coordinates,
                track_ids,
                confidences,
                class_ids
            ):

                # Bounding box
                x1, y1, x2, y2 = map(int, box)

                class_name = model.names[class_id]

                foot_x = (x1 + x2) // 2
                foot_y = y1

                foot_point = (foot_x, foot_y)

                polygon_result = cv2.pointPolygonTest(
                    ZONE_POLYGON,
                    foot_point,
                    False
                )

                is_inside = polygon_result >= 0

                if track_id not in track_states:

                    track_states[track_id] = {
                        "inside": False,
                        "entry_time": None
                    }


                # Önceki frame'deki zone durumu
                previous_inside = track_states[track_id]["inside"]

                dwell_time = 0.0

                if is_inside and not previous_inside:

                    track_states[track_id]["entry_time"] = time.time()

                    print(
                        f"[ENTRY] "
                        f"ID {track_id} | "
                        f"{class_name} entered the zone"
                    )

                if is_inside:

                    occupancy += 1

                    entry_time = track_states[track_id]["entry_time"]

                    if entry_time is not None:
                        dwell_time = time.time() - entry_time

                    status = f"IN | {dwell_time:.1f}s"

                    # Sarı
                    box_color = (0, 255, 255)

                else:

                    if previous_inside:

                        entry_time = track_states[track_id]["entry_time"]

                        if entry_time is not None:
                            total_dwell = time.time() - entry_time
                        else:
                            total_dwell = 0.0

                        print(
                            f"[EXIT] "
                            f"ID {track_id} | "
                            f"{class_name} left the zone "
                            f"after {total_dwell:.2f} seconds"
                        )

                        # Timer'ı sıfırla
                        track_states[track_id]["entry_time"] = None


                    status = "OUT"

                    box_color = (0, 255, 0)

                track_states[track_id]["inside"] = is_inside

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    box_color,
                    2
                )

                cv2.circle(
                    frame,
                    foot_point,
                    6,
                    (0, 0, 255),
                    -1
                )

                label = (
                    f"ID {track_id} | "
                    f"{class_name} | "
                    f"{confidence:.2f} | "
                    f"{status}"
                )


                cv2.putText(
                    frame,
                    label,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    box_color,
                    2
                )

        current_time = time.time()

        fps = 1 / max(
            current_time - previous_time,
            1e-6
        )

        previous_time = current_time

        # FPS
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # Occupancy
        cv2.putText(
            frame,
            f"ZONE OCCUPANCY: {occupancy}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 0),
            2
        )

        cv2.imshow(
            "EdgeVision Analytics - Tracking",
            frame
        )

        # q ile çık
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()