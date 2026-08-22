import cv2
import time
from ultralytics import YOLO
import numpy as np


MODEL_PATH = "yolo26s.pt"
SOURCE = 0

ZONE_POLYGON = np.array([
    [0, 0],
    [350, 0],
    [350, 350],
    [0, 350]
], dtype=np.int32)

def main():

    # --------------------------------------------------
    # 1. YOLO modelini yükle
    # --------------------------------------------------
    model = YOLO(MODEL_PATH)
    
    # --------------------------------------------------
    # 2. Kamerayı aç
    # --------------------------------------------------
    cap = cv2.VideoCapture(SOURCE)

    if not cap.isOpened():
        print("Camera could not be opened.")
        return

    previous_time = time.time()

    while True:

        # --------------------------------------------------
        # 3. Kameradan bir frame al
        # --------------------------------------------------
        ret, frame = cap.read()

        if not ret:
            break

        cv2.polylines(
            frame,
            [ZONE_POLYGON],
            True,
            (255, 255, 0),
            3
        )   
        # --------------------------------------------------
        # 4. YOLO + ByteTrack
        # --------------------------------------------------
        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[0, 67],
            conf=0.35,
            device="mps",
            verbose=False
        )

        result = results[0]

        # --------------------------------------------------
        # 5. Bounding box bilgilerini al
        # --------------------------------------------------
        boxes = result.boxes
        occupancy = 0

        if boxes is not None and boxes.id is not None:

            coordinates = boxes.xyxy.cpu().numpy()

            track_ids = boxes.id.int().cpu().tolist()

            confidences = boxes.conf.cpu().tolist()

            class_ids = boxes.cls.int().cpu().tolist()

            # --------------------------------------------------
            # 6. Her kişiyi ekrana çiz
            # --------------------------------------------------

            for box, track_id, confidence, class_ids in zip(
                coordinates,
                track_ids,
                confidences,
                class_ids
            ):
                x1, y1, x2, y2 = map(int, box)
                class_name = model.names[class_ids]

                # Kişinin zemindeki yaklaşık konumu
                foot_x = (x1 + x2) // 2
                foot_y = y1

                foot_point = (foot_x, foot_y)

                # Nokta polygon içinde mi?
                inside = cv2.pointPolygonTest(
                    ZONE_POLYGON,
                    foot_point,
                    False
                )

                if inside >= 0:
                    occupancy += 1
                    status = "IN ZONE"
                    box_color = (0, 255, 255)

                else:
                    status = "OUT"
                    box_color = (0, 255, 0)

                # Bounding box
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    box_color,
                    2
                )

                # Ayağın/zemin temasının yaklaşık noktası
                cv2.circle(
                    frame,
                    foot_point,
                    6,
                    (0, 0, 255),
                    -1
                )

                label = (
                    f"ID {track_id} | "
                    f"{confidence:.2f} | "
                    f"{status} | "
                    f"{class_name}"
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
        cv2.putText(
            frame,
            f"ZONE OCCUPANCY: {occupancy}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 0),
            2
        )
        
        # --------------------------------------------------
        # 7. FPS hesapla
        # --------------------------------------------------
        current_time = time.time()

        fps = 1 / max(
            current_time - previous_time,
            1e-6
        )

        previous_time = current_time

        # --------------------------------------------------
        # 8. FPS'i ekrana yaz
        # --------------------------------------------------
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # --------------------------------------------------
        # 9. Görüntüyü göster
        # --------------------------------------------------
        cv2.imshow(
            "EdgeVision Analytics - Tracking",
            frame
        )

        # q basılırsa çık
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # --------------------------------------------------
    # 10. Kaynakları temizle
    # --------------------------------------------------
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()