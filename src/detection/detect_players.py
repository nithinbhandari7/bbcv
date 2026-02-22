
from ultralytics import YOLO
import cv2
import sys

model = YOLO("yolov8n.pt")  # small & fast

video_path = sys.argv[1]

cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    for box in results.boxes:
        cls = int(box.cls[0])
        if cls == 0:  # class 0 = person
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

    cv2.imshow("Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()


"""
from ultralytics import YOLO
import cv2
import numpy as np
import sys


def hardwood_mask(frame_bgr: np.ndarray) -> np.ndarray:
    H, W = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # Wider wood range (more forgiving)
    lower = np.array([5, 15, 50])
    upper = np.array([45, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    # Clean up
    kernel = np.ones((9, 9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    out = np.zeros_like(mask)

    # Court is typically lower part of frame and somewhat central horizontally
    # We'll keep components whose centroid falls in this region and are not tiny.
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        cx, cy = centroids[i]

        if area < 0.002 * H * W:   # drop tiny specks
            continue

        if (cy > 0.45 * H) and (0.10 * W < cx < 0.90 * W):
            out[labels == i] = 255

    # If nothing selected, fall back to original threshold mask (better than empty)
    if out.sum() == 0:
        out = mask

    return out


def filter_people_on_court(
    frame: np.ndarray,
    person_boxes_xyxy: list[tuple[int, int, int, int]],
    min_height_frac: float = 0.08,   # relaxed from 0.12
    top_k: int = 14,                 # a bit more buffer
    min_keep: int = 6,               # fallback threshold
) -> tuple[list[tuple[int, int, int, int]], np.ndarray]:
    H, W = frame.shape[:2]
    court_mask = hardwood_mask(frame)

    # 1) size filter first (cheap)
    size_ok = []
    for (x1, y1, x2, y2) in person_boxes_xyxy:
        w = x2 - x1
        h = y2 - y1
        if w <= 0 or h <= 0:
            continue
        if h < min_height_frac * H:
            continue
        size_ok.append((x1, y1, x2, y2))

    # 2) court footpoint filter
    filtered = []
    for (x1, y1, x2, y2) in size_ok:
        fx = int((x1 + x2) / 2)
        fy = int(y2)
        fx = int(np.clip(fx, 0, W - 1))
        fy = int(np.clip(fy, 0, H - 1))
        if court_mask[fy, fx] != 0:
            filtered.append((x1, y1, x2, y2))

    # Fallback: if too few remain, skip court filter entirely
    if len(filtered) < min_keep:
        filtered = size_ok

    # 3) keep top-K
    filtered.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    filtered = filtered[:top_k]

    return filtered, court_mask


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python src/detection/detect_players.py <path_to_clip.mp4>")
        return 1

    video_path = sys.argv[1]
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: could not open video: {video_path}")
        return 1

    # YOLOv8 nano is fast; you can switch to yolov8s.pt for better accuracy (slower)
    model = YOLO("yolov8n.pt")

    print("Controls: q = quit")
    print("Filtering: hardwood mask + bbox size + top-K by area")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)[0]

        # Collect person detections
        person_boxes = []
        for box in results.boxes:
            cls = int(box.cls[0])
            if cls == 0:  # COCO class 0 = person
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                person_boxes.append((x1, y1, x2, y2))

        filtered_boxes, court_mask = filter_people_on_court(
            frame,
            person_boxes,
            min_height_frac=0.12,  # tweak if too strict/loose
            top_k=12,              # ~10 players + small buffer
        )

        # Draw boxes
        for (x1, y1, x2, y2) in filtered_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Debug: show court mask in the top-left corner
        mask_vis = cv2.cvtColor(court_mask, cv2.COLOR_GRAY2BGR)
        mask_vis = cv2.resize(mask_vis, (frame.shape[1] // 4, frame.shape[0] // 4))
        frame[0:mask_vis.shape[0], 0:mask_vis.shape[1]] = mask_vis

        # Debug: show counts
        cv2.putText(
            frame,
            f"persons raw: {len(person_boxes)} | filtered: {len(filtered_boxes)}",
            (20, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.imshow("Detection (Filtered)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""