import cv2
import numpy as np
import sys
from ultralytics import YOLO

# Usage:
# python src/mapping/map_release_frame.py runs/release_frames/shot_0001_release.jpg assets/court_left.png runs/homography/H_shot_0001.npy

def in_bounds(x, y, w, h, margin=5):
    return (-margin <= x < w + margin) and (-margin <= y < h + margin)

def main():
    if len(sys.argv) < 4:
        print("Usage: python src/mapping/map_release_frame.py <release.jpg> <court.png> <H.npy>")
        return 1

    release_path = sys.argv[1]
    court_path = sys.argv[2]
    H_path = sys.argv[3]

    frame = cv2.imread(release_path)
    court = cv2.imread(court_path)
    H = np.load(H_path)

    if frame is None or court is None:
        print("Could not load images.")
        return 1

    model = YOLO("yolov8n.pt")
    results = model(frame, verbose=False)[0]

    # Collect footpoints (bottom-center)
    footpoints = []
    for box in results.boxes:
        if int(box.cls[0]) == 0:  # person
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            fx = int((x1 + x2) / 2)
            fy = int(y2)
            footpoints.append([fx, fy])

    if not footpoints:
        print("No people detected.")
        return 1

    pts = np.array(footpoints, dtype=np.float32).reshape(-1, 1, 2)
    mapped = cv2.perspectiveTransform(pts, H).reshape(-1, 2)

    out = court.copy()
    kept = 0

    for (x, y) in mapped:
        xi, yi = int(round(x)), int(round(y))
        if in_bounds(xi, yi, out.shape[1], out.shape[0], margin=0):
            cv2.circle(out, (xi, yi), 6, (0, 0, 255), -1)
            kept += 1

    print(f"Detected people: {len(footpoints)} | Mapped in-bounds: {kept}")

    cv2.imshow("Mapped Footpoints (Release Frame)", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
