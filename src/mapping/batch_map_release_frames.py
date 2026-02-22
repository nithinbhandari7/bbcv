import os
import json
import glob
import cv2
import numpy as np
from ultralytics import YOLO

RELEASE_DIR = "runs/release_frames"
H_DIR = "runs/homography"
OUT_DIR = "runs/release_mapped"
MODEL_PATH = "yolov8n.pt"   # assumes ultralytics can load this name

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    model = YOLO(MODEL_PATH)

    release_paths = sorted(glob.glob(os.path.join(RELEASE_DIR, "shot_*_release.jpg")))
    if not release_paths:
        # maybe you used .png or .jpeg
        release_paths = sorted(glob.glob(os.path.join(RELEASE_DIR, "shot_*_release.*")))

    print(f"Found {len(release_paths)} release frames")

    for release_path in release_paths:
        fname = os.path.basename(release_path)
        clip_id = fname.replace("_release.jpg", "").replace("_release.png", "").replace("_release.jpeg", "")

        H_path = os.path.join(H_DIR, f"H_{clip_id}.npy")
        if not os.path.exists(H_path):
            print(f"[SKIP] Missing H: {H_path}")
            continue

        frame = cv2.imread(release_path)
        if frame is None:
            print(f"[SKIP] Could not read: {release_path}")
            continue

        H = np.load(H_path)

        res = model(frame, verbose=False)[0]

        dets = []
        footpoints = []
        boxes = []

        for box in res.boxes:
            if int(box.cls[0]) == 0:  # person
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                fx = (x1 + x2) / 2.0
                fy = y2
                footpoints.append([fx, fy])
                boxes.append([x1, y1, x2, y2])

        if not footpoints:
            out = {
                "clip_id": clip_id,
                "release_frame_img": release_path,
                "num_people_detected": 0,
                "detections": []
            }
            out_path = os.path.join(OUT_DIR, f"{clip_id}.json")
            with open(out_path, "w") as f:
                json.dump(out, f, indent=2)
            print(f"[OK] {clip_id}: 0 detections")
            continue

        pts = np.array(footpoints, dtype=np.float32).reshape(-1, 1, 2)
        mapped = cv2.perspectiveTransform(pts, H).reshape(-1, 2)

        for i in range(len(mapped)):
            dets.append({
                "bbox_xyxy": boxes[i],
                "foot_img": footpoints[i],
                "foot_court_px": mapped[i].tolist()
            })

        out = {
            "clip_id": clip_id,
            "release_frame_img": release_path,
            "num_people_detected": len(dets),
            "detections": dets
        }

        out_path = os.path.join(OUT_DIR, f"{clip_id}.json")
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)

        print(f"[OK] {clip_id}: {len(dets)} detections → {out_path}")

    print("Done. Outputs in runs/release_mapped/")

if __name__ == "__main__":
    main()