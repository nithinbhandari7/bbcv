import os
import cv2
import pandas as pd

CLIPS_DIR = "data/clips"
LABELS_CSV = "data/labels/shots.csv"
OUT_DIR = "runs/release_frames"

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(LABELS_CSV)

for _, row in df.iterrows():
    clip_id = row["clip_id"]
    release_frame = int(row["release_frame"])
    video_path = os.path.join(CLIPS_DIR, f"{clip_id}.mp4")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Could not open:", video_path)
        continue

    cap.set(cv2.CAP_PROP_POS_FRAMES, release_frame)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Could not read frame:", clip_id, release_frame)
        continue

    out_path = os.path.join(OUT_DIR, f"{clip_id}_release.jpg")
    cv2.imwrite(out_path, frame)
    print("Saved:", out_path)
