"""
ball_track_fast.py

Fast NBA broadcast ball tracking:
- Single YOLO inference per frame (classes: person + sports ball)
- Optional downscale for inference speed
- Simple gating-based ball tracking
- Draw results on original frame

Run:
  python src/detection/ball_track_fast.py

Keys:
  q: quit
  space: pause/unpause
  r: reset ball track
"""

import cv2
import numpy as np
from ultralytics import YOLO

# -----------------------------
# Config
# -----------------------------
VIDEO_PATH = "data/clips/shot_0002.mp4"
MODEL_PATH = "yolov8s.pt"   # s is often best speed/quality tradeoff on Mac

PERSON_CLASS_ID = 0
BALL_CLASS_ID = 32

# Inference speedups
USE_MPS_IF_AVAILABLE = True          # Apple GPU (Metal). If it errors, set False.
INFER_W = 1280                       # resize width for inference (e.g., 960/1280). 0 = no resize.
IMGSZ = 960                          # internal YOLO imgsz (keep modest for speed)

# Detection thresholds
CONF = 0.06                          # shared conf for both classes
IOU = 0.60

# Ball tracking / gating
GATE_PX = 240                        # max allowed jump (in inference-scale pixels)
MAX_MISSES = 60                      # after this many missed frames, forget ball

# Draw
DRAW_PLAYERS = True
DRAW_ALL_BALL_DETS = True

WINDOW_NAME = "Fast Ball Tracking (1x YOLO/frame)"


# -----------------------------
# Helpers
# -----------------------------
def resize_keep_aspect(frame, target_w):
    if target_w <= 0:
        return frame, 1.0
    h, w = frame.shape[:2]
    if w == target_w:
        return frame, 1.0
    scale = target_w / float(w)
    new_h = int(round(h * scale))
    out = cv2.resize(frame, (target_w, new_h), interpolation=cv2.INTER_AREA)
    return out, scale

def xyxy_to_int(b):
    return tuple(int(x) for x in b)

def center_xyxy(x1, y1, x2, y2):
    return (0.5*(x1+x2), 0.5*(y1+y2))

def clamp(v, lo, hi):
    return max(lo, min(hi, v))


# -----------------------------
# Main
# -----------------------------
def main():
    model = YOLO(MODEL_PATH)

    # (small speed win)
    try:
        model.fuse()
    except Exception:
        pass

    device = "mps" if USE_MPS_IF_AVAILABLE else None

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    last_ball = None   # (cx, cy) in inference-scale coords
    miss = 0
    paused = False

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                break

        # Resize for inference (much faster)
        infer_frame, scale = resize_keep_aspect(frame, INFER_W)

        # 1 YOLO call for both person + ball
        try:
            res = model(
                infer_frame,
                verbose=False,
                conf=CONF,
                iou=IOU,
                classes=[PERSON_CLASS_ID, BALL_CLASS_ID],
                imgsz=IMGSZ,
                device=device
            )[0]
        except Exception as e:
            # If MPS causes issues, fall back to CPU automatically
            if device == "mps":
                print("[WARN] MPS failed, falling back to CPU:", e)
                device = None
                continue
            else:
                raise

        players = []
        ball_dets = []

        for box in res.boxes:
            cls = int(box.cls[0])
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
            conf = float(box.conf[0])
            cx, cy = center_xyxy(x1, y1, x2, y2)

            if cls == PERSON_CLASS_ID:
                players.append((conf, (x1, y1, x2, y2)))
            elif cls == BALL_CLASS_ID:
                ball_dets.append((conf, (x1, y1, x2, y2), (cx, cy)))

        # Choose ball detection (tracking + gating)
        chosen = None
        if ball_dets:
            if last_ball is None:
                # pick highest confidence to initialize
                chosen = max(ball_dets, key=lambda t: t[0])
            else:
                # pick best within gating distance
                best = None
                best_score = -1e18
                for bconf, bbox, (cx, cy) in ball_dets:
                    dist = float(np.hypot(cx - last_ball[0], cy - last_ball[1]))
                    if dist > GATE_PX:
                        continue
                    # score: conf - small distance penalty
                    score = bconf - 0.0025 * dist
                    if score > best_score:
                        best_score = score
                        best = (bconf, bbox, (cx, cy), dist)
                if best is not None:
                    bconf, bbox, (cx, cy), dist = best
                    chosen = (bconf, bbox, (cx, cy))
                else:
                    chosen = None

        if chosen is not None:
            bconf, bbox, (cx, cy) = chosen
            last_ball = (cx, cy)
            miss = 0
        else:
            miss += 1
            if miss >= MAX_MISSES:
                last_ball = None
                miss = 0

        # ---- Draw on original frame ----
        H0, W0 = frame.shape[:2]
        # map inference coords back to original
        inv_scale = 1.0 / scale

        # HUD
        cv2.rectangle(frame, (0, 0), (W0, 34), (0, 0, 0), -1)
        hud = f"model={MODEL_PATH} device={'mps' if device=='mps' else 'cpu'} infer_w={INFER_W} imgsz={IMGSZ} balls={len(ball_dets)} miss={miss} tracked={'yes' if last_ball else 'no'}"
        cv2.putText(frame, hud, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # players (optional)
        if DRAW_PLAYERS:
            # only draw a few biggest to reduce clutter
            players_sorted = sorted(players, key=lambda t: (t[1][2]-t[1][0])*(t[1][3]-t[1][1]), reverse=True)[:10]
            for pconf, (x1, y1, x2, y2) in players_sorted:
                X1, Y1, X2, Y2 = [int(v * inv_scale) for v in (x1, y1, x2, y2)]
                cv2.rectangle(frame, (X1, Y1), (X2, Y2), (0, 140, 0), 2)

        # all ball dets (optional)
        if DRAW_ALL_BALL_DETS:
            for bconf, (x1, y1, x2, y2), (cx, cy) in ball_dets:
                X1, Y1, X2, Y2 = [int(v * inv_scale) for v in (x1, y1, x2, y2)]
                cv2.rectangle(frame, (X1, Y1), (X2, Y2), (0, 180, 0), 1)
                cv2.putText(frame, f"{bconf:.2f}", (X1, max(12, Y1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 0), 1)

        # chosen / tracked ball
        if last_ball is not None:
            bx, by = last_ball
            BX, BY = int(bx * inv_scale), int(by * inv_scale)
            cv2.circle(frame, (BX, BY), 6, (255, 255, 0), -1)
            cv2.putText(frame, "ball(track)", (BX + 8, BY - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1 if not paused else 0) & 0xFF
        if key == ord('q'):
            break
        if key == ord(' '):
            paused = not paused
        if key == ord('r'):
            last_ball = None
            miss = 0

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()