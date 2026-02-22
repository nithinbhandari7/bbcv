import cv2
import numpy as np
from ultralytics import YOLO

VIDEO_PATH = "data/clips/shot_0001.mp4"
MODEL_PATH = "yolov8m.pt"   # m/l recommended for small objects in broadcast

PERSON_CLASS_ID = 0
BALL_CLASS_ID = 32          # COCO "sports ball"

# Detection params
PERSON_CONF = 0.25
BALL_CONF = 0.01            # low recall; ROIs will reduce false positives
IOU = 0.60
IMGSZ = 1536                # helps a lot at 1080p

# ROI params (upper body / hands zone)
ROI_EXPAND_X = 0.15         # expand player box horizontally
ROI_TOP_FRAC = 0.05         # start a bit below head (avoid scoreboard/crowd)
ROI_BOTTOM_FRAC = 0.60      # upper 60% of bbox (hands + torso)
MIN_ROI_SIZE = 60           # ignore tiny ROIs

# Tracking-ish
GATE_PX = 180               # max jump from last ball position
MAX_MISSES = 20             # after misses, forget last position

DRAW = True

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def xyxy_to_int(b):
    return tuple(int(x) for x in b)

def box_center(x1, y1, x2, y2):
    return (0.5*(x1+x2), 0.5*(y1+y2))

def build_hand_roi(px1, py1, px2, py2, W, H):
    bw = px2 - px1
    bh = py2 - py1

    # expand horizontally
    ex = ROI_EXPAND_X * bw
    x1 = clamp(px1 - ex, 0, W)
    x2 = clamp(px2 + ex, 0, W)

    # take upper portion of bbox
    y1 = clamp(py1 + ROI_TOP_FRAC * bh, 0, H)
    y2 = clamp(py1 + ROI_BOTTOM_FRAC * bh, 0, H)

    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    if (x2 - x1) < MIN_ROI_SIZE or (y2 - y1) < MIN_ROI_SIZE:
        return None
    return (x1, y1, x2, y2)

def main():
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    last_ball = None  # (cx, cy)
    misses = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        H, W = frame.shape[:2]

        # --- 1) Detect players on full frame ---
        res_p = model(frame, verbose=False, conf=PERSON_CONF, iou=IOU, classes=[PERSON_CLASS_ID], imgsz=IMGSZ)[0]
        players = []
        for b in res_p.boxes:
            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().tolist()
            players.append((float(x1), float(y1), float(x2), float(y2), float(b.conf[0])))

        # Sort biggest first (often helps prioritize relevant players)
        players.sort(key=lambda t: (t[2]-t[0])*(t[3]-t[1]), reverse=True)

        # Build ROIs
        rois = []
        for (x1, y1, x2, y2, conf) in players:
            roi = build_hand_roi(x1, y1, x2, y2, W, H)
            if roi is not None:
                rois.append(roi)

        # --- 2) Ball detection inside each ROI ---
        ball_candidates = []
        for (rx1, ry1, rx2, ry2) in rois:
            roi_img = frame[ry1:ry2, rx1:rx2]
            if roi_img.size == 0:
                continue

            res_b = model(roi_img, verbose=False, conf=BALL_CONF, iou=IOU, classes=[BALL_CLASS_ID], imgsz=IMGSZ)[0]
            for bb in res_b.boxes:
                bx1, by1, bx2, by2 = bb.xyxy[0].cpu().numpy().tolist()
                bconf = float(bb.conf[0])

                # shift ROI coords back to full-frame coords
                bx1 += rx1; bx2 += rx1
                by1 += ry1; by2 += ry1

                cx, cy = box_center(bx1, by1, bx2, by2)

                # optional gating against last known ball
                if last_ball is not None:
                    dist = float(np.hypot(cx - last_ball[0], cy - last_ball[1]))
                    if dist > GATE_PX:
                        continue
                else:
                    dist = 0.0

                # score: prefer high conf and closeness to last_ball (if present)
                score = bconf - 0.0025 * dist
                ball_candidates.append((score, bconf, (bx1, by1, bx2, by2), (cx, cy)))

        chosen = None
        if ball_candidates:
            chosen = max(ball_candidates, key=lambda t: t[0])
            _, bconf, bbox, (cx, cy) = chosen
            last_ball = (cx, cy)
            misses = 0
        else:
            misses += 1
            if misses >= MAX_MISSES:
                last_ball = None
                misses = 0

        # --- Draw ---
        if DRAW:
            # draw player ROIs
            for (rx1, ry1, rx2, ry2) in rois[:12]:
                cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (80, 80, 80), 2)

            # draw chosen ball
            if chosen is not None:
                _, bconf, (bx1, by1, bx2, by2), (cx, cy) = chosen
                cv2.rectangle(frame, xyxy_to_int((bx1, by1, bx2, by2)), (0, 255, 255), 2)
                cv2.circle(frame, (int(cx), int(cy)), 5, (255, 255, 0), -1)
                cv2.putText(frame, f"BALL {bconf:.2f}", (int(bx1), max(20, int(by1)-8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            elif last_ball is not None:
                cv2.circle(frame, (int(last_ball[0]), int(last_ball[1])), 5, (255, 255, 0), -1)
                cv2.putText(frame, "ball (last)", (int(last_ball[0])+8, int(last_ball[1])-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            hud = f"model={MODEL_PATH} imgsz={IMGSZ} ball_conf={BALL_CONF} rois={len(rois)} last_ball={'yes' if last_ball else 'no'}"
            cv2.rectangle(frame, (0, 0), (W, 35), (0, 0, 0), -1)
            cv2.putText(frame, hud, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow("Player-ROI Ball Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()