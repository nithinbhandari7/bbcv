"""
ball_track.py

Ball detection + tracking for basketball clips using YOLO (COCO "sports ball" class)
and a Kalman filter + ROI cropping + shape/size filtering + dynamic gating.

Run:
  python src/detection/ball_track.py

Keys:
  q   quit
  space pause/unpause
  r   reset tracker (re-acquire from full frame)
  +/- increase/decrease ROI size
"""

import cv2
import numpy as np
from ultralytics import YOLO

# -----------------------------
# Config
# -----------------------------
VIDEO_PATH = "data/clips/shot_0001.mp4"

# Bigger model helps a LOT for small objects (ball)
MODEL_PATH = "yolov8m.pt"          # try yolov8l.pt if you can
BALL_CLASS_ID = 32                 # COCO "sports ball"

# Recall-focused detection
CONF = 0.02
IOU = 0.60
IMGSZ = 1280                       # 960 or 1280 usually better than default

# ROI cropping (after we have a track)
ROI_HALF = 240                     # press +/- to adjust while running
ROI_GROW_IF_LOST = True            # when ball is lost, expand ROI gradually

# Candidate filtering (tune per resolution/zoom)
# For 720p/1080p broadcast, these are reasonable starting points.
MIN_BALL_AREA = 10                 # too small => noise
MAX_BALL_AREA = 1400               # too big => not ball (tune if close-up zoom)
MAX_ASPECT = 2.8                   # ball boxes shouldn't be too stretched

# Gating (dynamic based on estimated speed)
BASE_GATE = 90
MAX_GATE = 320
SPEED_GAIN = 3.2                   # gate = BASE_GATE + SPEED_GAIN * speed_px
DIST_PENALTY = 0.0030              # score penalty per pixel distance

# Kalman noise (tune if jittery / laggy)
PROCESS_NOISE = 1e-2
MEAS_NOISE = 7e-1

# Behavior when not detected
MAX_MISSES_BEFORE_RESET = 25       # after too many misses, reset to full-frame search

DRAW_RAW_DETS = True
DRAW_CHOSEN = True
DRAW_TRACK = True

WINDOW_NAME = "Ball detection + tracking"


# -----------------------------
# Helpers
# -----------------------------
def make_kalman():
    # state: [x, y, vx, vy]
    kf = cv2.KalmanFilter(4, 2)

    kf.transitionMatrix = np.array(
        [[1, 0, 1, 0],
         [0, 1, 0, 1],
         [0, 0, 1, 0],
         [0, 0, 0, 1]],
        dtype=np.float32
    )

    kf.measurementMatrix = np.array(
        [[1, 0, 0, 0],
         [0, 1, 0, 0]],
        dtype=np.float32
    )

    kf.processNoiseCov = np.eye(4, dtype=np.float32) * PROCESS_NOISE
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * MEAS_NOISE
    kf.errorCovPost = np.eye(4, dtype=np.float32) * 1.0
    return kf


def bbox_center_xyxy(x1, y1, x2, y2):
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def crop_roi(frame, cx, cy, half):
    h, w = frame.shape[:2]
    x0 = int(clamp(cx - half, 0, w))
    y0 = int(clamp(cy - half, 0, h))
    x1 = int(clamp(cx + half, 0, w))
    y1 = int(clamp(cy + half, 0, h))
    roi = frame[y0:y1, x0:x1]
    return roi, x0, y0


def filter_ball_candidate(x1, y1, x2, y2):
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    area = bw * bh
    aspect = max(bw / bh, bh / bw)
    if area < MIN_BALL_AREA or area > MAX_BALL_AREA:
        return False
    if aspect > MAX_ASPECT:
        return False
    return True


# -----------------------------
# Main
# -----------------------------
def main():
    global ROI_HALF

    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    kf = make_kalman()
    has_init = False
    paused = False

    miss_count = 0
    roi_half_dynamic = ROI_HALF

    prev_track = None  # (x,y) for speed estimate

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                break

        # Predict position (even if not initialized, OpenCV returns something)
        pred = kf.predict()
        pred_x = float(pred[0, 0])
        pred_y = float(pred[1, 0])

        h, w = frame.shape[:2]

        # Dynamic gate based on speed
        speed_px = 0.0
        if prev_track is not None and has_init:
            speed_px = float(np.hypot(pred_x - prev_track[0], pred_y - prev_track[1]))

        gate_px = int(clamp(BASE_GATE + SPEED_GAIN * speed_px, BASE_GATE, MAX_GATE))

        # Decide detection region
        use_roi = has_init and miss_count < MAX_MISSES_BEFORE_RESET
        det_frame = frame
        x_off = 0
        y_off = 0

        if use_roi:
            # If we're missing, optionally expand ROI a bit to re-acquire
            half = roi_half_dynamic
            if ROI_GROW_IF_LOST and miss_count > 0:
                half = int(clamp(roi_half_dynamic + miss_count * 8, roi_half_dynamic, roi_half_dynamic * 2))

            det_frame, x_off, y_off = crop_roi(frame, pred_x, pred_y, half)

            # Draw ROI box
            cv2.rectangle(
                frame,
                (x_off, y_off),
                (x_off + det_frame.shape[1], y_off + det_frame.shape[0]),
                (60, 60, 60),
                2
            )

        # Run YOLO detection (sports ball)
        res = model(
            det_frame,
            verbose=False,
            conf=CONF,
            iou=IOU,
            classes=[BALL_CLASS_ID],
            imgsz=IMGSZ
        )[0]

        candidates = []
        for box in res.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
            conf = float(box.conf[0])

            # shift ROI coords -> full-frame coords
            x1 += x_off
            x2 += x_off
            y1 += y_off
            y2 += y_off

            # filter by size/shape
            if not filter_ball_candidate(x1, y1, x2, y2):
                continue

            cx, cy = bbox_center_xyxy(x1, y1, x2, y2)
            candidates.append((conf, (x1, y1, x2, y2), (cx, cy)))

        # Draw raw detections
        if DRAW_RAW_DETS:
            for conf, (x1, y1, x2, y2), (cx, cy) in candidates:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 180, 0), 1)
                cv2.circle(frame, (int(cx), int(cy)), 3, (0, 180, 0), -1)
                cv2.putText(
                    frame, f"{conf:.2f}",
                    (int(x1), max(12, int(y1) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 0), 1
                )

        chosen = None
        if candidates:
            if has_init:
                # Choose best candidate under gate with combined score
                best = None
                best_score = -1e18
                for conf, bbox, (cx, cy) in candidates:
                    dist = float(np.hypot(cx - pred_x, cy - pred_y))
                    if dist > gate_px:
                        continue
                    score = conf - DIST_PENALTY * dist
                    if score > best_score:
                        best_score = score
                        best = (conf, bbox, (cx, cy), dist)
                chosen = best
            else:
                # No init: pick highest confidence
                conf, bbox, (cx, cy) = max(candidates, key=lambda t: t[0])
                chosen = (conf, bbox, (cx, cy), 0.0)

        # Update Kalman filter
        if chosen is not None:
            conf, (x1, y1, x2, y2), (cx, cy), dist = chosen
            meas = np.array([[np.float32(cx)], [np.float32(cy)]], dtype=np.float32)

            if not has_init:
                kf.statePost = np.array([[cx], [cy], [0], [0]], dtype=np.float32)
                has_init = True
            else:
                kf.correct(meas)

            miss_count = 0
            roi_half_dynamic = ROI_HALF  # reset ROI size to base when we have a good detection

            if DRAW_CHOSEN:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
                cv2.putText(
                    frame, f"BALL {conf:.2f} d={dist:.0f} gate={gate_px}",
                    (int(x1), max(20, int(y1) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2
                )

        else:
            # No detection this frame
            miss_count += 1
            if miss_count >= MAX_MISSES_BEFORE_RESET:
                has_init = False
                prev_track = None

        # Draw filtered track point
        if DRAW_TRACK and has_init:
            fx = float(kf.statePost[0, 0])
            fy = float(kf.statePost[1, 0])
            cv2.circle(frame, (int(fx), int(fy)), 6, (255, 255, 0), -1)
            cv2.putText(
                frame, "track",
                (int(fx) + 8, int(fy) - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2
            )
            prev_track = (fx, fy)

        # HUD
        hud1 = f"{VIDEO_PATH} | model={MODEL_PATH} imgsz={IMGSZ} conf={CONF} iou={IOU}"
        hud2 = f"roi_half={ROI_HALF} gate={gate_px} misses={miss_count}/{MAX_MISSES_BEFORE_RESET} init={has_init}"
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 56), (0, 0, 0), -1)
        cv2.putText(frame, hud1, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(frame, hud2, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1 if not paused else 0) & 0xFF
        if key == ord('q'):
            break
        if key == ord(' '):
            paused = not paused
        if key == ord('r'):
            has_init = False
            miss_count = 0
            prev_track = None
        if key == ord('+') or key == ord('='):
            ROI_HALF = int(clamp(ROI_HALF + 30, 60, 800))
        if key == ord('-') or key == ord('_'):
            ROI_HALF = int(clamp(ROI_HALF - 30, 60, 800))

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()