import os
import json
import glob
import cv2
import numpy as np
from ultralytics import YOLO

RELEASE_DIR = "runs/release_frames"
OUT_PATH = "data/labels/shooter_labels.json"

# Swap to yolov8s.pt (better recall) if you can afford it:
MODEL_PATH = "yolov8s.pt"   # was: yolov8n.pt

# Detection tuning (recall-focused for labeling)
DET_CONF = 0.15
DET_IOU = 0.60

# If you have nearby frames saved, we’ll try to include them automatically.
# This is “best effort” based on filename patterns.
USE_NEIGHBOR_FRAMES = True
NEIGHBOR_OFFSETS = [-2, -1, 1, 2]  # tries _t-2, _t-1, _t+1, _t+2 etc.
MAX_EXTRA_SAME_PREFIX = 6          # also grabs a few other frames with same clip_id prefix

WINDOW_NAME = "Label Shooter (Release Frame)"

# Controls:
#   n / → : next clip
#   p / ← : previous clip
#   [ or a: previous detection
#   ] or d: next detection
#   s: save selected detection as shooter for this clip
#   0-9: quick select detection index (0-9)
#   r: clear shooter label for this clip
#   m: manual box mode (click-drag to create a bbox)
#   esc: cancel manual mode
#   q: quit (saves automatically)

# -----------------------------
# IO helpers
# -----------------------------
def load_existing(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}

def save_labels(labels, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(labels, f, indent=2)
    os.replace(tmp, path)

# -----------------------------
# Detection helpers
# -----------------------------
def run_person_dets(model, img, conf=DET_CONF, iou=DET_IOU):
    """Run YOLO on an image and return person detections."""
    res = model(img, verbose=False, conf=conf, iou=iou)[0]
    dets = []
    for box in res.boxes:
        if int(box.cls[0]) == 0:  # person
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
            dets.append({
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "conf": float(box.conf[0]),
            })
    return dets

def parse_clip_id_from_fname(fname):
    clip_id = fname
    for suf in ["_release.jpg", "_release.png", "_release.jpeg"]:
        if clip_id.endswith(suf):
            clip_id = clip_id.replace(suf, "")
            break
    return clip_id

def neighbor_candidates(release_path, clip_id):
    """
    Best-effort: find neighboring frames on disk for the same clip.
    Works with patterns like:
      shot_123_release.jpg
      shot_123_t-1.jpg / shot_123_t+1.jpg
      shot_123_pre.jpg / shot_123_post.jpg
      shot_123_frame_000123.jpg
    If nothing exists, returns [release_path].
    """
    dirp = os.path.dirname(release_path)
    base, ext = os.path.splitext(os.path.basename(release_path))
    ext_lower = ext.lower()

    paths = [release_path]

    # 1) Common neighbor naming: replace "_release" with "_t-1" etc
    if "_release" in base:
        for off in NEIGHBOR_OFFSETS:
            tag = f"_t{off:+d}"  # _t-1, _t+2
            cand_base = base.replace("_release", tag)
            cand = os.path.join(dirp, cand_base + ext_lower)
            if os.path.exists(cand) and cand not in paths:
                paths.append(cand)

        # also try _pre / _post (common)
        for tag in ["_pre", "_post", "_before", "_after"]:
            cand_base = base.replace("_release", tag)
            for e in [ext_lower, ".jpg", ".png", ".jpeg"]:
                cand = os.path.join(dirp, cand_base + e)
                if os.path.exists(cand) and cand not in paths:
                    paths.append(cand)
                    break

    # 2) Grab a few other frames with the same clip_id prefix (best effort)
    # This helps if you saved things like shot_123_frame_001.jpg alongside release.
    same_prefix = sorted(glob.glob(os.path.join(dirp, f"{clip_id}*.*")))
    # Keep only images
    same_prefix = [p for p in same_prefix if os.path.splitext(p)[1].lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]]
    # Remove the release path itself
    same_prefix = [p for p in same_prefix if os.path.abspath(p) != os.path.abspath(release_path)]

    # Add up to MAX_EXTRA_SAME_PREFIX extras
    for p in same_prefix[:MAX_EXTRA_SAME_PREFIX]:
        if p not in paths:
            paths.append(p)

    # If we found nothing besides release, just return release
    return paths

# -----------------------------
# Drawing
# -----------------------------
def draw(frame, dets, selected_idx, clip_id, saved_idx, manual_mode=False, manual_box=None):
    vis = frame.copy()
    h, w = vis.shape[:2]

    # header bar
    bar_h = 70
    cv2.rectangle(vis, (0, 0), (w, bar_h), (0, 0, 0), -1)

    sel_txt = selected_idx if selected_idx is not None else "None"
    mm = "ON" if manual_mode else "off"
    text = f"{clip_id} | dets={len(dets)} | selected={sel_txt} | saved={saved_idx} | manual={mm}"
    cv2.putText(vis, text, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)

    # draw detections
    for i, d in enumerate(dets):
        x1, y1, x2, y2 = d["bbox"]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        is_sel = (selected_idx == i)
        color = (0, 255, 255) if is_sel else (0, 255, 0)
        thickness = 4 if is_sel else 2

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)

        # label background
        src = d.get("src", "")
        conf = d.get("conf", None)
        label = f"{i}"
        if conf is not None:
            label += f" ({conf:.2f})"
        if src:
            label += f" [{src}]"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        y_top = max(0, y1 - th - 10)
        cv2.rectangle(vis, (x1, y_top), (x1 + tw + 10, y_top + th + 8), color, -1)
        cv2.putText(vis, label, (x1 + 5, y_top + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

    # draw manual box preview
    if manual_box is not None:
        x1, y1, x2, y2 = manual_box
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(vis, "manual box", (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # footer help
    help_lines = [
        "Keys: [ / a prev det | ] / d next det | 0-9 select | s save | r reset | n next | p prev | m manual box | esc cancel manual | q quit",
        f"Detection: MODEL={os.path.basename(MODEL_PATH)} conf={DET_CONF} iou={DET_IOU} | neighbor_frames={'ON' if USE_NEIGHBOR_FRAMES else 'off'}",
    ]
    y = h - 15
    for line in help_lines:
        cv2.putText(vis, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        y -= 18

    return vis

# -----------------------------
# Manual bbox mouse handling
# -----------------------------
class ManualBoxState:
    def __init__(self):
        self.enabled = False
        self.dragging = False
        self.start = None
        self.current = None
        self.final_box = None  # (x1,y1,x2,y2) when completed

    def reset(self):
        self.dragging = False
        self.start = None
        self.current = None
        self.final_box = None

def clamp_box(x1, y1, x2, y2, w, h):
    x1 = int(max(0, min(x1, w - 1)))
    x2 = int(max(0, min(x2, w - 1)))
    y1 = int(max(0, min(y1, h - 1)))
    y2 = int(max(0, min(y2, h - 1)))
    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])
    return x1, y1, x2, y2

def mouse_cb(event, x, y, flags, param):
    st: ManualBoxState = param["state"]
    img_shape = param["shape"]  # (h, w)
    h, w = img_shape

    if not st.enabled:
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        st.dragging = True
        st.start = (x, y)
        st.current = (x, y)
        st.final_box = None

    elif event == cv2.EVENT_MOUSEMOVE and st.dragging:
        st.current = (x, y)

    elif event == cv2.EVENT_LBUTTONUP and st.dragging:
        st.dragging = False
        st.current = (x, y)
        x1, y1 = st.start
        x2, y2 = st.current
        x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, w, h)
        # Ignore tiny boxes
        if (x2 - x1) >= 5 and (y2 - y1) >= 5:
            st.final_box = (x1, y1, x2, y2)

# -----------------------------
# Main
# -----------------------------
def main():
    model = YOLO(MODEL_PATH)

    release_paths = sorted(glob.glob(os.path.join(RELEASE_DIR, "shot_*_release.jpg")))
    if not release_paths:
        release_paths = sorted(glob.glob(os.path.join(RELEASE_DIR, "shot_*_release.*")))

    if not release_paths:
        print(f"No release frames found in {RELEASE_DIR}")
        return 1

    labels = load_existing(OUT_PATH)

    idx = 0
    selected_det = None

    manual = ManualBoxState()
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    while True:
        release_path = release_paths[idx]
        fname = os.path.basename(release_path)
        clip_id = parse_clip_id_from_fname(fname)

        frame = cv2.imread(release_path)
        if frame is None:
            print("Could not read:", release_path)
            idx = min(idx + 1, len(release_paths) - 1)
            continue

        h, w = frame.shape[:2]

        # set mouse callback for manual mode
        cv2.setMouseCallback(WINDOW_NAME, mouse_cb, param={"state": manual, "shape": (h, w)})

        # Run detection (single frame or multi-frame)
        dets = []
        if USE_NEIGHBOR_FRAMES:
            cand_paths = neighbor_candidates(release_path, clip_id)
            for p in cand_paths:
                img2 = cv2.imread(p)
                if img2 is None:
                    continue
                dets2 = run_person_dets(model, img2, conf=DET_CONF, iou=DET_IOU)
                src = os.path.basename(p)
                for d in dets2:
                    d["src"] = src
                dets.extend(dets2)
        else:
            dets = run_person_dets(model, frame, conf=DET_CONF, iou=DET_IOU)
            for d in dets:
                d["src"] = os.path.basename(release_path)

        # Sort left-to-right for stable numbering
        dets.sort(key=lambda d: (d["bbox"][0] + d["bbox"][2]) / 2.0)

        saved_idx = labels.get(clip_id, {}).get("shooter_det_idx", None)

        if selected_det is None and len(dets) > 0:
            selected_det = saved_idx if (saved_idx is not None and saved_idx < len(dets)) else 0

        # Manual box preview
        manual_preview = None
        if manual.enabled and manual.start is not None and manual.current is not None:
            x1, y1 = manual.start
            x2, y2 = manual.current
            manual_preview = clamp_box(x1, y1, x2, y2, w, h)

        vis = draw(
            frame,
            dets,
            selected_det,
            clip_id,
            saved_idx,
            manual_mode=manual.enabled,
            manual_box=manual_preview
        )
        cv2.imshow(WINDOW_NAME, vis)

        key = cv2.waitKey(20) & 0xFF  # 20ms so mouse preview updates

        # If a manual box was completed, add it as a new detection and auto-select it
        if manual.final_box is not None:
            x1, y1, x2, y2 = manual.final_box
            dets.append({"bbox": [x1, y1, x2, y2], "conf": 1.0, "src": "manual"})
            dets.sort(key=lambda d: (d["bbox"][0] + d["bbox"][2]) / 2.0)
            selected_det = next(
                (i for i, d in enumerate(dets) if d.get("src") == "manual" and d["bbox"] == [x1, y1, x2, y2]),
                len(dets) - 1
            )
            manual.reset()
            manual.enabled = False  # turn off after one box

        # quit
        if key == ord('q'):
            save_labels(labels, OUT_PATH)
            cv2.destroyAllWindows()
            print("Saved:", OUT_PATH)
            return 0

        # cancel manual mode
        if key == 27:  # ESC
            manual.reset()
            manual.enabled = False

        # toggle manual mode
        if key == ord('m'):
            manual.enabled = not manual.enabled
            manual.reset()

        # next / prev clip (arrow keys vary by OS; keep your old codes + letter keys)
        if key in [ord('n'), 83]:  # 'n' or right arrow-ish
            save_labels(labels, OUT_PATH)
            idx = min(idx + 1, len(release_paths) - 1)
            selected_det = None
            manual.reset()
            manual.enabled = False
            continue
        if key in [ord('p'), 81]:  # 'p' or left arrow-ish
            save_labels(labels, OUT_PATH)
            idx = max(idx - 1, 0)
            selected_det = None
            manual.reset()
            manual.enabled = False
            continue

        # cycle detections
        if key in [ord(']'), ord('d')]:
            if dets:
                selected_det = 0 if selected_det is None else (selected_det + 1) % len(dets)
        if key in [ord('['), ord('a')]:
            if dets:
                selected_det = 0 if selected_det is None else (selected_det - 1) % len(dets)

        # quick select 0-9
        if ord('0') <= key <= ord('9'):
            num = key - ord('0')
            if dets and num < len(dets):
                selected_det = num

        # save shooter
        if key == ord('s'):
            if dets and selected_det is not None and selected_det < len(dets):
                labels[clip_id] = {
                    "release_frame_img": release_path,
                    "shooter_det_idx": int(selected_det),
                    "shooter_bbox_xyxy": dets[selected_det]["bbox"],
                    "shooter_conf": dets[selected_det].get("conf", None),
                    "shooter_src_frame": dets[selected_det].get("src", None),
                    "det_conf_used": DET_CONF,
                    "det_iou_used": DET_IOU,
                    "model": MODEL_PATH,
                    "neighbor_frames_used": bool(USE_NEIGHBOR_FRAMES),
                }
                save_labels(labels, OUT_PATH)
                print(f"[SAVED] {clip_id} shooter_det_idx={selected_det}")

        # reset shooter label
        if key == ord('r'):
            if clip_id in labels:
                del labels[clip_id]
                save_labels(labels, OUT_PATH)
                print(f"[RESET] {clip_id}")

if __name__ == "__main__":
    raise SystemExit(main())