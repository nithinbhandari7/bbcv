import os
import sys
import json
import numpy as np
import cv2

# Usage:
# python src/mapping/calibrate_homography.py runs/release_frames/shot_0001_release.jpg assets/court_full.png

def click_points(img, window_name, num_points=8):
    pts = []
    disp = img.copy()

    def mouse_cb(event, x, y, flags, param):
        nonlocal disp
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < num_points:
            pts.append([x, y])
            cv2.circle(disp, (x, y), 6, (0, 255, 0), -1)
            cv2.putText(disp, str(len(pts)), (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_cb)

    while True:
        cv2.imshow(window_name, disp)
        key = cv2.waitKey(20) & 0xFF

        # r = reset
        if key == ord('r'):
            pts = []
            disp = img.copy()

        # q = quit early
        if key == ord('q'):
            break

        if len(pts) >= num_points:
            break

    cv2.destroyWindow(window_name)
    return np.array(pts, dtype=np.float32)

def main():
    if len(sys.argv) < 3:
        print("Usage: python src/mapping/calibrate_homography.py <release_frame.jpg> <court_full.png>")
        return 1

    release_path = sys.argv[1]
    court_path = sys.argv[2]

    release = cv2.imread(release_path)
    court = cv2.imread(court_path)

    if release is None:
        print("Could not read release frame:", release_path)
        return 1
    if court is None:
        print("Could not read court image:", court_path)
        return 1

    clip_id = os.path.basename(release_path).split("_release")[0]
    out_dir = "runs/homography"
    os.makedirs(out_dir, exist_ok=True)

    print("\nClick 4 matching points in BOTH images.")
    print("Tips: pick line intersections/corners that are clearly visible.")
    print("Controls: r = reset points, q = quit\n")

    # Click in broadcast (source)
    src_pts = click_points(release, f"SOURCE (broadcast) - {clip_id}", num_points=8)
    if len(src_pts) < 4:
        print("Not enough points clicked in source.")
        return 1

    # Click in template (destination)
    dst_pts = click_points(court, f"DEST (court template) - {clip_id}", num_points=8)
    if len(dst_pts) < 4:
        print("Not enough points clicked in dest.")
        return 1

    H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None:
        print("Homography failed.")
        return 1

    # Save
    npy_path = os.path.join(out_dir, f"H_{clip_id}.npy")
    json_path = os.path.join(out_dir, f"pts_{clip_id}.json")
    np.save(npy_path, H)
    with open(json_path, "w") as f:
        json.dump({"clip_id": clip_id, "src_pts": src_pts.tolist(), "dst_pts": dst_pts.tolist()}, f, indent=2)

    print(f"\n✅ Saved homography: {npy_path}")
    print(f"✅ Saved points:     {json_path}")

    # Quick sanity: warp the release frame onto the template size
    warped = cv2.warpPerspective(release, H, (court.shape[1], court.shape[0]))
    preview = cv2.addWeighted(court, 0.6, warped, 0.4, 0)
    cv2.imshow("Sanity Check: court (base) + warped broadcast overlay", preview)
    print("Close the preview window to finish.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())