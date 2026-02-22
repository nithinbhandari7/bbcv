import cv2
import csv
import os
import sys


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python src/labeling/label_shot.py <path_to_clip.mp4>")
        return 1

    video_path = sys.argv[1]
    if not os.path.isfile(video_path):
        print(f"Error: file not found: {video_path}")
        return 1

    clip_id = os.path.splitext(os.path.basename(video_path))[0]
    labels_path = "data/labels/shots.csv"
    ensure_parent_dir(labels_path)

    # Labels we will collect
    release_frame = None
    result = None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: OpenCV could not open the video. Try re-encoding with ffmpeg.")
        return 1

    # Read the first frame so we can display something even before playing
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read any frames from the video.")
        cap.release()
        return 1

    # OpenCV is now at frame 1 (0-indexed). We'll track our own frame_idx.
    frame_idx = 0

    print("\nControls:")
    print("  p  : pause / play")
    print("  n  : next frame (when paused)")
    print("  b  : back 5 frames (when paused)")
    print("  r  : mark release frame")
    print("  m  : mark make")
    print("  x  : mark miss")
    print("  q  : quit (saves only if release + result are set)\n")

    paused = False

    while True:
        # If not paused, advance to the next frame
        if not paused:
            ret, frame = cap.read()
            if not ret:
                # End of video
                break
            frame_idx += 1

        display = frame.copy()

        # Overlay status text
        status_line_1 = f"Clip: {clip_id}"
        status_line_2 = f"Frame: {frame_idx}"
        status_line_3 = f"Release: {release_frame if release_frame is not None else '-'}"
        status_line_4 = f"Result: {result if result is not None else '-'}"
        status_line_5 = f"{'PAUSED' if paused else 'PLAYING'} (p=toggle, n=next, b=back)"

        y = 40
        for line in [status_line_1, status_line_2, status_line_3, status_line_4, status_line_5]:
            cv2.putText(display, line, (30, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            y += 35

        cv2.imshow("Label Shot", display)

        # When paused, wait indefinitely for a key; when playing, poll with small delay
        key = cv2.waitKey(30 if not paused else 0) & 0xFF

        if key == ord('p'):
            paused = not paused
            print("Paused" if paused else "Playing")

        elif key == ord('n') and paused:
            # Step forward one frame
            ret, next_frame = cap.read()
            if ret:
                frame = next_frame
                frame_idx += 1

        elif key == ord('b') and paused:
            # Step backward 5 frames (clamped to 0)
            new_pos = max(frame_idx - 5, 0)
            cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
            ret, back_frame = cap.read()
            if ret:
                frame = back_frame
                frame_idx = new_pos

        elif key == ord('r'):
            release_frame = frame_idx
            print(f"Release frame marked: {release_frame}")

        elif key == ord('m'):
            result = "make"
            print("Result: MAKE")

        elif key == ord('x'):
            result = "miss"
            print("Result: MISS")

        elif key == ord('q'):
            break

        # If the user closes the window, exit cleanly
        if cv2.getWindowProperty("Label Shot", cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()

    # Save label if complete
    if release_frame is not None and result is not None:
        file_exists = os.path.isfile(labels_path)

        with open(labels_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow(["clip_id", "release_frame", "result"])
            writer.writerow([clip_id, release_frame, result])

        print(f"✅ Saved label to {labels_path}: {clip_id}, {release_frame}, {result}")
        return 0

    print("⚠️ Incomplete label (need both release_frame + result). Nothing saved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
