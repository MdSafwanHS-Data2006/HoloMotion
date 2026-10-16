"""
HoloMotion - Real-Time Object Detection and Tracking

What this program does:
1. Opens video input from either a live webcam or a video file.
2. Detects objects in each video frame using:
   - Standard YOLOv8 (for 80 common objects: phones, cups, bottles, books, scissors, etc.)
   - OR YOLO-World (Open-Vocabulary mode to detect pens, pencils, glasses, watches, etc.)
3. Uses ByteTrack to assign and maintain consistent tracking IDs (#1, #2, ...) across frames.
4. Optimized for small objects held in front of a person/face:
   - Lower default confidence threshold (0.20) so handheld items are not filtered out.
   - Overlap-friendly IoU so small items in front of the body are not swallowed by the person box.
   - Interactive hotkeys: '[' and ']' to adjust confidence live, and 'P' to toggle the person box.
5. Displays real-time FPS, object labels, and tracking IDs, and quits cleanly on 'Q'.
"""

import argparse
import os
import sys
import time
import cv2
from ultralytics import YOLO

# Ensure Qt uses xcb platform plugin on Linux desktops (avoids Wayland crashes)
if "QT_QPA_PLATFORM" not in os.environ:
    os.environ["QT_QPA_PLATFORM"] = "xcb"

# Redirect cache folders to writable temp paths on Fedora Linux
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


def parse_arguments():
    """
    Reads command-line options for video source, model selection, and detection tuning.
    """
    parser = argparse.ArgumentParser(
        description="Object Detection and Tracking (YOLOv8 + ByteTrack)"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video source: webcam index (e.g. '0') or video file path (default: '0').",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.20,
        help="Detection confidence threshold (default: 0.20, lower values help detect small objects like pens).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="assets/yolov8n.pt",
        help="Path to YOLO model weights (default: 'assets/yolov8n.pt').",
    )
    parser.add_argument(
        "--world",
        action="store_true",
        help="Enable YOLO-World open-vocabulary mode to detect pens, pencils, glasses, and custom objects.",
    )
    parser.add_argument(
        "--classes",
        type=str,
        default="",
        help="Comma-separated list of custom classes for YOLO-World (e.g. 'pen, pencil, phone, person, cup').",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image resolution (default: 640; higher resolutions detect small objects better).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Compute device for inference (default: 'cpu').",
    )
    parser.add_argument(
        "--no-loop",
        action="store_true",
        help="Do not loop video files when playback reaches the end.",
    )
    return parser.parse_args()


def load_model(args):
    """
    Loads either standard YOLOv8 or YOLO-World with custom vocabulary.
    """
    device = getattr(args, "device", "cpu")
    use_world = args.world or bool(args.classes) or "world" in args.model.lower()

    if use_world:
        from ultralytics import YOLOWorld

        world_model_path = args.model if "world" in args.model.lower() else "assets/yolov8s-worldv2.pt"
        if not os.path.exists(world_model_path):
            world_model_path = "yolov8s-worldv2.pt"

        print(f"[Info] Loading open-vocabulary YOLO-World model from '{world_model_path}' on {device.upper()}...")
        model = YOLOWorld(world_model_path)

        # Define custom classes: includes pens, pencils, glasses, and common desk items
        if args.classes:
            target_classes = [c.strip() for c in args.classes.split(",") if c.strip()]
        else:
            target_classes = [
                "pen",
                "pencil",
                "ballpoint pen",
                "cell phone",
                "glasses",
                "spectacles",
                "watch",
                "cup",
                "bottle",
                "book",
                "scissors",
                "mouse",
                "laptop",
                "person",
                "hand",
                "face",
            ]

        print(f"[Info] Active custom detection classes: {target_classes}")
        model.set_classes(target_classes)
        return model, True

    # Standard YOLOv8 model (COCO 80 classes)
    model_path = args.model
    if not os.path.exists(model_path):
        model_path = "yolov8n.pt"

    print(f"[Info] Loading standard YOLO model from '{model_path}' on {device.upper()}...")
    model = YOLO(model_path)
    print("[Info] Standard model loaded. (Detects phone, cup, bottle, book, scissors, laptop, etc.)")
    print("[Tip] To detect pens, pencils, and glasses, launch with '--world'!")
    return model, False


def create_tracker():
    """
    Initializes a ByteTrack tracker with optimized thresholds for small/handheld items.
    """
    from ultralytics.trackers.byte_tracker import BYTETracker
    from types import SimpleNamespace

    return BYTETracker(
        SimpleNamespace(
            tracker_type="bytetrack",
            track_high_thresh=0.20,
            track_low_thresh=0.08,
            new_track_thresh=0.15,
            track_buffer=30,
            match_thresh=0.8,
            fuse_score=True,
        )
    )


def open_video_source(source_str):
    """
    Opens webcam or video file with clear error reporting.
    """
    if source_str.isdigit():
        camera_index = int(source_str)
        print(f"[Info] Opening webcam device at index {camera_index}...")
        cap = cv2.VideoCapture(camera_index)
        is_file = False
    elif source_str.startswith("/dev/video"):
        print(f"[Info] Opening webcam device at '{source_str}'...")
        cap = cv2.VideoCapture(source_str)
        is_file = False
    else:
        if not os.path.exists(source_str):
            print(f"[Error] Video file not found: '{source_str}'")
            return None, False
        print(f"[Info] Opening video file: '{source_str}'...")
        cap = cv2.VideoCapture(source_str)
        is_file = True

    if not cap.isOpened():
        print(f"[Error] Could not open video source: '{source_str}'.")
        return None, False

    return cap, is_file


def draw_detection(frame, box_coords, label, score, track_id=None):
    """
    Draws a distinct bounding box and label tag for a detected object.
    """
    x1, y1, x2, y2 = box_coords

    # Color code: Amber/Gold for small items (pens, phones, glasses), Teal for others
    is_small_item = any(k in label.lower() for k in ["pen", "pencil", "phone", "glass", "watch", "cup", "book"])
    box_color = (0, 215, 255) if is_small_item else (0, 255, 180)
    tag_color = (0, 160, 210) if is_small_item else (0, 160, 110)

    # 1. Bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

    # 2. Text caption
    id_str = f"#{track_id} " if track_id is not None else ""
    caption = f"{id_str}{label} {int(score * 100)}%"

    # 3. Background label tag (draw above the box if space permits, else inside top)
    (text_w, text_h), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
    tag_top = (y1 - text_h - 8) if (y1 - text_h - 8) >= 0 else y1
    cv2.rectangle(
        frame,
        (x1, tag_top),
        (x1 + text_w + 8, tag_top + text_h + 8),
        tag_color,
        -1,
    )

    # 4. Text
    cv2.putText(
        frame,
        caption,
        (x1 + 4, tag_top + text_h + 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )


def draw_hud(frame, fps, object_count, confidence, is_world_model, show_person):
    """
    Draws a clean top status header showing FPS, object count, and live settings.
    """
    h, w = frame.shape[:2]

    # Top status bar
    cv2.rectangle(frame, (0, 0), (w, 36), (20, 20, 20), -1)

    model_label = "YOLO-World (Pen/Custom)" if is_world_model else "YOLOv8 (Standard COCO)"
    person_tag = "ON" if show_person else "MUTED"
    hud_left = f"FPS: {int(fps)} | Tracked: {object_count} | Conf: {confidence:.2f} | Person: {person_tag}"

    cv2.putText(
        frame,
        hud_left,
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (0, 255, 200),
        1,
        cv2.LINE_AA,
    )

    # Align model badge to top right dynamically
    (mw, _), _ = cv2.getTextSize(model_label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
    label_x = max(12, w - mw - 16)
    cv2.putText(
        frame,
        model_label,
        (label_x, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (0, 215, 255) if is_world_model else (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    # Bottom helper bar
    cv2.rectangle(frame, (0, h - 28), (w, h), (20, 20, 20), -1)
    helper_text = "[Q] Quit  |  [[ / ]] Adjust Conf  |  [P] Toggle Person Box (Helps see handheld items)"
    cv2.putText(
        frame,
        helper_text,
        (12, h - 9),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )


def main():
    args = parse_arguments()

    print("=" * 65)
    print("  Object Detection and Tracking (YOLOv8 / YOLO-World + ByteTrack)")
    print("=" * 65)

    # 1. Load Model (Standard or YOLO-World)
    try:
        model, is_world_model = load_model(args)
    except Exception as e:
        print(f"[Error] Failed to load model: {e}")
        sys.exit(1)

    # 2. Initialize ByteTrack tracker
    tracker = create_tracker()

    # 3. Open Video Source
    cap, is_file = open_video_source(args.source)
    if cap is None:
        sys.exit(1)

    # 4. Create Display Window
    window_name = "Object Detection and Tracking"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    current_conf = args.conf
    show_person = True  # Toggleable with 'P'

    print("\n[Live Keyboard Controls in Video Window]")
    print("  [Q / ESC] : Quit the application")
    print("  [[ / ]]   : Lower / Raise confidence threshold (allows faint objects to appear)")
    print("  [P]       : Toggle Person Bounding Box (hides person box so held items stand out)")
    print("=" * 65 + "\n")

    fps = 0.0
    frame_count = 0

    try:
        while True:
            t_start = time.time()

            ret, frame = cap.read()
            if not ret:
                if is_file and not getattr(args, "no_loop", False):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                if not ret:
                    print("[Info] End of video stream.")
                    break

            frame_count += 1

            # 1. Run YOLO object detection on the current frame
            # (predict ensures ALL detected objects meeting the confidence threshold are preserved)
            results = model.predict(
                source=frame,
                conf=current_conf,
                iou=0.55,
                imgsz=args.imgsz,
                device=getattr(args, "device", "cpu"),
                verbose=False,
            )

            detected_count = 0

            if results and len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes

                # 2. Update ByteTrack tracker to maintain persistent IDs across frames
                track_ids = {}
                if len(boxes) > 0:
                    try:
                        tracker.args.new_track_thresh = min(0.20, max(0.05, current_conf))
                        tracks = tracker.update(boxes.cpu().numpy(), frame)
                        if len(tracks) > 0:
                            # tracks shape: (N, 8) -> [x1, y1, x2, y2, track_id, score, cls, orig_detection_idx]
                            track_ids = {int(t[7]): int(t[4]) for t in tracks}
                    except Exception:
                        track_ids = {}

                # 3. Render all detected objects
                # Tracked objects show their persistent #ID;
                # Newly appeared or transient objects are still displayed!
                for i, box in enumerate(boxes):
                    cls_id = int(box.cls[0].item())
                    label = model.names.get(cls_id, f"class_{cls_id}")

                    # If user toggled person off, hide the large person box
                    if not show_person and label.lower() == "person":
                        continue

                    detected_count += 1
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                    score = float(box.conf[0].item())
                    track_id = track_ids.get(i)

                    draw_detection(frame, (x1, y1, x2, y2), label, score, track_id)

            # Draw HUD overlay
            draw_hud(frame, fps, detected_count, current_conf, is_world_model, show_person)

            # Display frame
            cv2.imshow(window_name, frame)

            # Calculate FPS
            t_elapsed = time.time() - t_start
            fps = 1.0 / max(0.001, t_elapsed)

            # Keyboard Input
            key = cv2.waitKey(1) & 0xFF
            if key in [ord("q"), ord("Q"), 27]:
                print("[Info] Quit key pressed. Exiting...")
                break

            # Live confidence adjustment
            elif key == ord("["):
                current_conf = max(0.05, round(current_conf - 0.05, 2))
                print(f"[Settings] Lowered confidence threshold to: {current_conf:.2f}")

            elif key == ord("]"):
                current_conf = min(0.95, round(current_conf + 0.05, 2))
                print(f"[Settings] Raised confidence threshold to: {current_conf:.2f}")

            # Toggle Person BBox visibility
            elif key in [ord("p"), ord("P")]:
                show_person = not show_person
                status = "VISIBLE" if show_person else "MUTED"
                print(f"[Settings] Person bounding box: {status}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"[Info] Cleaned up after {frame_count} frames. Goodbye!")


if __name__ == "__main__":
    main()
