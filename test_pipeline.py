"""
test_pipeline.py - Automated Verification Suite for V1: Object Detection and Tracking

Runs 5 targeted automated tests:
1. Python syntax & imports check
2. Standard YOLOv8 loading & CPU inference
3. Open-Vocabulary YOLO-World loading & custom vocabulary (pen, pencil, glasses, etc.)
4. Video file reading and ByteTrack tracking test
5. Webcam access and graceful error handling test
"""

import os
import sys
import numpy as np
import cv2

os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import main


def test_imports():
    print("[TEST 1/5] Checking Required Imports...")
    import cv2
    import torch
    import ultralytics
    try:
        import lap
    except ImportError:
        import lapx as lap
    import numpy as np
    print(f"  -> OpenCV: {cv2.__version__}")
    print(f"  -> PyTorch CPU: {torch.__version__} (CUDA available: {torch.cuda.is_available()})")
    print(f"  -> Ultralytics: {ultralytics.__version__}")
    print(f"  -> Lap tracker module: {lap.__file__}")
    print(f"  -> NumPy: {np.__version__}")
    print("  -> Imports test passed!")


def test_standard_model():
    print("[TEST 2/5] Testing Standard YOLOv8 Loading on CPU...")
    class Args:
        world = False
        classes = ""
        model = "assets/yolov8n.pt"
        device = "cpu"
    model, is_world = main.load_model(Args())
    assert model is not None, "Model failed to load"
    assert is_world is False, "Expected standard model"
    print(f"  -> Standard model loaded: {model.names[0]}, total classes: {len(model.names)}")
    print("  -> Standard model loading test passed!")
    return model


def test_yolo_world_custom_classes():
    print("[TEST 3/5] Testing YOLO-World Open-Vocabulary Model (Pens, Pencils, Glasses)...")
    class Args:
        world = True
        classes = "pen, pencil, glasses, cell phone, cup, person"
        model = "assets/yolov8s-worldv2.pt"
        device = "cpu"
    model, is_world = main.load_model(Args())
    assert model is not None, "YOLO-World failed to load"
    assert is_world is True, "Expected YOLO-World model"
    
    classes_list = list(model.names.values())
    print(f"  -> Custom classes loaded: {classes_list}")
    assert "pen" in classes_list, "'pen' missing from custom classes"
    assert "pencil" in classes_list, "'pencil' missing from custom classes"
    assert "glasses" in classes_list, "'glasses' missing from custom classes"
    print("  -> YOLO-World custom classes test passed!")


def test_video_file_processing(model):
    print("[TEST 4/5] Testing Video File Input & ByteTrack Tracking Loop...")
    sample_path = "assets/sample_video.mp4"
    assert os.path.exists(sample_path), "Sample video not found"

    cap, is_file = main.open_video_source(sample_path)
    assert cap is not None and is_file is True, "Failed to open sample video"

    frames_processed = 0
    detections_found = 0

    tracker = main.create_tracker()

    for _ in range(15):
        ret, frame = cap.read()
        if not ret:
            break
        frames_processed += 1
        results = model.predict(
            source=frame,
            conf=0.20,
            iou=0.55,
            device="cpu",
            verbose=False,
        )
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            detections_found += len(boxes)
            tracks = tracker.update(boxes.cpu().numpy(), frame) if len(boxes) > 0 else []
            track_ids = {int(t[7]): int(t[4]) for t in tracks} if len(tracks) > 0 else {}
            for i, box in enumerate(boxes):
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                label = model.names.get(int(box.cls[0].item()))
                conf = float(box.conf[0].item())
                track_id = track_ids.get(i)
                main.draw_detection(frame, xyxy, label, conf, track_id)

    cap.release()
    print(f"  -> Successfully processed {frames_processed} frames from sample video (total detection instances: {detections_found})")
    assert frames_processed > 0, "No frames processed"
    print("  -> Video file processing test passed!")


def test_camera_and_error_handling():
    print("[TEST 5/5] Testing Camera Access & Error Handling...")
    # 1. Invalid file error handling
    cap_invalid, _ = main.open_video_source("non_existent_file.mp4")
    assert cap_invalid is None, "Should fail on non-existent file"

    # 2. Camera probe test
    cap_cam, is_file = main.open_video_source("0")
    if cap_cam is not None:
        ret, frame = cap_cam.read()
        print(f"  -> Webcam index 0 opened successfully: read frame={ret}, shape={frame.shape if ret else None}")
        cap_cam.release()
    else:
        print("  -> Webcam index 0 could not be opened directly in this sandbox; error handled gracefully.")

    print("  -> Camera and error handling test passed!")


def run_all_tests():
    print("=" * 60)
    print("  Running HoloMotion Object Detection & Tracking Tests")
    print("=" * 60)

    test_imports()
    model = test_standard_model()
    test_yolo_world_custom_classes()
    test_video_file_processing(model)
    test_camera_and_error_handling()

    print("=" * 60)
    print("  ALL 5 TESTS PASSED SUCCESSFULLY! PROJECT VERIFIED.")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
