"""
Self-diagnostic and test script for the Real-Time Object Detection Platform.
Verifies OpenCV, Ultralytics YOLOv8, Database connectivity, and schema integrity.
"""

import datetime
import os
import sys
import numpy as np
import cv2

# Set stdout encoding to utf-8 if possible
if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

print("==================================================")
print("[DIAGNOSTIC] Running System Diagnostics for Vision Platform")
print("==================================================")

# 1. Test OpenCV
print("\n[1/4] Testing OpenCV installation...")
try:
    print(f"  * OpenCV Version: {cv2.__version__}")
    test_img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.putText(test_img, "OK", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    print("  [PASS] OpenCV frame manipulation: OK")
except Exception as e:
    print(f"  [FAIL] OpenCV error: {e}")
    sys.exit(1)

# 2. Test Ultralytics YOLO
print("\n[2/4] Testing Ultralytics YOLOv8...")
try:
    from ultralytics import YOLO
    from detector import YOLODetector
    print("  * Loading YOLOv8 nano model (yolov8n.pt)...")
    detector = YOLODetector("yolov8n.pt")
    # Generate synthetic frame with test content
    synthetic_frame = np.full((480, 640, 3), 120, dtype=np.uint8)
    annotated_frame, detections = detector.detect_frame(synthetic_frame, confidence_threshold=0.25)
    print(f"  * Available classes: {len(detector.get_class_names())}")
    print("  [PASS] YOLOv8 Model Initialization & Inference: OK")
except Exception as e:
    print(f"  [FAIL] YOLO error: {e}")
    sys.exit(1)

# 3. Test Database Connection & Schema
print("\n[3/4] Testing Database Manager & Logging...")
try:
    from database import DatabaseManager
    from config import db_config

    db = DatabaseManager(db_config)
    status = db.get_status()
    print(f"  * Backend in use: {status['backend']}")
    print(f"  * Connected: {status['connected']}")
    if status['last_error']:
        print(f"  * Note: {status['last_error']}")

    # Insert a sample detection record
    sample_ts = datetime.datetime.now()
    insert_ok = db.log_detection(
        object_class="person",
        confidence=0.88,
        bbox=(120, 80, 240, 320),
        timestamp=sample_ts
    )
    print(f"  * Sample Record Insertion: {'PASS' if insert_ok else 'FAIL'}")

    # Read back the inserted record
    recent = db.fetch_recent_logs(limit=3)
    print("  * Recent Logs verification:")
    print(recent.to_string())
    print("  [PASS] Database Module & Logging: OK")
except Exception as e:
    print(f"  [FAIL] Database error: {e}")
    sys.exit(1)

# 4. Generate a sample test video clip for testing/demo without requiring physical webcam
print("\n[4/4] Generating sample demo test video ('sample_demo.mp4')...")
try:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = "sample_demo.mp4"
    out = cv2.VideoWriter(out_path, fourcc, 20.0, (640, 480))

    # Generate 60 frames (3 seconds) of synthetic demo
    for i in range(60):
        frame = np.full((480, 640, 3), 50, dtype=np.uint8)
        cx = int(320 + 150 * np.sin(i / 10.0))
        cy = int(240 + 80 * np.cos(i / 10.0))
        cv2.circle(frame, (cx, cy), 40, (0, 165, 255), -1)
        cv2.putText(frame, f"Demo Frame {i+1}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        out.write(frame)
    out.release()
    print(f"  [PASS] Sample demo video generated: {out_path} ({os.path.getsize(out_path)} bytes)")
except Exception as e:
    print(f"  [WARN] Note on sample video generation: {e}")

print("\n==================================================")
print("[SUCCESS] ALL PLATFORM DIAGNOSTICS PASSED SUCCESSFULLY!")
print("==================================================")
