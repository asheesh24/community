"""
Main application entry point for the Real-Time Object Detection Platform.
Supports running both the Streamlit web dashboard and a native OpenCV HighGUI window.
"""

import argparse
import os
import sys
import time
import cv2

# Set stdout encoding for Windows console safety
if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from config import db_config, detection_config
from database import DatabaseManager
from detector import YOLODetector


def run_opencv_mode(camera_index: int = 0, conf_thresh: float = 0.50, enable_logging: bool = True):
    """Run pure OpenCV HighGUI video stream window."""
    print("==================================================")
    print("[INFO] Starting Real-Time Object Detection (OpenCV Mode)")
    print("==================================================")
    print(f"* Camera Index: {camera_index}")
    print(f"* Confidence Threshold: {conf_thresh}")
    print(f"* Real-Time Logging: {'Enabled' if enable_logging else 'Disabled'}")

    db = DatabaseManager(db_config)
    print(f"* Database Status: {db.backend.upper()} (Connected: {db.is_connected()})")
    if db.last_error:
        print(f"  Note: {db.last_error}")

    detector = YOLODetector(detection_config.model_name)
    print("* YOLO Model Loaded successfully.")
    print("* Press 'q' or 'ESC' on the OpenCV window to exit.\n")

    if sys.platform.startswith("win"):
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(camera_index)
    else:
        cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera with index {camera_index}.")
        print("[TIP] Try with an uploaded video or synthetic stream via Streamlit dashboard.")
        return

    last_logged = {}
    cooldown = 1.0  # seconds between repeated logs for the same class

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Camera frame could not be read. Exiting stream.")
                break

            annotated_frame, detections = detector.detect_frame(
                frame=frame,
                confidence_threshold=conf_thresh
            )

            # Log detections if enabled
            if enable_logging and detections and db.is_connected():
                now = time.time()
                to_log = []
                for det in detections:
                    c_name = det["object_class"]
                    if det["confidence"] >= 0.70:  # standard assignment threshold
                        if (now - last_logged.get(c_name, 0)) >= cooldown:
                            to_log.append(det)
                            last_logged[c_name] = now

                if to_log:
                    count = db.log_detections_batch(to_log)
                    if count > 0:
                        for d in to_log:
                            print(f"[DB LOG - {db.backend}]: {d['object_class']} ({d['confidence']:.2f})")

            # Display with OpenCV
            cv2.imshow("Real-Time Object Detection Platform (Press 'q' to Quit)", annotated_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n[INFO] Video stream stopped.")


def run_streamlit_dashboard():
    """Launch the interactive Streamlit dashboard."""
    detector = YOLODetector(detection_config.model_name)
    db = DatabaseManager(db_config)
    from ui import render_dashboard
    render_dashboard(detector, db)


def launch_streamlit_subprocess():
    """Spawn the Streamlit dashboard via CLI."""
    import subprocess
    cmd = [sys.executable, "-m", "streamlit", "run", __file__]
    print(f"[INFO] Launching Streamlit Dashboard: {' '.join(cmd)}")
    subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser(description="Real-Time Object Detection Platform")
    parser.add_argument("--mode", choices=["ui", "opencv", "test-db"], default="ui",
                        help="Operating mode: 'ui' (Streamlit dashboard), 'opencv' (pure OpenCV window), 'test-db' (test database connection)")
    parser.add_argument("--camera", type=int, default=0, help="Camera index for OpenCV mode")
    parser.add_argument("--conf", type=float, default=0.50, help="Confidence threshold")
    parser.add_argument("--no-log", action="store_true", help="Disable database logging")

    # Detect if invoked by streamlit runner directly
    is_streamlit = False
    try:
        import streamlit as st
        if hasattr(st, "runtime") and st.runtime.exists():
            is_streamlit = True
    except Exception:
        pass

    if is_streamlit or ("streamlit" in sys.modules and any("run" in arg for arg in sys.argv)):
        run_streamlit_dashboard()
        return

    args, unknown = parser.parse_known_args()

    if args.mode == "test-db":
        print("[INFO] Testing Database Configuration...")
        db = DatabaseManager(db_config)
        status = db.get_status()
        print(f"Backend: {status['backend']}")
        print(f"Connected: {status['connected']}")
        print(f"Host: {status['host']}")
        print(f"Database: {status['database']}")
        if status['last_error']:
            print(f"Details: {status['last_error']}")
        # Test inserting sample log
        if status['connected']:
            test_success = db.log_detection("person", 0.95, (100, 150, 80, 180))
            print(f"Test Log Insertion: {'SUCCESS' if test_success else 'FAILED'}")
            print("Recent Logs:\n", db.fetch_recent_logs(limit=5))

    elif args.mode == "opencv":
        run_opencv_mode(
            camera_index=args.camera,
            conf_thresh=args.conf,
            enable_logging=not args.no_log
        )
    else:
        # Launch streamlit
        launch_streamlit_subprocess()


if __name__ == "__main__":
    main()
