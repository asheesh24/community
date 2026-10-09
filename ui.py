"""
Streamlit User Interface for the Real-Time Object Detection Platform.
Provides an interactive dashboard for live video feeds, model adjustments,
and database logging inspection.
"""

import os
import sys
import time
from typing import Optional
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

from config import CLASS_CATEGORIES, COCO_CLASSES, detection_config
from database import DatabaseManager
from detector import YOLODetector


def render_dashboard(detector: YOLODetector, db_manager: DatabaseManager):
    """Render the full Streamlit dashboard."""
    st.set_page_config(
        page_title="Real-Time Object Detection Platform",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom styling for clean, professional look
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.2rem;
            font-weight: 700;
            color: #1E293B;
            margin-bottom: 0.2rem;
        }
        .sub-header {
            font-size: 1.05rem;
            color: #64748B;
            margin-bottom: 1.5rem;
        }
        .metric-card {
            background: #F8FAFC;
            border-radius: 8px;
            padding: 12px;
            border: 1px solid #E2E8F0;
        }
        .status-badge {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 9999px;
            font-size: 0.85rem;
            font-weight: 600;
        }
        .badge-mysql {
            background-color: #DEF7EC;
            color: #03543F;
        }
        .badge-sqlite {
            background-color: #FEF08A;
            color: #713F12;
        }
        .badge-disconnected {
            background-color: #FDE8E8;
            color: #9B1C1C;
        }
        </style>
    """, unsafe_allow_html=True)

    # Top Title & Subtitle
    st.markdown('<div class="main-header">🎯 Real-Time Object Detection Platform</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Powered by <b>OpenCV</b>, <b>Ultralytics YOLOv8</b>, and <b>MySQL Event Logging</b></div>',
        unsafe_allow_html=True
    )

    # Database Status Banner
    db_status = db_manager.get_status()
    if db_status["backend"] == "mysql":
        badge_html = f'<span class="status-badge badge-mysql">🟢 MySQL Connected ({db_status["host"]}:{db_status["database"]})</span>'
    elif db_status["backend"] == "sqlite":
        badge_html = f'<span class="status-badge badge-sqlite">🟡 Local SQLite Fallback Active ({db_status["database"]})</span>'
    else:
        badge_html = '<span class="status-badge badge-disconnected">🔴 Database Disconnected</span>'

    st.markdown(badge_html, unsafe_allow_html=True)
    st.write("")

    # Sidebar Controls
    st.sidebar.header("⚙️ Platform Settings")

    # Source Selection
    source_type = st.sidebar.selectbox(
        "📹 Video Source",
        ["Demo Sample Video (Included)", "Webcam", "Upload Video File", "Generate Demo Synthetic Stream"],
        index=0
    )

    camera_idx = 0
    uploaded_file = None
    if source_type == "Webcam":
        camera_idx = st.sidebar.number_input("Camera Index", min_value=0, max_value=5, value=detection_config.default_camera_index, step=1)
    elif source_type == "Upload Video File":
        uploaded_file = st.sidebar.file_uploader("Upload MP4 / AVI / MOV", type=["mp4", "avi", "mov", "mkv"])

    # Detection Filters
    st.sidebar.subheader("🔍 Detection Configuration")
    conf_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.10,
        max_value=1.00,
        value=detection_config.default_confidence,
        step=0.05
    )

    category_choice = st.sidebar.selectbox("Class Category Preset", list(CLASS_CATEGORIES.keys()))
    default_classes = CLASS_CATEGORIES[category_choice]
    selected_classes = st.sidebar.multiselect(
        "Filter Specific Objects",
        options=COCO_CLASSES,
        default=default_classes
    )

    # Database Logging Options
    st.sidebar.subheader("💾 Database Event Logging")
    enable_logging = st.sidebar.checkbox("Enable Real-Time Logging", value=True)
    log_min_conf = st.sidebar.slider(
        "Min Confidence to Log",
        min_value=0.50,
        max_value=1.00,
        value=0.70,
        step=0.05,
        help="Only detections with confidence >= this threshold are saved to the database."
    )
    log_cooldown = st.sidebar.slider(
        "Logging Cooldown per Class (seconds)",
        min_value=0.0,
        max_value=5.0,
        value=1.0,
        step=0.5,
        help="Prevents inserting hundreds of duplicate rows when an object remains in the frame."
    )

    if st.sidebar.button("📥 Seed Sample Detection Events"):
        seeded_count = db_manager.seed_sample_logs(count=15)
        st.sidebar.success(f"Added {seeded_count} sample detection events!")
        st.rerun()

    if st.sidebar.button("🔄 Reconnect Database"):
        new_status = db_manager.reconnect()
        st.sidebar.success(f"Backend: {new_status['backend']}")

    if st.sidebar.button("🗑️ Clear Database Logs"):
        if db_manager.clear_logs():
            st.sidebar.success("Database logs wiped successfully.")
            st.rerun()
        else:
            st.sidebar.error("Failed to clear logs.")

    # Metric Row
    metrics = db_manager.fetch_metrics()
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("Total Detections Logged", metrics["total_detections"])
    m_col2.metric("Unique Classes", metrics["unique_classes"])
    m_col3.metric("Top Detected Object", metrics["top_object"])
    fps_placeholder = m_col4.empty()
    fps_placeholder.metric("Inference FPS", f"{detector.fps:.1f}")

    # Main Visual Feed & Live Analytics
    col_video, col_charts = st.columns([2, 1])

    with col_video:
        st.subheader("🎥 Live Feed")
        feed_placeholder = st.empty()
        run_stream = st.checkbox("▶️ Start Live Stream", value=False)

    with col_charts:
        st.subheader("📊 Detected Class Distribution")
        chart_placeholder = st.empty()
        current_detections_box = st.empty()

    # Bottom Logs Section
    st.markdown("---")
    st.subheader("📋 Recent Database Logs")
    table_placeholder = st.empty()

    # Track logging cooldowns: {class_name: last_logged_timestamp}
    last_logged_time = {}

    def update_logs_table():
        df_logs = db_manager.fetch_recent_logs(limit=25)
        if not df_logs.empty:
            table_placeholder.dataframe(
                df_logs,
                use_container_width=True,
                hide_index=True
            )
        else:
            table_placeholder.info("No detection events recorded yet in the database.")

    def update_charts():
        curr_metrics = db_manager.fetch_metrics()
        dist_df = curr_metrics["class_distribution"]
        if not dist_df.empty:
            chart_placeholder.bar_chart(dist_df.set_index("object_class"))
        else:
            chart_placeholder.caption("No class distribution data available.")

    # Initial table and chart draw
    update_logs_table()
    update_charts()

    # Stream Execution
    if run_stream:
        cap = None
        temp_video_path = None

        if source_type == "Demo Sample Video (Included)":
            demo_path = "sample_demo.mp4"
            if os.path.exists(demo_path):
                cap = cv2.VideoCapture(demo_path)
            else:
                st.error("Demo video 'sample_demo.mp4' not found.")
                return
        elif source_type == "Webcam":
            if sys.platform.startswith("win"):
                cap = cv2.VideoCapture(int(camera_idx), cv2.CAP_DSHOW)
                if not cap.isOpened():
                    cap = cv2.VideoCapture(int(camera_idx))
            else:
                cap = cv2.VideoCapture(int(camera_idx))
        elif source_type == "Upload Video File" and uploaded_file is not None:
            import tempfile
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            temp_video_path = tfile.name
            cap = cv2.VideoCapture(temp_video_path)
        elif source_type == "Generate Demo Synthetic Stream":
            cap = None  # Synthetic handler below

        if source_type in ("Demo Sample Video (Included)", "Webcam", "Upload Video File"):
            if cap is None or not cap.isOpened():
                st.error(f"Could not open video source '{source_type}'. If using Webcam, check your camera permissions or index.")
                return

        frame_counter = 0

        try:
            while run_stream:
                if source_type == "Generate Demo Synthetic Stream":
                    # Generate an animated frame with realistic scene for YOLO detection
                    scene_path = "sample_scene.jpg"
                    if os.path.exists(scene_path):
                        scene_img = cv2.imread(scene_path)
                    else:
                        scene_img = None

                    if scene_img is not None:
                        t_val = time.time()
                        sx = int(15 * np.sin(t_val * 1.5))
                        sy = int(10 * np.cos(t_val * 1.5))
                        M = np.float32([[1, 0, sx], [0, 1, sy]])
                        frame = cv2.warpAffine(scene_img, M, (scene_img.shape[1], scene_img.shape[0]))
                        frame = cv2.resize(frame, (640, 480))
                    else:
                        frame = np.full((480, 640, 3), 40, dtype=np.uint8)
                        t_val = time.time()
                        px = int(200 + 100 * np.sin(t_val * 1.5))
                        py = int(180 + 40 * np.cos(t_val * 1.5))
                        cv2.rectangle(frame, (px, py), (px + 120, py + 220), (200, 200, 200), -1)
                        cv2.putText(frame, "Demo Stream", (20, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                else:
                    ret, frame = cap.read()
                    if not ret:
                        # Video finished or webcam disconnect
                        if source_type in ("Upload Video File", "Demo Sample Video (Included)"):
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop video
                            continue
                        else:
                            st.warning("Video stream ended or disconnected.")
                            break

                # Run YOLO detection & visual drawing
                annotated_frame, detections = detector.detect_frame(
                    frame=frame,
                    confidence_threshold=conf_threshold,
                    target_classes=selected_classes
                )

                # Event Logging to MySQL / Database
                if enable_logging and detections:
                    now = time.time()
                    to_log = []
                    for det in detections:
                        c_name = det["object_class"]
                        conf = det["confidence"]

                        if conf >= log_min_conf:
                            last_t = last_logged_time.get(c_name, 0)
                            if (now - last_t) >= log_cooldown:
                                to_log.append(det)
                                last_logged_time[c_name] = now

                    if to_log:
                        db_manager.log_detections_batch(to_log)

                # Convert annotated BGR frame to RGB for Streamlit display
                rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                feed_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

                # Update live stats every 10 frames
                frame_counter += 1
                if frame_counter % 10 == 0:
                    fps_placeholder.metric("Inference FPS", f"{detector.fps:.1f}")
                    update_logs_table()
                    update_charts()

                    # Show current frame detections
                    if detections:
                        current_detections_box.markdown(
                            "**Current Detections:** " + ", ".join([f"`{d['object_class']}` ({d['confidence']:.2f})" for d in detections])
                        )
                    else:
                        current_detections_box.caption("No target objects detected in current frame.")

                # Small delay to keep UI interactive
                time.sleep(0.01)

        finally:
            if cap is not None:
                cap.release()
