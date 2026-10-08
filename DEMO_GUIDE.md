# 🎥 1-2 Minute Video Demonstration Recording Guide

This guide provides a step-by-step walkthrough to record the 1-2 minute demonstration video required for **Assignment 5**.

---

## 🎯 Demonstration Objective
Show the instructor:
1. The platform starting up cleanly.
2. The Streamlit dashboard / OpenCV feed running in real-time with bounding boxes and FPS.
3. Object detection in action (e.g., detecting you as `"person"`, your phone as `"cell phone"`, a cup/bottle, etc.).
4. The database table updating live with timestamp, confidence, and bounding box coordinates.
5. Showing MySQL Workbench, phpMyAdmin, or the terminal query verifying records in `vision_platform.detection_logs`.

---

## 🛠️ Recommended Tools for Recording
- **Windows Game Bar:** Press `Win + Alt + R` to record the screen instantly.
- **OBS Studio:** Excellent for capturing side-by-side windows (Streamlit on left, MySQL Workbench/phpMyAdmin on right).
- **Loom / Clipchamp:** Quick and easy screen recording.

---

## 🎬 1-2 Minute Video Script & Timeline

| Time | Scene | What to Say & Do |
| :--- | :--- | :--- |
| **0:00 - 0:15** | Terminal & Launch | • Open terminal in project directory.<br>• Run: `streamlit run main.py`<br>• Briefly mention: *"Hello! This is my Real-Time Object Detection Platform built using Python, OpenCV, YOLOv8, and MySQL event logging."* |
| **0:15 - 0:40** | Dashboard Tour & Detection | • Open the browser at `http://localhost:8501`.<br>• Show the Status Badge (`🟢 MySQL Connected` or `🟡 Active Database`).<br>• Check `▶️ Start Live Stream`.<br>• Wave or hold up a cell phone or water bottle.<br>• Point out the bounding boxes, confidence badges, and real-time FPS counter. |
| **0:40 - 1:05** | Filter & Threshold Adjustments | • Demonstrate the sidebar controls:<br>  - Adjust Confidence slider (e.g., from 0.50 to 0.75).<br>  - Select a category filter (e.g., "People & Personal Items").<br>  - Show the live class distribution chart updating dynamically. |
| **1:05 - 1:35** | MySQL Real-Time Logging Verification | • Scroll down to the **📋 Recent Database Logs** section.<br>• Show new rows appearing with `log_id`, `timestamp`, `object_class`, `confidence`, and `bbox_x, bbox_y, bbox_w, bbox_h`.<br>• (Optional / Recommended) Switch to **MySQL Workbench** or run `python main.py --mode test-db` or `SELECT * FROM detection_logs ORDER BY log_id DESC LIMIT 10;` in MySQL CLI to show the actual database table rows! |
| **1:35 - 1:50** | Summary & Conclusion | • Wrap up: *"The platform runs modularly, decouples credentials safely via .env, and performs real-time persistence without dropping frame rate. Thank you!"* |

---

## 📤 Upload Instructions
1. Save the MP4 recording.
2. Upload to **YouTube (Set visibility to 'Unlisted')** OR upload to **Google Drive (Set share permissions to 'Anyone with the link can view')**.
3. Copy the link and paste it into the assignment submission form.
