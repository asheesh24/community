# 🎯 Real-Time Object Detection Platform with YOLO, OpenCV & MySQL Event Logging

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5.0.0-5C3EE8.svg)](https://opencv.org/)
[![YOLOv8](https://img.shields.io/badge/Ultralytics-YOLOv8-00FFFF.svg)](https://ultralytics.com/)
[![MySQL](https://img.shields.io/badge/MySQL-Connector-4479A1.svg)](https://www.mysql.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-grade, modular computer vision platform that performs real-time object detection on live video feeds using **Ultralytics YOLOv8** and **OpenCV**, automatically streams detected event metadata to a **MySQL** database, and provides an interactive **Streamlit** dashboard for live visual feeds, confidence adjustments, class filtering, and historical logs inspection.

---

## 📌 Project Architecture

The codebase strictly adheres to a modular design pattern with distinct separation of concerns:

```
real-time-object-detection/
├── main.py                     # Primary entry point (UI runner, OpenCV mode, DB diagnostics)
├── detector.py                 # Ultralytics YOLOv8 inference wrapper & OpenCV bounding box annotator
├── database.py                 # MySQL database manager with connection pooling and resilient fallback
├── ui.py                       # Modern Streamlit interactive dashboard
├── config.py                   # Centralized configuration, COCO class presets, and environment loaders
├── database_schema.sql         # SQL schema definitions for vision_platform database
├── test_system.py              # Automated self-diagnostic test suite
├── requirements.txt            # Python package dependencies
├── .env.example                # Template configuration file for database credentials
├── .env                        # Local environment variables (safely ignored by git)
├── .gitignore                  # Git safeguard preventing accidental credential or weight commits
└── README.md                   # Comprehensive project documentation
```

---

## 🚀 Key Features

- **⚡ Real-Time YOLOv8 Inference:** High-throughput object detection powered by Ultralytics YOLOv8 (supporting YOLOv8n nano for ultra-low latency or any standard YOLO model).
- **🎨 OpenCV Frame Annotation:** Dynamic, color-coded bounding boxes, class labels, confidence scores, and real-time FPS overlay.
- **🎛️ Dynamic Filtering & Thresholds:**
  - Adjustable detection confidence threshold slider.
  - Class category filters (People, Vehicles, Animals, Electronics, Food, or all 80 COCO classes).
  - Fine-grained object selection (e.g., detect only `"person"` or `"cell phone"`).
- **💾 Automated MySQL Event Logging:**
  - Persists detection events with timestamp, class name, confidence, and exact bounding box coordinates `(bbox_x, bbox_y, bbox_w, bbox_h)`.
  - Configurable minimum confidence threshold for database logging (e.g., log only detections $\ge 70\%$).
  - Per-class logging cooldown to prevent redundant insertions for stationary objects.
  - Resilient local fallback mode ensures uninterrupted operation even when the database server is offline.
- **📊 Interactive Streamlit Dashboard:**
  - Real-time video canvas with start/stop/pause stream controls.
  - Live metric KPI cards (Total Logged Events, Unique Classes, Top Detected Object, Inference FPS).
  - Dynamic class distribution bar chart.
  - Live tabular display of the most recent database records with search/filter and log-clearing capability.
- **🔒 Security & Best Practices:** Complete decoupling of database credentials via `.env` and `.gitignore`.

---

## 🗄️ Database Schema

The platform logs events into the `vision_platform` database under the `detection_logs` table:

```sql
CREATE DATABASE IF NOT EXISTS vision_platform;
USE vision_platform;

CREATE TABLE IF NOT EXISTS detection_logs (
    log_id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    object_class VARCHAR(50) NOT NULL,
    confidence FLOAT NOT NULL,
    bbox_x INT NOT NULL,
    bbox_y INT NOT NULL,
    bbox_w INT NOT NULL,
    bbox_h INT NOT NULL,
    INDEX idx_timestamp (timestamp),
    INDEX idx_object_class (object_class)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/Real-Time-Object-Detection.git
cd Real-Time-Object-Detection
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env` and configure your local MySQL credentials:
```bash
# Windows
copy .env.example .env

# Linux / macOS
cp .env.example .env
```

Edit `.env`:
```ini
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=your_mysql_password
DB_NAME=vision_platform
DB_FALLBACK_SQLITE=True
YOLO_MODEL=yolov8n.pt
DEFAULT_CONFIDENCE_THRESHOLD=0.50
DEFAULT_CAMERA_INDEX=0
```

### 5. Setup MySQL Database
Using **MySQL Workbench**, **XAMPP phpMyAdmin**, or the MySQL CLI:
```bash
mysql -u root -p < database_schema.sql
```
*(Note: If MySQL is not running or credentials are not yet set, the platform will automatically activate the local fallback without crashing).*

---

## 💻 Running the Application

### Mode 1: Interactive Streamlit Dashboard (Recommended)
Launch the web interface:
```bash
streamlit run main.py
```
Open your browser at `http://localhost:8501`.

### Mode 2: Native OpenCV Window Mode
Run a pure OpenCV stream (ideal for testing direct webcam feeds):
```bash
python main.py --mode opencv --camera 0 --conf 0.50
```
- Press `q` or `ESC` on the OpenCV window to exit.

### Mode 3: Self-Diagnostic & Database Verification
Run the integrated diagnostic suite:
```bash
python test_system.py
```
or
```bash
python main.py --mode test-db
```

---

## 🎥 Demonstration Video

A working demonstration of the platform running with real-time bounding boxes and simultaneous MySQL database logging updates is available here:
- **Video Demonstration:** [YouTube / Google Drive Demo Link](#)

---

## 📝 Learning Experience & Medium Article

Detailed engineering walkthrough discussing computer vision fundamentals, YOLO architecture, database schema design, and challenges faced during integration:
- **Medium Article:** [Building a Real-Time Object Detection Platform with YOLO, OpenCV, and MySQL](#)

---

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
