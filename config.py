"""
Configuration module for the Real-Time Object Detection Platform.
Loads environment variables securely and exposes system configurations.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent

# Load environment variables from .env file
load_dotenv(dotenv_path=BASE_DIR / ".env")


@dataclass(frozen=True)
class DatabaseConfig:
    host: str = os.getenv("DB_HOST", "localhost")
    port: int = int(os.getenv("DB_PORT", "3306"))
    user: str = os.getenv("DB_USER", "root")
    password: str = os.getenv("DB_PASSWORD", "")
    database: str = os.getenv("DB_NAME", "vision_platform")
    fallback_sqlite: bool = os.getenv("DB_FALLBACK_SQLITE", "True").lower() in ("true", "1", "t", "yes")
    sqlite_file: Path = BASE_DIR / "vision_platform_fallback.db"


@dataclass(frozen=True)
class DetectionConfig:
    model_name: str = os.getenv("YOLO_MODEL", "yolov8n.pt")
    default_confidence: float = float(os.getenv("DEFAULT_CONFIDENCE_THRESHOLD", "0.50"))
    default_camera_index: int = int(os.getenv("DEFAULT_CAMERA_INDEX", "0"))


# Standard 80 COCO Classes supported by YOLOv8
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# Quick category groupings for user convenience in UI
CLASS_CATEGORIES = {
    "All Classes": COCO_CLASSES,
    "People & Personal Items": ["person", "backpack", "umbrella", "handbag", "tie", "suitcase", "cell phone", "book"],
    "Vehicles & Transport": ["bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"],
    "Animals": ["bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"],
    "Indoor & Furniture": ["chair", "couch", "bed", "dining table", "tv", "laptop", "mouse", "keyboard", "clock"],
    "Kitchen & Food": ["bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "pizza"]
}

db_config = DatabaseConfig()
detection_config = DetectionConfig()
