"""
Object detection module utilizing Ultralytics YOLOv8 and OpenCV.
Extracts bounding boxes, labels, and confidence metrics in real time.
"""

import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
from ultralytics import YOLO

from config import COCO_CLASSES, detection_config


class YOLODetector:
    """Wrapper around Ultralytics YOLOv8 model for real-time inference and frame annotation."""

    # Curated color palette (BGR format for OpenCV) for clear and distinct class visualization
    PALETTE = [
        (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255),
        (49, 210, 207), (10, 249, 72), (23, 204, 146), (134, 219, 61),
        (52, 147, 26), (187, 212, 0), (168, 153, 44), (255, 194, 0),
        (255, 143, 0), (255, 96, 0), (255, 38, 0), (228, 0, 102),
        (205, 0, 180), (145, 0, 230), (84, 0, 255), (20, 0, 255)
    ]

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or detection_config.model_name
        self.model = YOLO(self.model_name)
        self.class_names = self.model.names  # Mapping {0: 'person', 1: 'bicycle', ...}
        self.fps: float = 0.0
        self._prev_frame_time: float = time.time()

    def get_class_names(self) -> List[str]:
        """Return the list of all detectable class names."""
        if isinstance(self.class_names, dict):
            return list(self.class_names.values())
        return COCO_CLASSES

    def _get_color_for_class(self, class_id: int) -> Tuple[int, int, int]:
        """Generate or pick a consistent color for a given class ID."""
        return self.PALETTE[class_id % len(self.PALETTE)]

    def detect_frame(
        self,
        frame: np.ndarray,
        confidence_threshold: float = 0.50,
        target_classes: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Run inference on a single frame, draw annotations, and extract detection records.

        Args:
            frame: Raw BGR image as a numpy array.
            confidence_threshold: Minimum confidence score [0.0 - 1.0].
            target_classes: Optional subset of class names to detect/annotate.

        Returns:
            Tuple of (annotated_frame, list_of_detections).
            Each detection contains:
                'object_class': str,
                'confidence': float,
                'bbox': (bbox_x, bbox_y, bbox_w, bbox_h)
        """
        curr_time = time.time()
        time_diff = curr_time - self._prev_frame_time
        if time_diff > 0:
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / time_diff) if self.fps > 0 else (1.0 / time_diff)
        self._prev_frame_time = curr_time

        # Run YOLO inference
        results = self.model.predict(
            source=frame,
            conf=confidence_threshold,
            verbose=False,
            device="cpu"  # Safe default across all environments
        )[0]

        annotated_frame = frame.copy()
        detections: List[Dict[str, Any]] = []

        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
            confs = results.boxes.conf.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy().astype(int)

            for box, conf, cls_id in zip(boxes, confs, classes):
                class_name = self.class_names.get(cls_id, f"class_{cls_id}")

                # Filter by selected target classes if specified
                if target_classes and class_name not in target_classes:
                    continue

                x1, y1, x2, y2 = [int(v) for v in box]
                # Convert to (bbox_x, bbox_y, bbox_w, bbox_h) as required by schema
                bbox_x = max(0, x1)
                bbox_y = max(0, y1)
                bbox_w = max(0, x2 - x1)
                bbox_h = max(0, y2 - y1)

                detections.append({
                    "object_class": class_name,
                    "confidence": round(float(conf), 4),
                    "bbox": (bbox_x, bbox_y, bbox_w, bbox_h)
                })

                # Visual annotation
                color = self._get_color_for_class(cls_id)
                # Draw bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

                # Draw label badge
                label = f"{class_name} {conf:.2f}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.55
                thickness = 1
                (label_w, label_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

                # Ensure badge stays inside frame
                badge_y1 = max(0, y1 - label_h - 8)
                badge_y2 = y1
                badge_x2 = min(frame.shape[1], x1 + label_w + 10)

                cv2.rectangle(
                    annotated_frame,
                    (x1, badge_y1),
                    (badge_x2, badge_y2),
                    color,
                    cv2.FILLED
                )
                cv2.putText(
                    annotated_frame,
                    label,
                    (x1 + 5, y1 - 4),
                    font,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                    cv2.LINE_AA
                )

        # Overlay FPS and Detection Counter on the top-left
        fps_text = f"FPS: {self.fps:.1f} | Detections: {len(detections)}"
        cv2.putText(
            annotated_frame,
            fps_text,
            (16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        return annotated_frame, detections
