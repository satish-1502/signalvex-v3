"""
detector.py — YOLO Human Detection Module for SIGNALVEX_V2
==========================================================
Drone-Based Human Detection and Rescue Mapping Prototype
Department of Electronics and Communication Engineering
Chaitanya Bharathi Institute of Technology | Sky Seekers Team

Features:
- Loads a pretrained Ultralytics YOLO model (YOLOv8n default, CPU compatible).
- Strictly filters for person class (COCO class 0).
- Supports adjustable confidence thresholds and configurable inference resolution (imgsz).
- Returns pixel coordinates in original frame space [x1, y1, x2, y2].
- Fully CPU-optimized for Windows laptops.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np


class HumanDetector:
    """
    YOLO-based human detector filtered strictly for person detections (COCO class 0).
    Coordinates are always returned in the original frame's pixel coordinate system.
    """

    PERSON_CLASS_ID = 0

    def __init__(
        self,
        model_path: str = "models/yolov8n.pt",
        conf_threshold: float = 0.35,
        imgsz: int = 640
    ):
        """
        Initialize detector.

        Args:
            model_path: Path to YOLO weights (.pt). Default: models/yolov8n.pt
            conf_threshold: Minimum confidence score to accept detection. Default: 0.35
            imgsz: Inference image size (640, 960, 1280). Default: 640
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        self.model = None
        self._load_model()

    def _load_model(self):
        """Loads the Ultralytics YOLO model with robust path and fallback handling."""
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "Ultralytics package is not installed. Run 'pip install ultralytics'."
            ) from e

        # Check given model path
        model_file = Path(self.model_path)
        if model_file.exists():
            self.model = YOLO(str(model_file))
            return

        # Check local models directory
        alt_path = Path(__file__).parent / "models" / "yolov8n.pt"
        if alt_path.exists():
            self.model_path = str(alt_path)
            self.model = YOLO(self.model_path)
            return

        # Fallback: attempt to load by name (Ultralytics will download automatically if connected)
        try:
            print(f"[SIGNALVEX_V2 Detector] Loading YOLO model via name: yolov8n.pt...")
            self.model = YOLO("yolov8n.pt")
            self.model_path = "yolov8n.pt"
        except Exception as err:
            raise RuntimeError(
                f"Failed to load YOLO model: {err}. Please ensure 'yolov8n.pt' is available in models/ folder."
            ) from err

    def detect(
        self,
        frame: np.ndarray,
        conf_threshold: Optional[float] = None,
        imgsz: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Run person detection on a single BGR frame.

        Args:
            frame: OpenCV BGR image numpy array.
            conf_threshold: Optional override for confidence threshold.
            imgsz: Optional override for inference image size.

        Returns:
            List of detection dicts:
            [
                {
                    "bbox": [x1, y1, x2, y2],
                    "confidence": 0.864,
                    "confidence_percent": 86.4,
                    "class_id": 0,
                    "class_name": "person",
                    "bbox_width": w,
                    "bbox_height": h
                },
                ...
            ]
        """
        if self.model is None:
            raise RuntimeError("YOLO model is not initialized.")

        conf = conf_threshold if conf_threshold is not None else self.conf_threshold
        size = imgsz if imgsz is not None else self.imgsz

        # Perform inference strictly on person class (class 0)
        results = self.model(
            frame,
            classes=[self.PERSON_CLASS_ID],
            conf=conf,
            imgsz=size,
            verbose=False
        )

        detections = []
        if not results or len(results) == 0:
            return detections

        first_res = results[0]
        boxes = first_res.boxes
        if boxes is None or len(boxes) == 0:
            return detections

        orig_h, orig_w = frame.shape[:2]

        for box in boxes:
            cls_id = int(box.cls[0].item())
            score = float(box.conf[0].item())

            if cls_id == self.PERSON_CLASS_ID and score >= conf:
                # Coordinates in original frame resolution [x1, y1, x2, y2]
                xyxy = box.xyxy[0].tolist()
                x1 = max(0, min(orig_w - 1, int(round(xyxy[0]))))
                y1 = max(0, min(orig_h - 1, int(round(xyxy[1]))))
                x2 = max(0, min(orig_w - 1, int(round(xyxy[2]))))
                y2 = max(0, min(orig_h - 1, int(round(xyxy[3]))))

                bbox_w = max(1, x2 - x1)
                bbox_h = max(1, y2 - y1)

                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": round(score, 4),
                    "confidence_percent": round(score * 100.0, 1),
                    "class_id": cls_id,
                    "class_name": "person",
                    "bbox_width": bbox_w,
                    "bbox_height": bbox_h
                })

        return detections
