"""
video_processor.py — Video Pipeline & Annotation Module for SIGNALVEX_V2
========================================================================
Drone-Based Human Detection and Rescue Mapping Prototype
Department of Electronics and Communication Engineering
Chaitanya Bharathi Institute of Technology | Sky Seekers Team

Features:
- Incremental frame-by-frame processing to handle large AI flood videos without memory bloat.
- Extracts accurate video metadata (dimensions, FPS, frame count, duration).
- Direct drawing on original video frames preserving true pixel coordinates and aspect ratio.
- Configurable box line thickness and label scale for dense / crowded aerial scenes (>35 people).
- Label tags displaying "ID {track_id} | {confidence}%" calculated from actual YOLO outputs.
- High-visibility color palette:
    * Cyan / Emerald for high-confidence tracks
    * Safety Amber / Orange for low-confidence or flagged tracks (< review_threshold)
- Incremental VideoWriter output to MP4 with codec fallback.
- Real-time progress callback integration with Streamlit.
"""

import os
import time
from typing import Dict, Any, Callable, Optional, Tuple, List
import cv2
import numpy as np

from tracker import HumanTracker
from logger import DetectionLogger


class VideoProcessor:
    """
    Manages end-to-end video decoding, YOLO tracking execution, frame annotation,
    detection logging, and annotated MP4 video generation.
    """

    def __init__(self, tracker: HumanTracker, logger: DetectionLogger):
        self.tracker = tracker
        self.logger = logger

    @staticmethod
    def get_video_info(video_path: str) -> Dict[str, Any]:
        """
        Validates video file and extracts metadata.
        Raises ValueError with clear message if video is unreadable or corrupted.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found at: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(
                f"Failed to open '{os.path.basename(video_path)}'. "
                "The file may be corrupted, empty, or encoded in an unsupported video codec."
            )

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps):
            fps = 24.0  # Sensible fallback for AI-generated video with missing metadata

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if width <= 0 or height <= 0:
            cap.release()
            raise ValueError(f"Invalid frame dimensions ({width}x{height}) in '{os.path.basename(video_path)}'.")

        duration_sec = total_frames / fps if fps > 0 and total_frames > 0 else 0.0
        minutes = int(duration_sec // 60)
        seconds = duration_sec % 60
        duration_formatted = f"{minutes:02d}:{seconds:05.2f}"

        cap.release()

        return {
            "file_name": os.path.basename(video_path),
            "width": width,
            "height": height,
            "fps": round(fps, 2),
            "total_frames": max(0, total_frames),
            "duration_sec": round(duration_sec, 2),
            "duration_formatted": duration_formatted
        }

    @staticmethod
    def format_timestamp(seconds: float) -> str:
        """Formats seconds to MM:SS.ss string."""
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins:02d}:{secs:05.2f}"

    @staticmethod
    def annotate_frame(
        frame: np.ndarray,
        tracks: List[Dict[str, Any]],
        frame_idx: int,
        timestamp_sec: float,
        timestamp_formatted: str,
        cumulative_detections: int,
        distinct_tracks_count: int,
        review_threshold: float = 0.50,
        box_thickness: int = 2,
        font_scale: float = 0.50
    ) -> np.ndarray:
        """
        Draws bounding boxes, persistent IDs, actual confidence percentages,
        and SAR telemetry status directly onto original video frames.
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        # 1. Semi-transparent top telemetry banner
        banner_h = max(34, int(h * 0.05))
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), (15, 23, 42), -1)  # Dark slate
        cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)

        # Telemetry text
        banner_text = (
            f"SIGNALVEX SAR | Frame: {frame_idx:04d} [{timestamp_formatted}] | "
            f"Active: {len(tracks)} | Distinct Tracks: {distinct_tracks_count} | Total Obs: {cumulative_detections}"
        )
        banner_font_scale = max(0.40, min(0.65, w / 1600.0))
        cv2.putText(
            annotated,
            banner_text,
            (12, int(banner_h * 0.70)),
            cv2.FONT_HERSHEY_SIMPLEX,
            banner_font_scale,
            (240, 246, 252),  # Crisp white/light blue
            1,
            cv2.LINE_AA
        )

        # 2. Draw person bounding boxes and labels
        for trk in tracks:
            bbox = trk["bbox"]
            track_id = trk["track_id"]
            conf = trk["confidence"]
            warning = trk.get("tracking_warning", "None")
            x1, y1, x2, y2 = bbox

            # Color scheme:
            # - Amber/Orange (BGR: 0, 165, 245) for flagged (low-confidence or warnings)
            # - High-visibility Emerald/Lime (BGR: 0, 230, 115) for confident stable tracks
            auto_flag = trk.get("automatic_flag", (conf < review_threshold or warning != "None"))
            if auto_flag:
                box_color = (0, 165, 245)  # Safety Amber / Orange
                tag_label = f"REV #{track_id} | {conf*100.0:.1f}%"
            else:
                box_color = (0, 230, 115)  # Signal Lime / Green
                tag_label = f"ID {track_id} | {conf*100.0:.1f}%"

            # Draw rectangular bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, box_thickness)

            # Label calculations
            (label_w, label_h), baseline = cv2.getTextSize(
                tag_label,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                1
            )

            # Place label slightly above box, or just inside if top edge touches banner
            tag_y1 = max(banner_h + 2, y1 - label_h - 6)
            tag_y2 = tag_y1 + label_h + 4
            tag_x2 = min(w - 2, x1 + label_w + 8)

            # Background pill for label
            cv2.rectangle(annotated, (x1, tag_y1), (tag_x2, tag_y2), box_color, -1)

            # Dark high-contrast label text
            cv2.putText(
                annotated,
                tag_label,
                (x1 + 4, tag_y2 - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (10, 15, 25),
                1,
                cv2.LINE_AA
            )

        return annotated

    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        conf_threshold: float = 0.35,
        review_threshold: float = 0.50,
        max_frames: Optional[int] = None,
        box_thickness: int = 2,
        font_scale: float = 0.50,
        progress_callback: Optional[Callable[[float, np.ndarray, Dict[str, Any]], None]] = None
    ) -> Dict[str, Any]:
        """
        Executes frame-by-frame detection, persistent tracking, logging, and video saving.

        Returns comprehensive summary dictionary with actual calculated metrics.
        """
        info = self.get_video_info(video_path)
        fps = info["fps"]
        total_frames = info["total_frames"]
        video_name = info["file_name"]

        # Ensure fresh tracking and logging session for this video
        self.tracker.reset()
        self.logger.clear()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        # Setup VideoWriter if output requested
        writer = None
        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            # Try mp4v codec
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (info["width"], info["height"]))
            if not writer.isOpened():
                # Fallback to XVID / avi container compatible fourcc
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                writer = cv2.VideoWriter(output_path, fourcc, fps, (info["width"], info["height"]))

        frames_to_process = min(total_frames, max_frames) if (max_frames and max_frames > 0) else total_frames
        if frames_to_process <= 0:
            frames_to_process = total_frames if total_frames > 0 else 1000

        frame_idx = 0
        start_time = time.time()
        last_annotated_frame = None

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                timestamp_sec = frame_idx / fps if fps > 0 else 0.0
                timestamp_formatted = self.format_timestamp(timestamp_sec)

                # 1. Run persistent multi-object tracking on current frame
                tracks = self.tracker.track_frame(
                    frame=frame,
                    frame_idx=frame_idx,
                    timestamp_sec=timestamp_sec,
                    conf_threshold=conf_threshold
                )

                # 2. Log each observation under its persistent track ID
                for det_idx, trk in enumerate(tracks):
                    bbox = trk["bbox"]
                    self.logger.log_detection(
                        video_name=video_name,
                        frame_number=frame_idx,
                        timestamp_seconds=timestamp_sec,
                        timestamp_formatted=timestamp_formatted,
                        track_id=trk["track_id"],
                        confidence=trk["confidence"],
                        x1=bbox[0],
                        y1=bbox[1],
                        x2=bbox[2],
                        y2=bbox[3],
                        bbox_width=trk["bbox_width"],
                        bbox_height=trk["bbox_height"],
                        automatic_flag=trk.get("automatic_flag", False),
                        review_status=trk.get("review_status", "Keep"),
                        tracking_warning=trk.get("tracking_warning", "None"),
                        flag_reason=trk.get("flag_reason", "No Flag"),
                        det_idx=det_idx
                    )

                # 3. Annotate original frame with bounding boxes and SAR telemetry
                distinct_count = len(self.tracker.tracks_metadata)
                cumulative_obs = self.tracker.total_observations

                annotated = self.annotate_frame(
                    frame=frame,
                    tracks=tracks,
                    frame_idx=frame_idx,
                    timestamp_sec=timestamp_sec,
                    timestamp_formatted=timestamp_formatted,
                    cumulative_detections=cumulative_obs,
                    distinct_tracks_count=distinct_count,
                    review_threshold=review_threshold,
                    box_thickness=box_thickness,
                    font_scale=font_scale
                )
                last_annotated_frame = annotated

                # 4. Write to output video
                if writer is not None:
                    writer.write(annotated)

                # 5. Invoke progress callback
                if progress_callback:
                    progress_ratio = min(1.0, frame_idx / float(frames_to_process)) if frames_to_process > 0 else 0.5
                    current_elapsed = time.time() - start_time
                    current_fps = round(frame_idx / current_elapsed, 1) if current_elapsed > 0 else 0.0

                    callback_stats = {
                        "frame_idx": frame_idx,
                        "total_frames": frames_to_process,
                        "timestamp_formatted": timestamp_formatted,
                        "current_active_tracks": len(tracks),
                        "distinct_track_ids": distinct_count,
                        "total_detections": cumulative_obs,
                        "current_fps": current_fps
                    }
                    progress_callback(progress_ratio, annotated, callback_stats)

                if max_frames and frame_idx >= max_frames:
                    break

        finally:
            cap.release()
            if writer is not None:
                writer.release()

        elapsed_sec = time.time() - start_time
        measured_fps = round(frame_idx / elapsed_sec, 2) if elapsed_sec > 0 else 0.0

        metrics = self.logger.get_summary_metrics()

        return {
            "video_name": video_name,
            "frames_processed": frame_idx,
            "total_frames": total_frames,
            "total_observations": metrics["total_observations"],
            "distinct_track_ids": metrics["distinct_track_ids"],
            "needs_review_count": metrics["needs_review_count"],
            "keep_count": metrics["keep_count"],
            "reject_count": metrics["reject_count"],
            "total_warnings": metrics["total_warnings"],
            "avg_confidence": metrics["avg_confidence"],
            "processing_time_sec": round(elapsed_sec, 2),
            "measured_fps": measured_fps,
            "video_duration_sec": info["duration_sec"],
            "output_video_path": output_path if (output_path and os.path.exists(output_path)) else None,
            "last_frame": last_annotated_frame
        }
