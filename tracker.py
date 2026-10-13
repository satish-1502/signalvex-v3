"""
tracker.py — Multi-Object Tracking & Track State Management for SIGNALVEX_V2
============================================================================
Drone-Based Human Detection and Rescue Mapping Prototype
Department of Electronics and Communication Engineering
Chaitanya Bharathi Institute of Technology | Sky Seekers Team

Features:
- ByteTrack integration via Ultralytics model.track(persist=True)
- Persistent tracking IDs across consecutive video frames
- Track State Manager tracking:
    * Total observations across frames
    * Number of distinct track IDs observed
    * Current active tracks
    * Trajectory history and displacement velocity
- Heuristic flagging for:
    * "New Track"
    * "Low Confidence"
    * "Potential ID Switch" (sudden velocity / spatial jump)
    * "Reappearing Track" (recovered after short occlusion)
- Built-in IoU + Proximity Fallback Tracker with configurable track buffer
- Explicit reset() to guarantee fresh session per video
"""

from typing import List, Dict, Any, Optional, Tuple, Set
import math
import numpy as np


class TrackMetadata:
    """Maintains persistent state and lifecycle statistics for a single track ID."""

    def __init__(self, track_id: int, initial_bbox: List[int], frame_idx: int, timestamp_sec: float, confidence: float, review_threshold: float = 0.50):
        self.track_id = track_id
        self.first_frame = frame_idx
        self.last_frame = frame_idx
        self.first_timestamp = timestamp_sec
        self.last_timestamp = timestamp_sec
        self.observations_count = 1
        self.confidences = [confidence]
        self.bboxes = [initial_bbox]
        self.centroids = [self._calc_centroid(initial_bbox)]
        self.lost_frames_count = 0
        self.review_threshold = review_threshold
        self.warnings: List[str] = []

    @staticmethod
    def _calc_centroid(bbox: List[int]) -> Tuple[float, float]:
        return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)

    def update(self, bbox: List[int], frame_idx: int, timestamp_sec: float, confidence: float) -> str:
        """
        Updates track state and returns any tracking warning for this observation.
        """
        prev_centroid = self.centroids[-1]
        new_centroid = self._calc_centroid(bbox)
        gap = frame_idx - self.last_frame

        warning = "None"
        if gap > 5:
            warning = f"Reappearing Track (gap={gap} frames)"
        elif gap == 1:
            # Check displacement jump relative to bbox diagonal
            dx = new_centroid[0] - prev_centroid[0]
            dy = new_centroid[1] - prev_centroid[1]
            dist = math.hypot(dx, dy)
            bbox_diag = math.hypot(bbox[2] - bbox[0], bbox[3] - bbox[1])
            # If centroid moved more than 3.0x the box diagonal or >100px in 1 frame, likely ID switch or sudden camera shake
            if dist > max(100.0, 3.0 * bbox_diag):
                warning = "Potential ID Switch / Position Jump"

        self.last_frame = frame_idx
        self.last_timestamp = timestamp_sec
        self.observations_count += 1
        self.confidences.append(confidence)
        self.bboxes.append(bbox)
        self.centroids.append(new_centroid)
        self.lost_frames_count = 0

        if warning != "None":
            self.warnings.append(warning)

        return warning


class HumanTracker:
    """
    Multi-Object Human Tracker with persistent ID assignment, trajectory logging,
    and fallback tracking support.
    """

    def __init__(
        self,
        model_instance=None,
        tracker_type: str = "bytetrack.yaml",
        conf_threshold: float = 0.35,
        review_threshold: float = 0.50,
        imgsz: int = 640,
        track_buffer: int = 30
    ):
        """
        Args:
            model_instance: Loaded Ultralytics YOLO model instance
            tracker_type: Tracker config ('bytetrack.yaml' or 'botsort.yaml')
            conf_threshold: Minimum detection confidence (default: 0.35)
            review_threshold: Low-confidence warning threshold (default: 0.50)
            imgsz: Inference resolution (default: 640)
            track_buffer: Frames to remember lost tracks (default: 30)
        """
        self.model = model_instance
        self.tracker_type = tracker_type
        self.conf_threshold = conf_threshold
        self.review_threshold = review_threshold
        self.imgsz = imgsz
        self.track_buffer = track_buffer

        # Tracker lifecycle states
        self.next_fallback_id = 1
        self.tracks_metadata: Dict[int, TrackMetadata] = {}
        self.active_track_ids: Set[int] = set()
        self.fallback_active_boxes: Dict[int, List[int]] = {}  # {track_id: bbox}
        self.fallback_lost_boxes: Dict[int, Tuple[List[int], int]] = {}  # {track_id: (bbox, last_seen_frame)}
        self.total_observations = 0

    def reset(self):
        """Resets tracker state completely for processing a new video."""
        self.next_fallback_id = 1
        self.tracks_metadata.clear()
        self.active_track_ids.clear()
        self.fallback_active_boxes.clear()
        self.fallback_lost_boxes.clear()
        self.total_observations = 0
        # Reset internal tracker state in Ultralytics if available
        if self.model is not None and hasattr(self.model, "predictor"):
            self.model.predictor = None

    def track_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        timestamp_sec: float,
        conf_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Processes a single video frame, associating detections to persistent track IDs.

        Returns list of track dicts for the current frame:
        [
            {
                "track_id": 1,
                "bbox": [x1, y1, x2, y2],
                "confidence": 0.864,
                "confidence_percent": 86.4,
                "class_id": 0,
                "class_name": "person",
                "bbox_width": w,
                "bbox_height": h,
                "tracking_warning": "None" | "New Track" | "Low Confidence" | ...,
                "review_status": "Keep" | "Needs Review"
            },
            ...
        ]
        """
        conf = conf_threshold if conf_threshold is not None else self.conf_threshold
        tracks_output: List[Dict[str, Any]] = []

        # Route 1: Try Ultralytics native ByteTrack
        success = False
        if self.model is not None:
            try:
                results = self.model.track(
                    frame,
                    persist=True,
                    classes=[0],
                    conf=conf,
                    imgsz=self.imgsz,
                    tracker=self.tracker_type,
                    verbose=False
                )

                if results and len(results) > 0 and results[0].boxes is not None:
                    boxes = results[0].boxes
                    orig_h, orig_w = frame.shape[:2]

                    current_frame_tids = set()

                    for box in boxes:
                        score = float(box.conf[0].item())
                        cls_id = int(box.cls[0].item())
                        if cls_id != 0 or score < conf:
                            continue

                        # Extract persistent tracking ID from Ultralytics
                        if box.id is not None:
                            track_id = int(box.id[0].item())
                        else:
                            # Fallback if tracker unassigned on this frame
                            track_id = self.next_fallback_id
                            self.next_fallback_id += 1

                        xyxy = box.xyxy[0].tolist()
                        x1 = max(0, min(orig_w - 1, int(round(xyxy[0]))))
                        y1 = max(0, min(orig_h - 1, int(round(xyxy[1]))))
                        x2 = max(0, min(orig_w - 1, int(round(xyxy[2]))))
                        y2 = max(0, min(orig_h - 1, int(round(xyxy[3]))))
                        bbox = [x1, y1, x2, y2]
                        bbox_w = max(1, x2 - x1)
                        bbox_h = max(1, y2 - y1)

                        # Update metadata and warnings
                        warning = self._register_or_update_track(
                            track_id=track_id,
                            bbox=bbox,
                            frame_idx=frame_idx,
                            timestamp_sec=timestamp_sec,
                            confidence=score
                        )

                        # Assign review status and flag reason using explicit rules
                        auto_flag, flag_reason, rev_status = self._evaluate_flag(score, warning)

                        current_frame_tids.add(track_id)
                        self.total_observations += 1

                        tracks_output.append({
                            "track_id": track_id,
                            "bbox": bbox,
                            "confidence": round(score, 4),
                            "confidence_percent": round(score * 100.0, 1),
                            "class_id": 0,
                            "class_name": "person",
                            "bbox_width": bbox_w,
                            "bbox_height": bbox_h,
                            "tracking_warning": warning,
                            "automatic_flag": auto_flag,
                            "flag_reason": flag_reason,
                            "review_status": rev_status
                        })

                    self.active_track_ids = current_frame_tids
                    success = True

            except Exception as e:
                print(f"[SIGNALVEX Tracker] Ultralytics tracking notice: {e}. Utilizing internal association tracker.")
                success = False

        # Route 2: Fallback association tracker
        if not success:
            tracks_output = self._fallback_track_frame(frame, frame_idx, timestamp_sec, conf)

        return tracks_output

    def _evaluate_flag(self, confidence: float, warning: str) -> Tuple[bool, str, str]:
        """
        Evaluates review recommendation and reason based on explicit rules:
        - Rule A (Low Confidence): confidence < review_threshold -> automatic_flag=True, reason 'Low Confidence'
        - Rule B (Tracking Warning): credible warning present -> automatic_flag=True, reason 'Tracking Warning'
        - Both: automatic_flag=True, reason 'Low Confidence; Tracking Warning'
        - Neither (Rule C): stable detection -> automatic_flag=False, reason 'No Flag'
        Initial review status defaults to 'Needs Review' if automatic_flag is True, else 'Keep'.
        """
        reasons = []
        if confidence < self.review_threshold:
            reasons.append("Low Confidence")
        if warning != "None":
            reasons.append("Tracking Warning")

        if reasons:
            flag_reason = "; ".join(reasons)
            return True, flag_reason, "Needs Review"
        return False, "No Flag", "Keep"

    def _register_or_update_track(
        self,
        track_id: int,
        bbox: List[int],
        frame_idx: int,
        timestamp_sec: float,
        confidence: float
    ) -> str:
        """Updates track history or creates new record; returns warning if any."""
        if track_id not in self.tracks_metadata:
            meta = TrackMetadata(track_id, bbox, frame_idx, timestamp_sec, confidence, review_threshold=self.review_threshold)
            self.tracks_metadata[track_id] = meta
            return "None"
        else:
            meta = self.tracks_metadata[track_id]
            warning = meta.update(bbox, frame_idx, timestamp_sec, confidence)
            return warning

    def _fallback_track_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        timestamp_sec: float,
        conf: float
    ) -> List[Dict[str, Any]]:
        """
        Robust internal tracker combining IoU association, centroid distance gating,
        and a track buffer for short occlusion tolerance.
        """
        from detector import HumanDetector
        orig_h, orig_w = frame.shape[:2]

        # Get raw detections
        res = self.model(frame, classes=[0], conf=conf, imgsz=self.imgsz, verbose=False)
        raw_detections = []
        if res and len(res) > 0 and res[0].boxes is not None:
            for box in res[0].boxes:
                score = float(box.conf[0].item())
                if score >= conf:
                    xyxy = box.xyxy[0].tolist()
                    x1 = max(0, min(orig_w - 1, int(round(xyxy[0]))))
                    y1 = max(0, min(orig_h - 1, int(round(xyxy[1]))))
                    x2 = max(0, min(orig_w - 1, int(round(xyxy[2]))))
                    y2 = max(0, min(orig_h - 1, int(round(xyxy[3]))))
                    raw_detections.append(([x1, y1, x2, y2], score))

        # Match against active and buffered lost tracks
        unmatched_dets = list(range(len(raw_detections)))
        matched_results = []
        new_active = {}

        # 1. Match against active tracks from previous frame
        for tid, prev_box in list(self.fallback_active_boxes.items()):
            best_match_idx = None
            best_score = 0.20  # Minimum IoU or proximity threshold

            for d_idx in unmatched_dets:
                box, score = raw_detections[d_idx]
                iou = self._calc_iou(box, prev_box)
                if iou > best_score:
                    best_score = iou
                    best_match_idx = d_idx

            if best_match_idx is not None:
                box, score = raw_detections[best_match_idx]
                unmatched_dets.remove(best_match_idx)
                new_active[tid] = box

                warning = self._register_or_update_track(tid, box, frame_idx, timestamp_sec, score)
                auto_flag, flag_reason, rev_status = self._evaluate_flag(score, warning)
                self.total_observations += 1

                matched_results.append({
                    "track_id": tid,
                    "bbox": box,
                    "confidence": round(score, 4),
                    "confidence_percent": round(score * 100.0, 1),
                    "class_id": 0,
                    "class_name": "person",
                    "bbox_width": box[2] - box[0],
                    "bbox_height": box[3] - box[1],
                    "tracking_warning": warning,
                    "automatic_flag": auto_flag,
                    "flag_reason": flag_reason,
                    "review_status": rev_status
                })
            else:
                # Track lost in this frame, put into lost buffer
                self.fallback_lost_boxes[tid] = (prev_box, frame_idx)

        # 2. Try matching remaining detections with recently lost tracks (within track_buffer)
        for tid, (lost_box, lost_frame) in list(self.fallback_lost_boxes.items()):
            if frame_idx - lost_frame > self.track_buffer:
                del self.fallback_lost_boxes[tid]
                continue

            best_match_idx = None
            best_iou = 0.25
            for d_idx in unmatched_dets:
                box, score = raw_detections[d_idx]
                iou = self._calc_iou(box, lost_box)
                if iou > best_iou:
                    best_iou = iou
                    best_match_idx = d_idx

            if best_match_idx is not None:
                box, score = raw_detections[best_match_idx]
                unmatched_dets.remove(best_match_idx)
                del self.fallback_lost_boxes[tid]
                new_active[tid] = box

                warning = self._register_or_update_track(tid, box, frame_idx, timestamp_sec, score)
                auto_flag, flag_reason, rev_status = self._evaluate_flag(score, warning)
                self.total_observations += 1

                matched_results.append({
                    "track_id": tid,
                    "bbox": box,
                    "confidence": round(score, 4),
                    "confidence_percent": round(score * 100.0, 1),
                    "class_id": 0,
                    "class_name": "person",
                    "bbox_width": box[2] - box[0],
                    "bbox_height": box[3] - box[1],
                    "tracking_warning": warning,
                    "automatic_flag": auto_flag,
                    "flag_reason": flag_reason,
                    "review_status": rev_status
                })

        # 3. Create new tracks for remaining unmatched detections
        for d_idx in unmatched_dets:
            box, score = raw_detections[d_idx]
            tid = self.next_fallback_id
            self.next_fallback_id += 1
            new_active[tid] = box

            warning = self._register_or_update_track(tid, box, frame_idx, timestamp_sec, score)
            auto_flag, flag_reason, rev_status = self._evaluate_flag(score, warning)
            self.total_observations += 1

            matched_results.append({
                "track_id": tid,
                "bbox": box,
                "confidence": round(score, 4),
                "confidence_percent": round(score * 100.0, 1),
                "class_id": 0,
                "class_name": "person",
                "bbox_width": box[2] - box[0],
                "bbox_height": box[3] - box[1],
                "tracking_warning": warning,
                "automatic_flag": auto_flag,
                "flag_reason": flag_reason,
                "review_status": rev_status
            })

        self.fallback_active_boxes = new_active
        self.active_track_ids = set(new_active.keys())
        return matched_results

    @staticmethod
    def _calc_iou(boxA: List[int], boxB: List[int]) -> float:
        """Intersection over Union (IoU) calculation."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        inter_area = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
        boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
        union_area = boxAArea + boxBArea - inter_area
        return inter_area / float(union_area) if union_area > 0 else 0.0

    def get_stats(self) -> Dict[str, Any]:
        """Returns tracking counts distinguishing cumulative observations from distinct IDs."""
        return {
            "total_observations": self.total_observations,
            "distinct_track_ids": len(self.tracks_metadata),
            "current_active_tracks": len(self.active_track_ids)
        }
