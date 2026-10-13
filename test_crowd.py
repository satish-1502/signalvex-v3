"""
test_crowd.py — Dense Crowd Stress Test for SIGNALVEX_V2 (>35 People)
=====================================================================
Validates that tracker, logger, frame annotator, and summary stats
properly handle crowded flood/rescue scenes with 40+ simultaneous people.
"""

import os
import sys
import numpy as np
import cv2

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from tracker import HumanTracker
from logger import DetectionLogger
from video_processor import VideoProcessor


def test_crowded_scene():
    print("=" * 70)
    print("SIGNALVEX_V2 CROWD STRESS TEST (>35 PEOPLE)")
    print("=" * 70)

    # Create synthetic frame: 1920x1080 (HD drone view)
    w, h = 1920, 1080
    test_frame = np.full((h, w, 3), 40, dtype=np.uint8)

    # Initialize tracker and logger
    tracker = HumanTracker(model_instance=None, conf_threshold=0.30)
    logger = DetectionLogger()

    # Generate 42 people distributed across flood scene
    NUM_PEOPLE = 42
    print(f"\nSimulating {NUM_PEOPLE} simultaneous people across 5 consecutive frames...")

    # Define base positions for 42 people (grid with jitter)
    rows = 6
    cols = 7
    base_coords = []
    x_step = w // (cols + 1)
    y_step = h // (rows + 1)

    for r in range(rows):
        for c in range(cols):
            cx = (c + 1) * x_step
            cy = (r + 1) * y_step
            bw = 36
            bh = 72
            base_coords.append([cx - bw // 2, cy - bh // 2, cx + bw // 2, cy + bh // 2])

    assert len(base_coords) >= NUM_PEOPLE

    # Simulate 5 consecutive frames with persistent motion
    for f_idx in range(1, 6):
        ts = f_idx / 24.0
        ts_fmt = VideoProcessor.format_timestamp(ts)

        # Slightly move each person (simulating drone drift / slow motion)
        frame_boxes = []
        for i, box in enumerate(base_coords[:NUM_PEOPLE]):
            drift_x = int(np.sin(f_idx + i) * 2)
            drift_y = int(np.cos(f_idx + i) * 2)
            curr_box = [box[0] + drift_x, box[1] + drift_y, box[2] + drift_x, box[3] + drift_y]
            conf = 0.65 + 0.25 * np.sin(i)
            frame_boxes.append((curr_box, float(conf)))

        # Feed detections into fallback tracker directly to test track state management
        raw_dets = frame_boxes
        unmatched = list(range(len(raw_dets)))
        matched = []
        new_active = {}

        for tid, prev_box in list(tracker.fallback_active_boxes.items()):
            best_idx = None
            best_iou = 0.20
            for d_idx in unmatched:
                b, s = raw_dets[d_idx]
                iou = tracker._calc_iou(b, prev_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = d_idx
            if best_idx is not None:
                b, s = raw_dets[best_idx]
                unmatched.remove(best_idx)
                new_active[tid] = b
                warn = tracker._register_or_update_track(tid, b, f_idx, ts, s)
                tracker.total_observations += 1
                matched.append({
                    "track_id": tid, "bbox": b, "confidence": round(s, 4),
                    "confidence_percent": round(s * 100, 1),
                    "class_id": 0, "class_name": "person",
                    "bbox_width": b[2] - b[0], "bbox_height": b[3] - b[1],
                    "tracking_warning": warn, "review_status": "Keep"
                })

        for d_idx in unmatched:
            b, s = raw_dets[d_idx]
            tid = tracker.next_fallback_id
            tracker.next_fallback_id += 1
            new_active[tid] = b
            warn = tracker._register_or_update_track(tid, b, f_idx, ts, s)
            tracker.total_observations += 1
            matched.append({
                "track_id": tid, "bbox": b, "confidence": round(s, 4),
                "confidence_percent": round(s * 100, 1),
                "class_id": 0, "class_name": "person",
                "bbox_width": b[2] - b[0], "bbox_height": b[3] - b[1],
                "tracking_warning": warn, "review_status": "Keep"
            })

        tracker.fallback_active_boxes = new_active
        tracker.active_track_ids = set(new_active.keys())

        # Log into detection logger
        for m in matched:
            b = m["bbox"]
            logger.log_detection(
                video_name="crowded_flood_drone.mp4",
                frame_number=f_idx,
                timestamp_seconds=ts,
                timestamp_formatted=ts_fmt,
                track_id=m["track_id"],
                confidence=m["confidence"],
                x1=b[0], y1=b[1], x2=b[2], y2=b[3],
                bbox_width=m["bbox_width"], bbox_height=m["bbox_height"],
                review_status=m["review_status"],
                tracking_warning=m["tracking_warning"]
            )

        # Annotate frame
        annotated = VideoProcessor.annotate_frame(
            frame=test_frame,
            tracks=matched,
            frame_idx=f_idx,
            timestamp_sec=ts,
            timestamp_formatted=ts_fmt,
            cumulative_detections=tracker.total_observations,
            distinct_tracks_count=len(tracker.tracks_metadata),
            box_thickness=2,
            font_scale=0.45
        )

    metrics = logger.get_summary_metrics()
    print(f"\n[Results]")
    print(f"- Target Count In Scene:      {NUM_PEOPLE} people (>35 requirement met)")
    print(f"- Total Cumulative Obs:       {metrics['total_observations']} (5 frames * {NUM_PEOPLE} = {5*NUM_PEOPLE})")
    print(f"- Distinct Track IDs:         {metrics['distinct_track_ids']}")
    print(f"- Persistent ID Consistency:  {metrics['distinct_track_ids'] == NUM_PEOPLE}")

    assert metrics['total_observations'] == NUM_PEOPLE * 5, "Total observations mismatch"
    assert metrics['distinct_track_ids'] == NUM_PEOPLE, "Track IDs drifted or fragmented!"

    summary_df = logger.get_track_summary_dataframe()
    assert len(summary_df) == NUM_PEOPLE, "Summary dataframe row count mismatch"
    print(f"- Track Summary Table Rows:   {len(summary_df)}")
    print(f"- Annotated Frame Shape:      {annotated.shape}")
    print("=" * 70)
    print("CROWD STRESS TEST (>35 PEOPLE) PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    test_crowded_scene()
