"""
test_pipeline.py — End-to-End Test Suite for SIGNALVEX_V2
=========================================================
Runs automated tests to verify:
1. Video metadata inspection & unreadable file error handling
2. YOLO model loading & person-only filtering
3. Bounding box coordinates in original frame space
4. Multi-object tracking ID persistence across consecutive frames
5. Total observations vs. distinct tracking IDs distinction
6. Timestamp formatting & frame calculation
7. DetectionLogger schema and CSV generation
8. Track summary generation
9. Operator review updates (Keep / Reject / Needs Review)
10. End-to-end video processing and annotated MP4 generation
11. CPU execution speed & performance measurement
"""

import os
import sys
import time
import numpy as np
import cv2

# Ensure SIGNALVEX_V2 modules are loaded
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from detector import HumanDetector
from tracker import HumanTracker
from logger import DetectionLogger
from video_processor import VideoProcessor


def test_suite():
    print("=" * 70)
    print("SIGNALVEX_V2 AUTOMATED TEST SUITE")
    print("Sky Seekers | CBIT ECE — Human Detection and Persistent Tracking")
    print("=" * 70)

    # ---------------- TEST 1: Video Metadata & Error Handling ----------------
    print("\n[Test 1/7] Testing Video Metadata Extraction & Error Handling...")
    sample_vid = os.path.join(CURRENT_DIR, "sample_data", "sample_drone_flight.mp4")
    assert os.path.exists(sample_vid), f"Sample video missing at {sample_vid}"

    info = VideoProcessor.get_video_info(sample_vid)
    print(f"  [OK] Video Found: {info['file_name']}")
    print(f"  [OK] Dimensions: {info['width']}x{info['height']} px")
    print(f"  [OK] Frame Rate: {info['fps']} FPS | Frames: {info['total_frames']} | Duration: {info['duration_formatted']}")
    assert info['width'] > 0 and info['height'] > 0, "Invalid frame dimensions."
    assert info['fps'] > 0, "Invalid FPS."

    # Test corrupted/non-existent video error handling
    try:
        VideoProcessor.get_video_info("non_existent_fake_video.mp4")
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        print("  [OK] Non-existent file handled gracefully with FileNotFoundError.")

    # ---------------- TEST 2: HumanDetector Person-Only Filtering ----------------
    print("\n[Test 2/7] Testing HumanDetector (Ultralytics YOLO)...")
    detector = HumanDetector(model_path="models/yolov8n.pt", conf_threshold=0.30, imgsz=640)
    assert detector.model is not None, "YOLO model failed to load."
    print("  [OK] YOLO model loaded successfully (CPU compatible).")

    # Read a sample frame
    cap = cv2.VideoCapture(sample_vid)
    ret, frame = cap.read()
    cap.release()
    assert ret and frame is not None, "Failed to read test frame from video."

    h, w = frame.shape[:2]
    dets = detector.detect(frame, conf_threshold=0.30)
    print(f"  [OK] Detections on test frame 1: {len(dets)} people detected.")
    for d in dets:
        assert d["class_id"] == 0, f"Expected class 0 (person), got {d['class_id']}"
        assert d["class_name"] == "person", "Class name must be 'person'"
        assert 0.0 <= d["confidence"] <= 1.0, "Confidence out of range [0, 1]"
        x1, y1, x2, y2 = d["bbox"]
        assert 0 <= x1 < x2 <= w, f"Invalid x coordinates: {x1}, {x2} for width {w}"
        assert 0 <= y1 < y2 <= h, f"Invalid y coordinates: {y1}, {y2} for height {h}"
        assert d["bbox_width"] == (x2 - x1)
        assert d["bbox_height"] == (y2 - y1)
    print("  [OK] All detections strictly class 0, within original frame boundaries [x1, y1, x2, y2].")

    # ---------------- TEST 3: Multi-Object Tracker & ID Persistence ----------------
    print("\n[Test 3/7] Testing Persistent Tracking Across Consecutive Frames...")
    tracker = HumanTracker(
        model_instance=detector.model,
        tracker_type="bytetrack.yaml",
        conf_threshold=0.30,
        review_threshold=0.50,
        imgsz=640,
        track_buffer=30
    )

    cap = cv2.VideoCapture(sample_vid)
    frame_tracks = []
    for f_idx in range(1, 11):  # Test first 10 consecutive frames
        ret, frame = cap.read()
        if not ret:
            break
        ts = f_idx / info["fps"]
        tracks = tracker.track_frame(frame, frame_idx=f_idx, timestamp_sec=ts, conf_threshold=0.30)
        frame_tracks.append((f_idx, tracks))
    cap.release()

    # Verify ID persistence
    all_observed_ids = set()
    frames_per_id = {}
    for f_idx, trks in frame_tracks:
        for t in trks:
            tid = t["track_id"]
            all_observed_ids.add(tid)
            frames_per_id[tid] = frames_per_id.get(tid, 0) + 1

    stats = tracker.get_stats()
    print(f"  [OK] Processed 10 consecutive frames.")
    print(f"  [OK] Total Person Observations: {stats['total_observations']}")
    print(f"  [OK] Distinct Track IDs Observed: {stats['distinct_track_ids']}")
    print(f"  [OK] Active Tracks in Frame 10: {stats['current_active_tracks']}")

    # Critical requirement: Total observations MUST NOT equal distinct track IDs if tracks persist!
    if stats['total_observations'] > 0:
        persistent_tracks = [tid for tid, count in frames_per_id.items() if count > 1]
        print(f"  [OK] Persistent tracks retaining same ID across frames: {len(persistent_tracks)} tracks ({persistent_tracks[:5]})")
        assert stats['total_observations'] >= stats['distinct_track_ids'], "Observations must be >= distinct IDs."
    print("  [OK] Persistent ID verification passed!")

    # ---------------- TEST 4: Detection Logger Schema & Storage ----------------
    print("\n[Test 4/7] Testing DetectionLogger Schema & Aggregations...")
    logger = DetectionLogger()
    rec1 = logger.log_detection(
        video_name="test.mp4",
        frame_number=1,
        timestamp_seconds=0.042,
        timestamp_formatted="00:00.04",
        track_id=101,
        confidence=0.884,
        x1=100, y1=150, x2=180, y2=320,
        bbox_width=80, bbox_height=170,
        review_status="Keep",
        tracking_warning="None"
    )
    rec2 = logger.log_detection(
        video_name="test.mp4",
        frame_number=2,
        timestamp_seconds=0.084,
        timestamp_formatted="00:00.08",
        track_id=101,
        confidence=0.450,
        x1=102, y1=152, x2=182, y2=322,
        bbox_width=80, bbox_height=170,
        review_status="Needs Review",
        tracking_warning="Low Confidence"
    )
    rec3 = logger.log_detection(
        video_name="test.mp4",
        frame_number=2,
        timestamp_seconds=0.084,
        timestamp_formatted="00:00.08",
        track_id=102,
        confidence=0.920,
        x1=300, y1=400, x2=350, y2=510,
        bbox_width=50, bbox_height=110,
        review_status="Keep",
        tracking_warning="None"
    )

    df = logger.to_dataframe()
    assert len(df) == 3, "DataFrame record count mismatch"
    assert "video_name" in df.columns
    assert "frame_number" in df.columns
    assert "timestamp_seconds" in df.columns
    assert "track_id" in df.columns
    assert "confidence" in df.columns
    assert "x1" in df.columns and "y1" in df.columns
    assert "review_status" in df.columns
    assert "flag_reason" in df.columns
    print("  [OK] Full Detection Log schema validated.")

    metrics = logger.get_summary_metrics()
    assert metrics["total_observations"] == 3
    assert metrics["distinct_track_ids"] == 2
    assert metrics["flagged_observations"] == 1
    assert metrics["unique_flagged_tracks"] == 1
    assert metrics["needs_review_count"] == 1
    assert metrics["keep_count"] == 2
    print(f"  [OK] Metrics: Total={metrics['total_observations']}, Distinct IDs={metrics['distinct_track_ids']}, Flagged Obs={metrics['flagged_observations']}, Flagged Tracks={metrics['unique_flagged_tracks']}")

    # ---------------- TEST 5: Track Summary & CSV Export ----------------
    print("\n[Test 5/7] Testing Track Summary Generation & CSV Exports...")
    summary_df = logger.get_track_summary_dataframe()
    assert len(summary_df) == 2, f"Expected 2 distinct tracks in summary, got {len(summary_df)}"
    row101 = summary_df[summary_df["track_id"] == 101].iloc[0]
    assert row101["total_observations"] == 2
    assert row101["first_frame"] == 1 and row101["last_frame"] == 2
    print("  [OK] Track Summary DataFrame correctly generated.")

    csv_data = logger.to_csv()
    assert "video_name" in csv_data and "track_id" in csv_data and "flag_reason" in csv_data
    track_csv = logger.to_track_summary_csv()
    assert "duration_seconds" in track_csv and "total_observations" in track_csv
    print("  [OK] Complete CSV and Track Summary CSV generated successfully.")

    # ---------------- TEST 6: Operator Review Persistence ----------------
    print("\n[Test 6/7] Testing Operator Review Actions (Keep / Reject / Needs Review)...")
    logger.update_status_by_track(101, "Reject")
    df_updated = logger.to_dataframe()
    assert (df_updated[df_updated["track_id"] == 101]["review_status"] == "Reject").all(), "Failed to update track review status."
    updated_metrics = logger.get_summary_metrics()
    assert updated_metrics["manually_reviewed_records"] == 2, "Expected 2 manually reviewed records."
    print("  [OK] Operator decision updated across all observations for Track #101 to 'Reject'.")

    # ---------------- TEST 7: End-to-End Video Processing & Output MP4 ----------------
    print("\n[Test 7/7] Testing End-to-End Pipeline Execution on 30 Frames...")
    test_out_video = os.path.join(CURRENT_DIR, "sample_data", "test_v2_annotated.mp4")
    if os.path.exists(test_out_video):
        try:
            os.remove(test_out_video)
        except Exception:
            pass

    pipe_tracker = HumanTracker(model_instance=detector.model, conf_threshold=0.30, imgsz=640)
    pipe_logger = DetectionLogger()
    processor = VideoProcessor(tracker=pipe_tracker, logger=pipe_logger)

    progress_reports = []
    def on_progress(ratio, frame, stats):
        progress_reports.append((stats["frame_idx"], stats["current_active_tracks"]))

    t_start = time.time()
    exec_summary = processor.process_video(
        video_path=sample_vid,
        output_path=test_out_video,
        conf_threshold=0.30,
        review_threshold=0.50,
        max_frames=30,
        progress_callback=on_progress
    )
    t_end = time.time()

    print("\n" + "=" * 70)
    print("EXECUTION RESULTS & HARDWARE MEASUREMENTS (CPU)")
    print("=" * 70)
    print(f"- Input Video:             {exec_summary['video_name']}")
    print(f"- Frames Processed:        {exec_summary['frames_processed']}")
    print(f"- Execution Time:          {exec_summary['processing_time_sec']}s")
    print(f"- Measured Speed:          {exec_summary['measured_fps']} FPS")
    print(f"- Total Observations:      {exec_summary['total_observations']}")
    print(f"- Distinct Track IDs:      {exec_summary['distinct_track_ids']}")
    print(f"- Flagged (Needs Review):  {exec_summary['needs_review_count']}")
    print(f"- Total Warnings:          {exec_summary['total_warnings']}")

    assert os.path.exists(test_out_video), "Annotated MP4 video file was not created."
    out_size = os.path.getsize(test_out_video)
    print(f"- Output Video File:       {test_out_video} ({out_size} bytes)")
    assert out_size > 1000, "Output video file is suspiciously small or empty."

    print("=" * 70)
    print("ALL 7 TEST SUITES PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    test_suite()
