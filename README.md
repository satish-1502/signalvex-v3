# SIGNALVEX — AI-Based Human Detection and Persistent Tracking in Flood-Disaster Video

**Sky Seekers | Rescue Intelligence Research Group**  
*Department of Electronics and Communication Engineering*  
*Chaitanya Bharathi Institute of Technology (CBIT)*  

---

## 1. Project Overview & Objective

**SIGNALVEX** is a software-only search-and-rescue (SAR) support prototype designed for aerial drone flood disaster footage. It takes AI-generated or recorded aerial drone RGB video, detects visible humans using a pretrained Ultralytics YOLO model, tracks individuals persistently across video frames using ByteTrack, burns rectangular bounding boxes and tracking IDs into the original video, and provides a real-time Streamlit Ground Rescue Dashboard with an operator review workflow and CSV data export.

### Core Processing Workflow

$$\text{Upload Flood Video} \longrightarrow \text{Read Original Frames} \longrightarrow \text{YOLO Human Detection} \longrightarrow \text{Confidence Filtering} \longrightarrow \text{Multi-Object Tracking} \longrightarrow \text{Bounding Boxes \& IDs} \longrightarrow \text{Timestamped Log} \longrightarrow \text{Ground Dashboard} \longrightarrow \text{Operator Review} \longrightarrow \text{CSV \& MP4 Export}$$

---

## 2. Key Features

- **Incrementally Streamed Video Ingestion:** OpenCV-based frame-by-frame decoding prevents memory exhaustion on large AI video files.
- **Strict Human Filtering:** Restricts detection exclusively to COCO Class 0 (`person`).
- **Original-Frame Coordinate Preservation:** Retains pixel coordinates `[x1, y1, x2, y2]`, width, and height in the native video resolution without distorted preview transformations.
- **Persistent Person Tracking:** Implements ByteTrack (`model.track(persist=True)`) and an internal association tracker with a configurable occlusion buffer (`track_buffer=30`), ensuring the same person retains their tracking ID across consecutive frames.
- **Distinction Between Detections and Unique People:** Explicitly separates *Total Observations* (frame-by-frame detections) from *Distinct Track IDs* (approximate individual identities) and *Active Targets* (current frame).
- **Dense Crowd Support:** Tested and verified on crowded flood scenes containing $>35$ simultaneous people.
- **Real-Time Operator Review:** Allows ground operators to assign `Keep`, `Reject`, or `Needs Review` status on individual detections or across entire tracks.
- **Multi-Format Export:** One-click downloads for Complete Detection Log CSV, Reviewed Detection Log CSV, Approximate Track Summary CSV, and Annotated MP4 Video.

---

## 3. Data Schemas

### A. Detection Log Schema (`signalvex_detection_log.csv`)

| Column Name | Type | Description |
|---|---|---|
| `video_name` | `string` | Source video file name |
| `frame_number` | `int` | Current video frame index (1-based) |
| `timestamp_seconds` | `float` | Presentation timestamp in seconds (calculated as `frame_number / fps`) |
| `timestamp_formatted` | `string` | Presentation timestamp in `MM:SS.ss` format |
| `track_id` | `int` | Persistent multi-object tracking identifier |
| `confidence` | `float` | Raw model confidence score `[0.0000 - 1.0000]` |
| `confidence_percent` | `float` | Scaled confidence percentage `[0.0 - 100.0%]` (matches video overlay) |
| `x1` | `int` | Left edge coordinate in native frame pixels |
| `y1` | `int` | Top edge coordinate in native frame pixels |
| `x2` | `int` | Right edge coordinate in native frame pixels |
| `y2` | `int` | Bottom edge coordinate in native frame pixels |
| `bbox_width` | `int` | Bounding box width `(x2 - x1)` in native pixels |
| `bbox_height` | `int` | Bounding box height `(y2 - y1)` in native pixels |
| `automatic_flag` | `bool` | Rule recommendation: `True` if low confidence (< review_threshold) or warning |
| `flag_reason` | `string` | Transparent reason: `Low Confidence`, `Tracking Warning`, `Manual Review`, or `No Flag` |
| `review_status` | `string` | Operator decision state: `Keep`, `Reject`, or `Needs Review` |
| `manual_reviewed` | `bool` | Operator decision flag: `True` if explicitly reviewed by human operator |
| `tracking_warning` | `string` | Heuristic warning: `None`, `Potential ID Switch`, or `Reappearing Track` |

### B. Track Summary Schema (`signalvex_track_summary.csv`)

| Column Name | Type | Description |
|---|---|---|
| `track_id` | `int` | Distinct tracking identifier |
| `first_frame` | `int` | Frame index where this person first appeared |
| `last_frame` | `int` | Frame index where this person was last observed |
| `first_timestamp` | `string` | First observed timestamp (`MM:SS.ss`) |
| `last_timestamp` | `string` | Last observed timestamp (`MM:SS.ss`) |
| `duration_seconds` | `float` | Total visible duration in seconds |
| `total_observations` | `int` | Total number of frames in which this track was detected |
| `avg_confidence_percent` | `float` | Mean confidence percentage across all observations |
| `min_confidence_percent` | `float` | Lowest confidence percentage recorded |
| `max_confidence_percent` | `float` | Peak confidence percentage recorded |
| `review_status` | `string` | Final human operator status (`Keep`, `Reject`, `Needs Review`) |
| `warnings_count` | `int` | Count of anomaly flags associated with this track |
| `tracking_note` | `string` | Disclaimer: *"Approximate track estimate — not verified unique biological survivors"* |

---

## 4. Execution & Testing

### Step 1: Open PowerShell in Working Copy
```powershell
cd C:\Users\HP\.gemini\antigravity\scratch\SIGNALVEX_WORKING_COPY
```

### Step 2: Run Automated Audit Test Suite
```powershell
..\drone-rescue-ai\venv\Scripts\python.exe test_audit_fixes.py
..\drone-rescue-ai\venv\Scripts\python.exe test_pipeline.py
..\drone-rescue-ai\venv\Scripts\python.exe test_crowd.py
```

### Step 3: Launch Improved Streamlit Dashboard on Independent Port (e.g. 8503)
```powershell
..\drone-rescue-ai\venv\Scripts\python.exe -m streamlit run app.py --server.port 8503
```
Access at `http://localhost:8503` without affecting existing public or local instances on ports 8501/8502.

---

## 5. End-to-End Demonstration Checklist

1. **Verify Startup:** Observe the title **SIGNALVEX — AI-Based Human Detection and Rescue Analysis** and CBIT ECE Sky Seekers banner.
2. **Video Ingestion:** Choose either the packaged aerial video or upload an AI-generated flood video (`.mp4`). Verify video metadata (dimensions, FPS, duration).
3. **Configure Parameters:**
   - Confidence threshold (e.g. 0.35)
   - Flag threshold (e.g. 0.50)
   - Inference resolution (640 or 960 for small aerial targets)
   - Box line thickness and font size
4. **Trigger Processing:** Click **🚀 Start Processing Video**.
5. **Observe Live Output:**
   - Real-time progress bar and FPS measurement.
   - Live annotated feed displaying bounding boxes, `ID {trk} | {conf}%` tags, and telemetry bar.
   - Cumulative observations counter vs. distinct track IDs.
6. **Inspect Summary Metrics:** Confirm the distinction between Total Observations and Distinct Track IDs.
7. **Filter Detection Log:** Filter by Track ID, minimum confidence, frame range, or review status.
8. **Inspect Track History:** Open the Track Inspector to see observation count, timeline, and confidence graph.
9. **Perform Operator Review:** Select a track ID and update status to `Keep` or `Reject`. Confirm table updates immediately.
10. **Export Results:**
    - Click **Download Complete Detection Log (CSV)**.
    - Click **Download Reviewed Detection Log (CSV)**.
    - Click **Download Track Summary (CSV)**.
    - Click **Download Annotated Video (MP4)**.

---

## 6. Academic Scope, Disclaimers & Safety Notes

In accordance with the college project presentation (*Sky Seekers | CBIT ECE*):
1. **Software Prototype:** Runs on recorded RGB video. No physical drone hardware is operated.
2. **RGB Visible Light Only:** Detections are based solely on human silhouettes in visible light. Thermal imaging, Wi-Fi CSI sensing, radar, and acoustic detection are future research paths.
3. **Tracking Identity Limitations:** Track IDs are software motion trajectories, not biological identifications. Occlusions or abrupt camera motion may cause ID switching.
4. **No Real GPS / Geo-Location:** No GPS telemetry or pixel-to-ground coordinates are claimed or simulated in this submission. All coordinates are native pixel bounds `[x1, y1, x2, y2]`.
5. **Non-Life-Safety Tool:** This software is an academic decision-support aid designed for college evaluation. It is not certified for autonomous life-safety operations.
