"""
app.py — Streamlit Ground Dashboard for SIGNALVEX_V2
====================================================
SIGNALVEX — AI-Based Human Detection and Rescue Analysis
Drone-Based Human Detection and Rescue Mapping Prototype
Department of Electronics and Communication Engineering
Chaitanya Bharathi Institute of Technology (CBIT) | Sky Seekers Team

Pipeline:
Upload Flood Video -> Read Original Frames -> YOLO Human Detection ->
Confidence Filtering -> ByteTrack Multi-Object Tracking -> Bounding Boxes & Tracking IDs ->
Timestamped Detection Log -> Streamlit Dashboard -> Operator Review -> CSV & Video Export
"""

import os
import sys
import tempfile
import time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# Ensure current folder is first in path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from detector import HumanDetector
from tracker import HumanTracker
from logger import DetectionLogger
from video_processor import VideoProcessor

# Streamlit Page Configuration
st.set_page_config(
    page_title="SIGNALVEX | AI Human Detection & Persistent Tracking",
    page_icon="🚁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Tech Dark / SAR CSS Styling
st.markdown("""
<style>
    /* Dark Slate & Navy SAR Palette */
    .stApp {
        background-color: #0b1120;
        color: #f1f5f9;
    }
    .main-title {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        color: #38bdf8;
        margin-bottom: 2px;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #94a3b8;
        font-weight: 500;
        margin-top: 0px;
        margin-bottom: 12px;
    }
    .cbit-badge {
        font-size: 0.80rem;
        color: #38bdf8;
        background-color: #0f172a;
        border: 1px solid #1e293b;
        padding: 5px 12px;
        border-radius: 6px;
        display: inline-block;
        margin-bottom: 16px;
        font-family: monospace;
    }
    .sar-card {
        background-color: #0f172a;
        border: 1px solid #1e293b;
        border-left: 4px solid #0284c7;
        padding: 14px 18px;
        border-radius: 8px;
        margin-bottom: 12px;
    }
    .sar-val {
        font-size: 1.7rem;
        font-weight: 700;
        color: #f8fafc;
        font-family: monospace;
    }
    .sar-label {
        font-size: 0.78rem;
        color: #94a3b8;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.5px;
        margin-top: 2px;
    }
    .notice-box {
        background-color: #0f172a;
        border: 1px solid #0369a1;
        border-radius: 8px;
        padding: 12px 16px;
        font-size: 0.88rem;
        color: #bae6fd;
        margin-bottom: 16px;
    }
    .warning-box {
        background-color: #1c1917;
        border: 1px solid #b45309;
        border-radius: 8px;
        padding: 12px 16px;
        font-size: 0.88rem;
        color: #fde68a;
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session State
if "detection_logger" not in st.session_state:
    st.session_state.detection_logger = DetectionLogger()
if "processing_complete" not in st.session_state:
    st.session_state.processing_complete = False
if "processing_summary" not in st.session_state:
    st.session_state.processing_summary = None
if "annotated_video_path" not in st.session_state:
    st.session_state.annotated_video_path = None
if "active_video_name" not in st.session_state:
    st.session_state.active_video_name = None


# Cached Model Loader
@st.cache_resource(show_spinner="Loading YOLOv8n detector weights into memory...")
def load_detector_model(model_path: str, conf_threshold: float, imgsz: int):
    return HumanDetector(model_path=model_path, conf_threshold=conf_threshold, imgsz=imgsz)


# =========================================================================
# SIDEBAR CONTROLS & MODEL PARAMETERS
# =========================================================================
with st.sidebar:
    st.markdown("### 🚁 **SIGNALVEX_V2**")
    st.markdown("**AI Human Detection & Rescue Tracking**")
    st.caption("Chaitanya Bharathi Institute of Technology (CBIT)\nDept. of ECE · Sky Seekers Research Group")

    st.divider()
    st.markdown("#### ⚙️ Detection & Tracking Settings")

    model_default_path = os.path.join(CURRENT_DIR, "models", "yolov8n.pt")
    if not os.path.exists(model_default_path):
        model_default_path = "yolov8n.pt"

    conf_thresh = st.slider(
        "Confidence Threshold (min)",
        min_value=0.10,
        max_value=0.90,
        value=0.35,
        step=0.05,
        help="Detection threshold. Values around 0.30–0.40 maximize recall on distant drone views."
    )

    review_threshold = st.slider(
        "Flag for Review Threshold",
        min_value=0.20,
        max_value=0.80,
        value=0.50,
        step=0.05,
        help="Detections below this threshold are drawn in amber and flagged as 'Needs Review'."
    )

    imgsz_choice = st.selectbox(
        "Inference Resolution (imgsz)",
        options=[640, 960, 1280],
        index=0,
        help="Larger resolutions detect small humans in high-altitude drone footage at moderate CPU cost."
    )

    st.markdown("#### 🎯 Visualization & Display")
    box_thickness = st.slider("Bounding Box Thickness (px)", min_value=1, max_value=4, value=2)
    font_scale = st.slider("Label Font Scale", min_value=0.35, max_value=0.80, value=0.48, step=0.02)
    track_buffer = st.slider("Track Buffer (frames)", min_value=10, max_value=90, value=30, step=5,
                             help="How many frames to remember a track through temporary occlusions.")

    max_frames_to_process = st.number_input(
        "Frame Limit (0 = Full Video)",
        min_value=0,
        max_value=10000,
        value=0,
        step=50,
        help="Optional limit for rapid faculty live demonstrations on CPU."
    )

    st.divider()
    st.markdown("#### 🛡️ Compliance & Limitations")
    st.caption("""
    • **RGB Silhouette Only:** Detects visible human-like shapes.
    • **Software Track IDs:** IDs represent motion trajectories, not biological identification.
    • **Non-Life Safety:** Decision-support software for college demonstration.
    """)


# =========================================================================
# MAIN HEADER & ARCHITECTURE
# =========================================================================
st.markdown('<div class="main-title">SIGNALVEX — AI-Based Human Detection and Rescue Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Drone-Based Human Detection and Persistent Tracking in Flood-Disaster Video</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="cbit-badge"><b>Sky Seekers | CBIT ECE</b> &nbsp;·&nbsp; Architecture: Recorded Video ➔ Preprocess ➔ YOLO (Person Class 0) ➔ ByteTrack ➔ Detection Log ➔ Operator Review ➔ CSV/MP4 Export</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="notice-box">
<b>System Objective:</b> Analyzes recorded aerial drone footage of flood-disaster scenes (including dense crowds with &gt;35 people), detects humans, assigns persistent tracking IDs across consecutive frames, calculates true frame pixel coordinates, and enables operator review.
</div>
""", unsafe_allow_html=True)


# =========================================================================
# SECTION 1: VIDEO INPUT & METADATA
# =========================================================================
st.subheader("1. Video Source & Ingestion")

sample_dir = os.path.join(CURRENT_DIR, "sample_data")
sample_files = []
if os.path.exists(sample_dir):
    sample_files = [f for f in os.listdir(sample_dir) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]

input_option = st.radio(
    "Select Video Source:",
    ["Upload Flood Video (MP4 / AVI / MOV)", "Use Packaged Aerial Drone Footage"],
    horizontal=True
)

selected_video_path = None
uploaded_file_name = None

if input_option == "Upload Flood Video (MP4 / AVI / MOV)":
    uploaded_file = st.file_uploader(
        "Upload aerial flood-disaster footage (MP4, AVI, MOV)",
        type=["mp4", "avi", "mov"],
        help="Accepts AI-generated or real drone-view flood disaster footage."
    )
    if uploaded_file is not None:
        temp_dir = tempfile.gettempdir()
        temp_input_path = os.path.join(temp_dir, f"signalvex_v2_{uploaded_file.name}")
        with open(temp_input_path, "wb") as f:
            f.write(uploaded_file.read())
        selected_video_path = temp_input_path
        uploaded_file_name = uploaded_file.name
else:
    if sample_files:
        selected_sample = st.selectbox("Packaged Drone Footage:", sample_files)
        selected_video_path = os.path.join(sample_dir, selected_sample)
        uploaded_file_name = selected_sample
    else:
        st.warning("No sample video found in `sample_data/`. Please upload a video above.")

# Video Metadata Display
if selected_video_path and os.path.exists(selected_video_path):
    try:
        video_info = VideoProcessor.get_video_info(selected_video_path)
        col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
        with col_m1:
            st.metric("Source Video", video_info["file_name"][:20])
        with col_m2:
            st.metric("Resolution", f"{video_info['width']} × {video_info['height']} px")
        with col_m3:
            st.metric("Frame Rate", f"{video_info['fps']} FPS")
        with col_m4:
            st.metric("Total Frames", f"{video_info['total_frames']}")
        with col_m5:
            st.metric("Video Duration", video_info["duration_formatted"])
    except Exception as e:
        st.error(f"Error reading video metadata: {e}. Please ensure the video is not corrupted.")
        selected_video_path = None


# =========================================================================
# START PROCESSING TRIGGER & LIVE WORKFLOW
# =========================================================================
st.write("")
col_btn, _ = st.columns([1, 4])
with col_btn:
    start_btn = st.button("🚀 Start Processing Video", type="primary", disabled=(selected_video_path is None))

if start_btn and selected_video_path:
    try:
        st.info("Initializing YOLO detection model and ByteTrack tracking engine...")

        # Load detector and tracker
        detector = load_detector_model(model_default_path, conf_thresh, imgsz_choice)
        tracker = HumanTracker(
            model_instance=detector.model,
            tracker_type="bytetrack.yaml",
            conf_threshold=conf_thresh,
            review_threshold=review_threshold,
            imgsz=imgsz_choice,
            track_buffer=track_buffer
        )

        # Clear and reset logger
        st.session_state.detection_logger.clear()
        processor = VideoProcessor(tracker=tracker, logger=st.session_state.detection_logger)

        # Prepare output video destination
        temp_dir = tempfile.gettempdir()
        out_video_name = f"signalvex_annotated_{int(time.time())}.mp4"
        annotated_out_path = os.path.join(temp_dir, out_video_name)

        # Live UI Elements
        st.markdown("#### ⚡ Real-Time Processing Progress")
        progress_bar = st.progress(0.0)
        status_banner = st.empty()
        live_col_img, live_col_stats = st.columns([2, 1])

        with live_col_img:
            live_feed_slot = st.empty()
        with live_col_stats:
            live_stats_slot = st.empty()

        # Real-time progress callback
        def progress_callback(ratio, frame, stats):
            progress_bar.progress(ratio)
            status_banner.markdown(
                f"**Frame {stats['frame_idx']} / {stats['total_frames']}** "
                f"({stats['timestamp_formatted']}) &nbsp;|&nbsp; "
                f"Speed: `{stats['current_fps']} FPS` &nbsp;|&nbsp; "
                f"Active Targets: `{stats['current_active_tracks']}`"
            )

            # Display annotated live frame (converted from BGR to RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            live_feed_slot.image(
                rgb_frame,
                caption=f"SIGNALVEX Live Feed — Frame {stats['frame_idx']} [{stats['timestamp_formatted']}]",
                use_container_width=True
            )

            # Live metrics card
            live_stats_slot.markdown(f"""
            <div class="sar-card">
                <div class="sar-val">{stats['current_active_tracks']}</div>
                <div class="sar-label">Active Tracks This Frame</div>
            </div>
            <div class="sar-card">
                <div class="sar-val">{stats['distinct_track_ids']}</div>
                <div class="sar-label">Distinct Track IDs Observed</div>
            </div>
            <div class="sar-card">
                <div class="sar-val">{stats['total_detections']}</div>
                <div class="sar-label">Cumulative Observations</div>
            </div>
            <div class="sar-card">
                <div class="sar-val">{stats['current_fps']} FPS</div>
                <div class="sar-label">Instantaneous Processing Speed</div>
            </div>
            """, unsafe_allow_html=True)

        frames_limit = int(max_frames_to_process) if max_frames_to_process > 0 else None

        # Execute end-to-end video processing
        summary = processor.process_video(
            video_path=selected_video_path,
            output_path=annotated_out_path,
            conf_threshold=conf_thresh,
            review_threshold=review_threshold,
            max_frames=frames_limit,
            box_thickness=box_thickness,
            font_scale=font_scale,
            progress_callback=progress_callback
        )

        st.session_state.processing_complete = True
        st.session_state.processing_summary = summary
        st.session_state.annotated_video_path = annotated_out_path
        st.session_state.active_video_name = uploaded_file_name
        st.success(f"Processing complete! Analyzed {summary['frames_processed']} frames in {summary['processing_time_sec']}s at {summary['measured_fps']} FPS.")

    except Exception as e:
        st.error(f"Video processing encountered an error: {e}")
        st.session_state.processing_complete = False


# =========================================================================
# SECTION 2: SUMMARY METRICS & DISTINCTION NOTICE
# =========================================================================
if st.session_state.processing_complete and st.session_state.processing_summary:
    summary = st.session_state.processing_summary

    # Compute live metrics from the active logger session
    metrics = st.session_state.detection_logger.get_summary_metrics()

    st.divider()
    st.subheader("2. Search-and-Rescue Detection Summary")

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.markdown(f"""
        <div class="sar-card" style="border-left-color: #38bdf8;">
            <div class="sar-val" style="color: #38bdf8;">{metrics['distinct_track_ids']}</div>
            <div class="sar-label">Unique Track IDs (1)</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="sar-card">
            <div class="sar-val">{metrics['total_observations']}</div>
            <div class="sar-label">Total Observations (2)</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="sar-card" style="border-left-color: #f59e0b;">
            <div class="sar-val" style="color: #f59e0b;">{metrics['flagged_observations']}</div>
            <div class="sar-label">Flagged Observations (3)</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="sar-card" style="border-left-color: #ec4899;">
            <div class="sar-val" style="color: #ec4899;">{metrics['unique_flagged_tracks']}</div>
            <div class="sar-label">Unique Flagged Track IDs (4)</div>
        </div>
        """, unsafe_allow_html=True)
    with m5:
        st.markdown(f"""
        <div class="sar-card" style="border-left-color: #10b981;">
            <div class="sar-val" style="color: #10b981;">{metrics['manually_reviewed_records']}</div>
            <div class="sar-label">Manually Reviewed Records (5)</div>
        </div>
        """, unsafe_allow_html=True)

    # Secondary performance and flag reason cards
    sec1, sec2, sec3, sec4 = st.columns(4)
    with sec1:
        st.caption(f"**Measured Speed:** `{summary.get('measured_fps', 0)} FPS`")
    with sec2:
        st.caption(f"**Execution Time:** `{summary.get('processing_time_sec', 0)}s`")
    with sec3:
        st.caption(f"**Operator Keep Count:** `{metrics['keep_count']}` | **Reject:** `{metrics['reject_count']}`")
    with sec4:
        st.caption(f"**Average Confidence:** `{metrics['avg_confidence'] * 100.0:.1f}%`")

    # Flag Reasons Transparency Breakdown
    flag_counts = metrics.get("flag_reason_counts", {})
    st.markdown("##### 🔍 Flagging Transparency Breakdown")
    fb1, fb2, fb3, fb4 = st.columns(4)
    with fb1:
        st.info(f"**Low Confidence:** {flag_counts.get('Low Confidence', 0)} obs")
    with fb2:
        st.warning(f"**Tracking Warning:** {flag_counts.get('Tracking Warning', 0)} obs")
    with fb3:
        st.success(f"**Manual Review:** {flag_counts.get('Manual Review', 0)} obs")
    with fb4:
        st.caption(f"**No Flag (Stable):** {flag_counts.get('No Flag', 0)} obs")

    st.markdown("""
    <div class="notice-box">
    <b>Critical Identity Notice:</b> <b>Total Observations</b> reflects the frame-by-frame person detections across all frames. 
    <b>Unique Track IDs</b> reflects the software's persistent tracking association across consecutive frames.
    Due to potential occlusions, crowd density, or drone motion, track IDs are approximate tracking trajectories and <b>not verified biological identities</b>.
    </div>
    """, unsafe_allow_html=True)


    # =========================================================================
    # =========================================================================
    # SECTION 3: DETECTION LOG TABLE, FILTERS & DIAGNOSTIC INSPECTOR
    # =========================================================================
    st.divider()
    st.subheader("3. Timestamped Detection Log & Operator Review")

    df = st.session_state.detection_logger.to_dataframe()

    if not df.empty:
        # Display Filters Explanation
        st.markdown("##### 🎛️ Observation Display Filters")
        st.caption("Adjusting display filters controls which rows are rendered below; it **does NOT modify or delete** any underlying detection records.")

        f_col1, f_col2, f_col3, f_col4 = st.columns(4)

        with f_col1:
            status_filter = st.multiselect(
                "Filter by Review Status:",
                options=["Keep", "Reject", "Needs Review"],
                default=["Keep", "Reject", "Needs Review"],
                help="Filters rows based on current operator decision or initial rule assignment."
            )
        with f_col2:
            min_conf_filter = st.slider(
                "Minimum Display Confidence (%):",
                min_value=0.0,
                max_value=100.0,
                value=float(conf_thresh * 100.0),
                step=5.0,
                help="Controls table visibility threshold. Underlying detection records remain fully preserved."
            )
        with f_col3:
            all_track_ids = sorted(list(df["track_id"].unique()))
            selected_track_filter = st.selectbox(
                "Filter by Track ID:",
                options=["All Tracks"] + [f"Track #{t}" for t in all_track_ids]
            )
        with f_col4:
            flag_rule_filter = st.selectbox(
                "Filter by Flag Recommendation:",
                options=["All Records", "Automatically Flagged Only", "Unflagged (Normal) Only"],
                help="Filter by rule recommendation independently of operator status."
            )

        # Apply filtering
        filter_mask = (
            (df["review_status"].isin(status_filter)) &
            (df["confidence_percent"] >= min_conf_filter)
        )
        if selected_track_filter != "All Tracks":
            tid = int(selected_track_filter.split("#")[1])
            filter_mask = filter_mask & (df["track_id"] == tid)

        if flag_rule_filter == "Automatically Flagged Only":
            filter_mask = filter_mask & (df["automatic_flag"] == True)
        elif flag_rule_filter == "Unflagged (Normal) Only":
            filter_mask = filter_mask & (df["automatic_flag"] == False)

        filtered_df = df[filter_mask]

        # Explicit observation count visibility banner
        total_obs = len(df)
        vis_obs = len(filtered_df)
        pct_vis = (vis_obs / total_obs * 100.0) if total_obs > 0 else 0.0

        if vis_obs == total_obs:
            st.info(f"**Showing all {total_obs:,} recorded observations (100% visible).**")
        else:
            st.warning(
                f"**Showing {vis_obs:,} of {total_obs:,} recorded observations ({pct_vis:.1f}% visible).** "
                f"*(Filters active: Min Conf ≥ {min_conf_filter}%, Status: {', '.join(status_filter)}, Flag: {flag_rule_filter})*"
            )

        # Filtered sub-metrics summary if subset is active
        if vis_obs < total_obs and vis_obs > 0:
            sub_m = st.session_state.detection_logger.get_summary_metrics(filtered_df)
            st.caption(
                f"📊 **Filtered View Metrics:** Distinct Tracks: `{sub_m['distinct_track_ids']}` | "
                f"Flagged: `{sub_m['flagged_observations']}` | "
                f"Keep: `{sub_m['keep_count']}` | Reject: `{sub_m['reject_count']}` | "
                f"Avg Conf: `{sub_m['avg_confidence']*100.0:.1f}%`"
            )

        # Display Dataframe with clean user-facing columns
        display_columns = [
            "frame_number", "timestamp_formatted", "track_id",
            "confidence_percent", "confidence", "bbox_coords",
            "bbox_width", "bbox_height", "automatic_flag", "flag_reason",
            "review_status", "manual_reviewed", "tracking_warning"
        ]
        st.dataframe(
            filtered_df[display_columns].rename(columns={
                "frame_number": "Frame #",
                "timestamp_formatted": "Timestamp",
                "track_id": "Track ID",
                "confidence_percent": "Confidence (%)",
                "confidence": "Raw Score",
                "bbox_coords": "Bounding Box [x1, y1, x2, y2]",
                "bbox_width": "Width",
                "bbox_height": "Height",
                "automatic_flag": "Auto Flagged",
                "flag_reason": "Flag Reason",
                "review_status": "Review Status",
                "manual_reviewed": "Manual Decision",
                "tracking_warning": "Warning"
            }),
            use_container_width=True,
            height=300
        )

        # Raw Confidence Diagnostic Inspector (Directly Addresses Issue 1 & Issue 3)
        with st.expander("🔬 Raw Confidence & Video-to-Log Verification Inspector", expanded=False):
            st.markdown("""
            **Diagnostic Audit Tool:** Cross-references the original YOLO model inference confidence, video overlay text, 
            log table values, and CSV export records for identical detections.
            """)
            diag_f1, diag_f2 = st.columns([3, 1])
            with diag_f1:
                diag_filter = st.radio(
                    "Inspect Confidence Range:",
                    [
                        "All Recorded Detections",
                        "High Confidence (≥ 80%) — Compare with Video Overlay",
                        "Good Confidence (≥ 70%)",
                        "Review Eligible (< 50%)",
                        "Specific Track ID"
                    ],
                    horizontal=True
                )
            with diag_f2:
                diag_tid = st.selectbox("Track ID (for inspection):", all_track_ids, format_func=lambda x: f"Track #{x}")

            diag_sub = df.copy()
            if diag_filter == "High Confidence (≥ 80%) — Compare with Video Overlay":
                diag_sub = diag_sub[diag_sub["confidence_percent"] >= 80.0]
            elif diag_filter == "Good Confidence (≥ 70%)":
                diag_sub = diag_sub[diag_sub["confidence_percent"] >= 70.0]
            elif diag_filter == "Review Eligible (< 50%)":
                diag_sub = diag_sub[diag_sub["confidence_percent"] < 50.0]
            elif diag_filter == "Specific Track ID":
                diag_sub = diag_sub[diag_sub["track_id"] == diag_tid]

            diag_display = pd.DataFrame({
                "Frame #": diag_sub["frame_number"],
                "Timestamp": diag_sub["timestamp_formatted"],
                "Track ID": diag_sub["track_id"],
                "Raw YOLO Score (0-1)": diag_sub["confidence"],
                "Video Overlay Text": diag_sub.apply(
                    lambda r: f"{'REV' if r.get('automatic_flag', False) else 'ID'} #{r['track_id']} | {r['confidence_percent']:.1f}%",
                    axis=1
                ),
                "Table Conf (%)": diag_sub["confidence_percent"],
                "CSV Export Conf": diag_sub["confidence"],
                "CSV Conf (%)": diag_sub["confidence_percent"],
                "Bounding Box [x1, y1, x2, y2]": diag_sub["bbox_coords"],
                "Auto Flag": diag_sub["automatic_flag"].map({True: "Flagged", False: "Normal"}),
                "Flag Reason": diag_sub["flag_reason"],
                "Review Status": diag_sub["review_status"]
            })

            st.dataframe(diag_display, use_container_width=True, height=280)
            st.caption(
                f"✅ **Audit Verification:** Found **{len(diag_display):,}** observations matching filter. "
                "Notice: For every detection, `Raw YOLO Score * 100` matches the Video Overlay label, "
                "the Table Confidence (%), and both CSV columns without discrepancy."
            )

        # Track Trajectory & Timeline Inspector
        with st.expander("🔍 Track Trajectory & History Inspector", expanded=False):
            if all_track_ids:
                inspect_tid = st.selectbox(
                    "Select Track ID to inspect:",
                    options=all_track_ids,
                    format_func=lambda x: f"Track #{x}"
                )
                track_sub = df[df["track_id"] == inspect_tid].sort_values("frame_number")

                ti_c1, ti_c2, ti_c3, ti_c4 = st.columns(4)
                with ti_c1:
                    st.metric("Total Observations", len(track_sub))
                with ti_c2:
                    st.metric("First Seen", track_sub["timestamp_formatted"].iloc[0])
                with ti_c3:
                    st.metric("Last Seen", track_sub["timestamp_formatted"].iloc[-1])
                with ti_c4:
                    st.metric("Average Confidence", f"{track_sub['confidence_percent'].mean():.1f}%")

                st.line_chart(track_sub.set_index("frame_number")["confidence_percent"])


        # =====================================================================
        # SECTION 4: OPERATOR REVIEW ACTION FORM
        # =====================================================================
        st.markdown("#### ✍️ Operator Review Action Panel")
        tab_track_rev, tab_record_rev = st.tabs(["Review by Track ID", "Review by Individual Observation"])

        with tab_track_rev:
            st.caption("Apply an operator review status to all observations belonging to a selected track.")
            col_act1, col_act2, col_act3 = st.columns([2, 2, 2])
            with col_act1:
                target_track_review = st.selectbox(
                    "Select Track to Review:",
                    options=all_track_ids,
                    format_func=lambda x: f"Track #{x}",
                    key="select_track_rev"
                )
            with col_act2:
                new_review_status = st.selectbox(
                    "Assign Status to All Observations of Track:",
                    ["Keep", "Reject", "Needs Review"],
                    key="select_track_status"
                )
            with col_act3:
                st.write("")
                st.write("")
                if st.button("Apply Decision to Track", type="primary", key="btn_apply_track"):
                    st.session_state.detection_logger.update_status_by_track(target_track_review, new_review_status)
                    st.success(f"Updated all observations for Track #{target_track_review} to '{new_review_status}'!")
                    st.rerun()

        with tab_record_rev:
            st.caption("Apply an operator review status to a specific observation record key.")
            all_records = df["record_id"].tolist()
            if all_records:
                col_rec1, col_rec2, col_rec3 = st.columns([2, 2, 2])
                with col_rec1:
                    target_rec = st.selectbox(
                        "Select Record ID:",
                        options=all_records,
                        format_func=lambda r: f"{r} (Frame {df.loc[df['record_id']==r, 'frame_number'].iloc[0]}, Track #{df.loc[df['record_id']==r, 'track_id'].iloc[0]})",
                        key="select_rec_rev"
                    )
                with col_rec2:
                    new_rec_status = st.selectbox(
                        "Assign Status to Selected Record:",
                        ["Keep", "Reject", "Needs Review"],
                        key="select_rec_status"
                    )
                with col_rec3:
                    st.write("")
                    st.write("")
                    if st.button("Apply Decision to Record", type="primary", key="btn_apply_record"):
                        st.session_state.detection_logger.update_record_status(target_rec, new_rec_status)
                        st.success(f"Updated Record {target_rec} to '{new_rec_status}'!")
                        st.rerun()


        # =====================================================================
        # SECTION 5: EXPORTS (CSV & ANNOTATED VIDEO)
        # =====================================================================
        st.divider()
        st.subheader("4. Search-and-Rescue Data & Video Export")

        col_exp1, col_exp2, col_exp3, col_exp4 = st.columns(4)

        with col_exp1:
            full_csv = st.session_state.detection_logger.to_csv()
            st.download_button(
                label="📥 Complete Detection Log (CSV)",
                data=full_csv,
                file_name=f"signalvex_detection_log_{summary['video_name']}.csv",
                mime="text/csv",
                help="Exports complete frame-by-frame log with original pixel coordinates, track IDs, and confidence."
            )

        with col_exp2:
            reviewed_csv = st.session_state.detection_logger.to_reviewed_csv()
            st.download_button(
                label="📥 Reviewed Detection Log (CSV)",
                data=reviewed_csv,
                file_name=f"signalvex_reviewed_log_{summary['video_name']}.csv",
                mime="text/csv",
                help="Exports detection log including latest human operator review decisions."
            )

        with col_exp3:
            track_summary_csv = st.session_state.detection_logger.to_track_summary_csv()
            st.download_button(
                label="📥 Track Summary (CSV)",
                data=track_summary_csv,
                file_name=f"signalvex_track_summary_{summary['video_name']}.csv",
                mime="text/csv",
                help="Exports one row per observed track ID with start/end timestamps, duration, and average confidence."
            )

        with col_exp4:
            ann_vid_path = st.session_state.annotated_video_path
            if ann_vid_path and os.path.exists(ann_vid_path):
                with open(ann_vid_path, "rb") as vf:
                    st.download_button(
                        label="📥 Annotated Video (MP4)",
                        data=vf.read(),
                        file_name=f"signalvex_annotated_{summary['video_name']}",
                        mime="video/mp4",
                        help="Download original video with bounding boxes, persistent IDs, and telemetry burned in."
                    )
            else:
                st.info("Annotated video file is unavailable.")

    else:
        st.warning("No human detections were recorded in this video above the confidence threshold.")


# =========================================================================
# SECTION 6: ACADEMIC TRANSPARENCY & METHODOLOGY
# =========================================================================
st.divider()
with st.expander("📖 Academic Reference, Architecture & Methodology (Sky Seekers | CBIT ECE)", expanded=False):
    st.markdown("""
    ### SIGNALVEX Search-and-Rescue Prototype Overview

    **Sky Seekers Research Team | Chaitanya Bharathi Institute of Technology (CBIT)**
    - **Department:** Electronics and Communication Engineering
    - **Supervision & Presentation Reference:** SIGNALVEX — Drone-Based Human Detection and Rescue Mapping

    #### 1. Core Technical Pipeline
    - **Frame Ingestion:** Incremental OpenCV frame decoding preserving true pixel coordinates `[x1, y1, x2, y2]`.
    - **Human Detection:** Pretrained Ultralytics YOLO filtered exclusively for COCO Class 0 (`person`).
    - **Multi-Object Tracking:** ByteTrack multi-stage association linking bounding boxes to persistent track IDs across successive frames.
    - **Heuristic Warnings:** Flags potential ID switches (spatial/velocity jumps), reappearing tracks across frame gaps, and low-confidence candidates.
    - **Operator-in-the-Loop:** Decision-support ground dashboard allowing operators to Keep, Reject, or mark candidates for further review.

    #### 2. Explicit Scope Limitations
    1. **Visual RGB Only:** Operates strictly on visible light aerial video. Does not penetrate rubble or smoke.
    2. **Software Track IDs:** Tracking IDs are software trajectory associations. They do not represent verified biological individuals.
    3. **No Real GPS / Telemetry:** No hardware GPS, live drone telemetry, pixel-to-ground orthorectification, or thermal/acoustic sensors are claimed or implemented in this software prototype.
    4. **Life Safety Notice:** This application is an academic college project prototype designed for human decision support, not an automated life-safety system.
    """)
