"""
logger.py — Timestamped Detection Logging & CSV Export Module for SIGNALVEX_V2
==============================================================================
Drone-Based Human Detection and Rescue Mapping Prototype
Department of Electronics and Communication Engineering
Chaitanya Bharathi Institute of Technology | Sky Seekers Team

Features:
- Maintains structured Pandas-based detection tables for all video observations.
- Strict schema compliance:
    * video_name, frame_number, timestamp_seconds, timestamp_formatted
    * track_id, confidence, confidence_percent
    * x1, y1, x2, y2, bbox_width, bbox_height
    * review_status (Keep, Reject, Needs Review)
    * tracking_warning, record_id
- Human Operator Review persistence (per-record or per-track)
- Export utilities:
    1. Complete detection log CSV
    2. Reviewed detection log CSV
    3. Approximate Track Summary CSV (one row per distinct track ID)
"""

from typing import List, Dict, Any, Optional, Set
import pandas as pd
import numpy as np


class DetectionLogger:
    """
    Maintains timestamped person detection logs, track history, operator review decisions,
    and structured CSV exports for search-and-rescue analysis.
    """

    STATUS_KEEP = "Keep"
    STATUS_REJECT = "Reject"
    STATUS_NEEDS_REVIEW = "Needs Review"

    def __init__(self):
        self.records: List[Dict[str, Any]] = []
        self._record_keys: Set[str] = set()

    def clear(self):
        """Clears all logged detection records for a fresh processing run."""
        self.records.clear()
        self._record_keys.clear()

    def log_detection(
        self,
        video_name: str,
        frame_number: int,
        timestamp_seconds: float,
        timestamp_formatted: str,
        track_id: int,
        confidence: float,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        bbox_width: int,
        bbox_height: int,
        automatic_flag: bool = False,
        review_status: str = "Keep",
        tracking_warning: str = "None",
        flag_reason: str = "No Flag",
        det_idx: int = 0
    ) -> Optional[Dict[str, Any]]:
        """
        Logs a single person detection observation under its persistent track ID.
        Uses original frame coordinates and real model confidence.
        Deduplicates by stable observation key (video_name + frame_number + track_id + det_idx).
        """
        obs_key = f"{video_name}_{frame_number:05d}_{track_id:04d}_{det_idx:02d}"
        if obs_key in self._record_keys:
            # Prevent duplicate observation entry for the same frame, track ID, and detection index
            return None

        self._record_keys.add(obs_key)
        record_id = f"rec_{obs_key}_{len(self.records)}"

        entry = {
            "record_id": record_id,
            "obs_key": obs_key,
            "video_name": video_name,
            "frame_number": int(frame_number),
            "timestamp_seconds": round(float(timestamp_seconds), 3),
            "timestamp_formatted": str(timestamp_formatted),
            "track_id": int(track_id),
            "confidence": round(float(confidence), 4),
            "confidence_percent": round(float(confidence) * 100.0, 1),
            "x1": int(x1),
            "y1": int(y1),
            "x2": int(x2),
            "y2": int(y2),
            "bbox_width": int(bbox_width),
            "bbox_height": int(bbox_height),
            "bbox_coords": f"[{x1}, {y1}, {x2}, {y2}]",
            "automatic_flag": bool(automatic_flag),
            "flag_reason": flag_reason,
            "review_status": review_status,
            "tracking_warning": tracking_warning,
            "manual_reviewed": False,
            "is_manual_review": False
        }
        self.records.append(entry)
        return entry

    def update_record_status(self, record_id: str, new_status: str) -> bool:
        """Update review decision for an individual observation row."""
        for rec in self.records:
            if rec["record_id"] == record_id:
                rec["review_status"] = new_status
                rec["manual_reviewed"] = True
                rec["is_manual_review"] = True
                if "Manual Review" not in rec.get("flag_reason", ""):
                    prev_reason = rec.get("flag_reason", "No Flag")
                    if prev_reason == "No Flag":
                        rec["flag_reason"] = "Manual Review"
                    else:
                        rec["flag_reason"] = f"{prev_reason}; Manual Review"
                return True
        return False

    def update_status_by_track(self, track_id: int, new_status: str):
        """Update review decision across all observations belonging to a track ID."""
        for rec in self.records:
            if rec["track_id"] == track_id:
                rec["review_status"] = new_status
                rec["manual_reviewed"] = True
                rec["is_manual_review"] = True
                if "Manual Review" not in rec.get("flag_reason", ""):
                    prev_reason = rec.get("flag_reason", "No Flag")
                    if prev_reason == "No Flag":
                        rec["flag_reason"] = "Manual Review"
                    else:
                        rec["flag_reason"] = f"{prev_reason}; Manual Review"

    def to_dataframe(self) -> pd.DataFrame:
        """Returns the full detection log as a pandas DataFrame."""
        if not self.records:
            return pd.DataFrame(columns=[
                "record_id", "obs_key", "video_name", "frame_number", "timestamp_seconds",
                "timestamp_formatted", "track_id", "confidence", "confidence_percent",
                "x1", "y1", "x2", "y2", "bbox_width", "bbox_height",
                "bbox_coords", "automatic_flag", "flag_reason", "review_status",
                "manual_reviewed", "tracking_warning", "is_manual_review"
            ])
        return pd.DataFrame(self.records)

    def get_track_summary_dataframe(self) -> pd.DataFrame:
        """
        Aggregates observations into a Track Summary table:
        One row per distinct track ID with start/end timing, duration, observation count,
        confidence statistics, review status, and flagged status.
        """
        df = self.to_dataframe()
        if df.empty:
            return pd.DataFrame(columns=[
                "track_id", "first_frame", "last_frame", "first_timestamp",
                "last_timestamp", "duration_seconds", "total_observations",
                "avg_confidence_percent", "min_confidence_percent", "max_confidence_percent",
                "review_status", "is_track_flagged", "warnings_count", "tracking_note"
            ])

        grouped = df.groupby("track_id")
        summary_rows = []

        for tid, group in grouped:
            first_frame = int(group["frame_number"].min())
            last_frame = int(group["frame_number"].max())
            first_time = group["timestamp_seconds"].min()
            last_time = group["timestamp_seconds"].max()
            first_time_fmt = group.loc[group["timestamp_seconds"].idxmin()]["timestamp_formatted"]
            last_time_fmt = group.loc[group["timestamp_seconds"].idxmax()]["timestamp_formatted"]
            duration = round(last_time - first_time, 2)
            obs_count = len(group)
            avg_conf = round(group["confidence_percent"].mean(), 1)
            min_conf = round(group["confidence_percent"].min(), 1)
            max_conf = round(group["confidence_percent"].max(), 1)

            # Dominant review status
            statuses = group["review_status"].value_counts()
            dominant_status = statuses.index[0] if not statuses.empty else self.STATUS_KEEP

            # Track flagged if any observation needs review
            is_flagged = (group["review_status"] == self.STATUS_NEEDS_REVIEW).any()

            warnings = group[group["tracking_warning"] != "None"]
            warn_count = len(warnings)

            summary_rows.append({
                "track_id": int(tid),
                "first_frame": first_frame,
                "last_frame": last_frame,
                "first_timestamp": first_time_fmt,
                "last_timestamp": last_time_fmt,
                "duration_seconds": duration,
                "total_observations": obs_count,
                "avg_confidence_percent": avg_conf,
                "min_confidence_percent": min_conf,
                "max_confidence_percent": max_conf,
                "review_status": dominant_status,
                "is_track_flagged": is_flagged,
                "warnings_count": warn_count,
                "tracking_note": "Approximate track estimate — not verified unique biological survivors"
            })

        summary_df = pd.DataFrame(summary_rows).sort_values("track_id")
        return summary_df

    def to_csv(self, file_path: Optional[str] = None) -> str:
        """Exports the complete detection log to CSV with both raw and percentage confidence."""
        df = self.to_dataframe()
        cols = [
            "video_name", "frame_number", "timestamp_seconds", "timestamp_formatted",
            "track_id", "confidence", "confidence_percent", "x1", "y1", "x2", "y2",
            "bbox_width", "bbox_height", "automatic_flag", "flag_reason",
            "review_status", "manual_reviewed", "tracking_warning"
        ]
        export_df = df[cols] if not df.empty else pd.DataFrame(columns=cols)
        if file_path:
            export_df.to_csv(file_path, index=False)
        return export_df.to_csv(index=False)

    def to_reviewed_csv(self, file_path: Optional[str] = None) -> str:
        """Exports the reviewed detection log containing user decisions."""
        df = self.to_dataframe()
        cols = [
            "video_name", "frame_number", "timestamp_seconds", "timestamp_formatted",
            "track_id", "confidence", "confidence_percent", "x1", "y1", "x2", "y2",
            "bbox_width", "bbox_height", "automatic_flag", "flag_reason",
            "review_status", "manual_reviewed", "tracking_warning"
        ]
        export_df = df[cols] if not df.empty else pd.DataFrame(columns=cols)
        if file_path:
            export_df.to_csv(file_path, index=False)
        return export_df.to_csv(index=False)

    def to_track_summary_csv(self, file_path: Optional[str] = None) -> str:
        """Exports track summaries (one row per track) to CSV."""
        summary_df = self.get_track_summary_dataframe()
        if file_path:
            summary_df.to_csv(file_path, index=False)
        return summary_df.to_csv(index=False)

    def get_summary_metrics(self, df_override: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Calculates aggregate metrics distinguishing total frame observations
        from distinct tracking IDs, flagged counts, and manual decisions.
        Supports df_override to compute filtered metrics when requested.
        """
        df = df_override if df_override is not None else self.to_dataframe()
        if df.empty:
            return {
                "total_observations": 0,
                "distinct_track_ids": 0,
                "flagged_observations": 0,
                "unique_flagged_tracks": 0,
                "manually_reviewed_records": 0,
                "needs_review_count": 0,
                "keep_count": 0,
                "reject_count": 0,
                "avg_confidence": 0.0,
                "total_warnings": 0,
                "flag_reason_counts": {
                    "Low Confidence": 0,
                    "Tracking Warning": 0,
                    "Manual Review": 0,
                    "No Flag": 0
                }
            }

        distinct_tracks = int(df["track_id"].nunique())
        needs_review = int(len(df[df["review_status"] == self.STATUS_NEEDS_REVIEW]))
        keep = int(len(df[df["review_status"] == self.STATUS_KEEP]))
        reject = int(len(df[df["review_status"] == self.STATUS_REJECT]))
        avg_conf = float(df["confidence"].mean())
        total_warns = int(len(df[df["tracking_warning"] != "None"]))

        # Flagged unique track IDs (tracks that have at least 1 observation flagged or needing review)
        if "automatic_flag" in df.columns:
            flagged_mask = (df["automatic_flag"] == True) | (df["review_status"] == self.STATUS_NEEDS_REVIEW)
            flagged_obs_count = int(flagged_mask.sum())
            flagged_track_ids = df[flagged_mask]["track_id"].unique()
        else:
            flagged_obs_count = needs_review
            flagged_track_ids = df[df["review_status"] == self.STATUS_NEEDS_REVIEW]["track_id"].unique()
        unique_flagged_tracks = int(len(flagged_track_ids))

        # Manually reviewed records count
        if "manual_reviewed" in df.columns:
            manual_reviewed_count = int(df["manual_reviewed"].sum())
        elif "is_manual_review" in df.columns:
            manual_reviewed_count = int(df["is_manual_review"].sum())
        else:
            manual_reviewed_count = 0

        # Calculate breakdown for each supported flag reason (without double counting the total)
        reason_col = df["flag_reason"] if "flag_reason" in df.columns else pd.Series(["No Flag"] * len(df))
        low_conf_count = int(reason_col.str.contains("Low Confidence", regex=False).sum())
        track_warn_count = int(reason_col.str.contains("Tracking Warning", regex=False).sum())
        manual_rev_count = int(reason_col.str.contains("Manual Review", regex=False).sum())
        no_flag_count = int((reason_col == "No Flag").sum())

        return {
            "total_observations": len(df),
            "distinct_track_ids": distinct_tracks,
            "flagged_observations": flagged_obs_count,
            "unique_flagged_tracks": unique_flagged_tracks,
            "manually_reviewed_records": manual_reviewed_count,
            "needs_review_count": needs_review,
            "keep_count": keep,
            "reject_count": reject,
            "avg_confidence": round(avg_conf, 4),
            "total_warnings": total_warns,
            "flag_reason_counts": {
                "Low Confidence": low_conf_count,
                "Tracking Warning": track_warn_count,
                "Manual Review": manual_rev_count,
                "No Flag": no_flag_count
            }
        }
