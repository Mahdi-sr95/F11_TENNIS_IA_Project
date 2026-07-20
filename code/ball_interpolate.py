#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ball Track Interpolation (Final - High Quality / Fail-Fast)

Reads (fixed absolute path):
- C:\\Users\\Ost\\Desktop\\F11_TENNIS\\ball_tracks.csv

Uses:
- C:\\Users\\Ost\\Desktop\\F11_TENNIS\\court_points.json  (for image_size bounds)

Writes:
- C:\\Users\\Ost\\Desktop\\F11_TENNIS\\ball_tracks_interpolated.csv
- C:\\Users\\Ost\\Desktop\\F11_TENNIS\\interpolation_comparison.jpg

Main quality rules:
1) No back-fill before first detection and no forward-fill after last detection.
2) KF points are used only if inside image bounds.
3) Final source priority: det > kf (short gaps) > interp (shorter gaps) > none
4) Adds `source` column: det / kf / interp / none
"""

import json
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(r"C:\Users\Ost\Desktop\F11_TENNIS")

INPUT_CSV = Path(r"C:\Users\Ost\Desktop\F11_TENNIS\ball_tracks.csv")
COURT_POINTS_JSON = Path(r"C:\Users\Ost\Desktop\F11_TENNIS\court_points.json")

OUTPUT_CSV = PROJECT_ROOT / "ball_tracks_interpolated.csv"
OUTPUT_PLOT = PROJECT_ROOT / "interpolation_comparison.jpg"


def read_court_image_size(court_points_path: Path) -> Tuple[int, int]:
    if not court_points_path.exists():
        raise FileNotFoundError("court_points.json not found: {}".format(court_points_path))
    data = json.loads(court_points_path.read_text(encoding="utf-8"))
    img_size = data.get("image_size", None)
    if not isinstance(img_size, list) or len(img_size) != 2:
        raise ValueError("court_points.json must contain image_size=[w,h].")
    return int(img_size[0]), int(img_size[1])


class BallTrackInterpolator(object):
    def __init__(self, input_csv: Path, frame_size: Tuple[int, int]):
        self.input_csv = Path(input_csv)
        if not self.input_csv.exists():
            raise FileNotFoundError("Input CSV not found: {}".format(self.input_csv))

        self.frame_width = int(frame_size[0])
        self.frame_height = int(frame_size[1])

        self.df = pd.read_csv(self.input_csv)
        self._normalize_columns_inplace()

    def _normalize_columns_inplace(self):
        required = ["frame", "detected", "det_x", "det_y", "kf_x", "kf_y", "confidence"]
        for c in required:
            if c not in self.df.columns:
                # For strict quality, fail fast because your input format is known.
                raise ValueError("ball_tracks.csv missing column: {}".format(c))

        self.df["frame"] = pd.to_numeric(self.df["frame"], errors="raise").astype(int)
        self.df["detected"] = self.df["detected"].astype(bool)

        for c in ["det_x", "det_y", "kf_x", "kf_y", "confidence"]:
            self.df[c] = pd.to_numeric(self.df[c], errors="coerce")

        self.df = self.df.sort_values("frame").reset_index(drop=True)

    def _in_bounds(self, x: float, y: float) -> bool:
        if np.isnan(x) or np.isnan(y):
            return False
        return (0 <= x < self.frame_width) and (0 <= y < self.frame_height)

    def interpolate_inside_only(self):
        # Only interpolate between detections (no back/forward fill)
        self.df["interp_x"] = self.df["det_x"].copy().interpolate(method="linear", limit_area="inside")
        self.df["interp_y"] = self.df["det_y"].copy().interpolate(method="linear", limit_area="inside")

    def smooth(self, window_size: int = 5):
        self.df["smooth_x"] = self.df["interp_x"].rolling(window=window_size, center=True, min_periods=1).mean()
        self.df["smooth_y"] = self.df["interp_y"].rolling(window=window_size, center=True, min_periods=1).mean()

    def _gap_len_from_last_detection(self) -> pd.Series:
        missing = self.df["det_x"].isna() | self.df["det_y"].isna()
        out = np.zeros(len(self.df), dtype=int)
        cur = 0
        for i in range(len(self.df)):
            if bool(missing.iloc[i]):
                cur += 1
            else:
                cur = 0
            out[i] = cur
        return pd.Series(out, index=self.df.index)

    def combine_sources(self, max_kf_gap: int = 30, max_interp_gap: int = 80):
        gap_len = self._gap_len_from_last_detection()

        det_valid = self.df["det_x"].notna() & self.df["det_y"].notna()
        kf_present = self.df["kf_x"].notna() & self.df["kf_y"].notna()
        interp_present = self.df["smooth_x"].notna() & self.df["smooth_y"].notna()

        # Bounds checks
        kf_valid = kf_present & self.df.apply(lambda r: self._in_bounds(r["kf_x"], r["kf_y"]), axis=1)
        interp_valid = interp_present & self.df.apply(lambda r: self._in_bounds(r["smooth_x"], r["smooth_y"]), axis=1)

        self.df["final_x"] = np.nan
        self.df["final_y"] = np.nan
        self.df["source"] = "none"

        # 1) detections
        self.df.loc[det_valid, "final_x"] = self.df.loc[det_valid, "det_x"]
        self.df.loc[det_valid, "final_y"] = self.df.loc[det_valid, "det_y"]
        self.df.loc[det_valid, "source"] = "det"

        # 2) KF (short gaps only)
        no_det = ~det_valid
        use_kf = no_det & kf_valid & (gap_len <= int(max_kf_gap))
        self.df.loc[use_kf, "final_x"] = self.df.loc[use_kf, "kf_x"]
        self.df.loc[use_kf, "final_y"] = self.df.loc[use_kf, "kf_y"]
        self.df.loc[use_kf, "source"] = "kf"

        # 3) Interp (reasonable gaps only)
        still_missing = self.df["final_x"].isna() | self.df["final_y"].isna()
        use_interp = still_missing & interp_valid & (gap_len <= int(max_interp_gap))
        self.df.loc[use_interp, "final_x"] = self.df.loc[use_interp, "smooth_x"]
        self.df.loc[use_interp, "final_y"] = self.df.loc[use_interp, "smooth_y"]
        self.df.loc[use_interp, "source"] = "interp"

    def save(self, output_csv: Path):
        cols = [
            "frame", "detected",
            "det_x", "det_y",
            "kf_x", "kf_y",
            "interp_x", "interp_y",
            "smooth_x", "smooth_y",
            "final_x", "final_y",
            "source",
            "confidence",
        ]
        self.df[cols].to_csv(output_csv, index=False)

    def plot(self, output_plot: Path, sample_frames: int = 400):
        df_s = self.df.head(int(sample_frames)) if sample_frames else self.df

        fig, axes = plt.subplots(2, 1, figsize=(15, 10))

        axes[0].plot(df_s["frame"], df_s["det_x"], "o", markersize=3, alpha=0.6, label="Detection")
        axes[0].plot(df_s["frame"], df_s["kf_x"], "s", markersize=2, alpha=0.4, label="Kalman")
        axes[0].plot(df_s["frame"], df_s["smooth_x"], "-", linewidth=2, alpha=0.7, label="Smoothed(interp)")
        axes[0].plot(df_s["frame"], df_s["final_x"], "-", linewidth=2, alpha=0.9, color="red", label="Final")
        axes[0].set_title("X coordinate")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        axes[1].plot(df_s["frame"], df_s["det_y"], "o", markersize=3, alpha=0.6, label="Detection")
        axes[1].plot(df_s["frame"], df_s["kf_y"], "s", markersize=2, alpha=0.4, label="Kalman")
        axes[1].plot(df_s["frame"], df_s["smooth_y"], "-", linewidth=2, alpha=0.7, label="Smoothed(interp)")
        axes[1].plot(df_s["frame"], df_s["final_y"], "-", linewidth=2, alpha=0.9, color="red", label="Final")
        axes[1].set_title("Y coordinate (image)")
        axes[1].invert_yaxis()
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(output_plot, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def run(self, output_csv: Path, output_plot: Optional[Path] = None):
        self.interpolate_inside_only()
        self.smooth(window_size=5)
        self.combine_sources(max_kf_gap=30, max_interp_gap=80)
        self.save(output_csv)
        if output_plot is not None:
            self.plot(output_plot, sample_frames=400)


def main():
    frame_size = read_court_image_size(COURT_POINTS_JSON)

    interpolator = BallTrackInterpolator(
        input_csv=INPUT_CSV,
        frame_size=frame_size
    )

    interpolator.run(
        output_csv=OUTPUT_CSV,
        output_plot=OUTPUT_PLOT
    )

    print("✓ Wrote: {}".format(OUTPUT_CSV))
    print("✓ Wrote: {}".format(OUTPUT_PLOT))


if __name__ == "__main__":
    main()
