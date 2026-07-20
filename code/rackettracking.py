import argparse
import csv
import os
import sys
from dataclasses import dataclass
from typing import Optional, List, Tuple

import cv2
import numpy as np
from scipy import interpolate
from scipy.signal import savgol_filter


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

DEFAULT_VIDEO_PATH = os.path.join(PROJECT_ROOT, "data", "clips", "input_video.mp4")
DEFAULT_KEYPOINTS_CSV = os.path.join(PROJECT_ROOT, "results", "racketpose", "racket_keypoints_2d.csv")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "racketpose")
DEFAULT_OUT_CSV = os.path.join(DEFAULT_OUTPUT_DIR, "racket_trajectory_smooth.csv")
DEFAULT_OUT_REPORT = os.path.join(DEFAULT_OUTPUT_DIR, "tracking_quality_report.txt")
DEFAULT_OUT_DEBUG_VIDEO = os.path.join(DEFAULT_OUTPUT_DIR, "overlay_racket_tracking.mp4")


@dataclass
class TrajectoryPoint:
    frame_index: int
    center_x: Optional[float]
    center_y: Optional[float]
    angle_deg: Optional[float]
    handle_x: Optional[float]
    handle_y: Optional[float]
    head_x: Optional[float]
    head_y: Optional[float]
    smoothed: bool
    interpolated: bool
    confidence: float


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _file_must_exist(path: str, what: str) -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{what} not found: {path}")


def _read_keypoints_csv(path: str) -> List[Tuple[int, int, Optional[Tuple]]]:
    keypoints = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame_idx = int(row["frame"])
            detected = int(row["detected"])
            if detected == 1:
                cx = float(row["center_x"])
                cy = float(row["center_y"])
                angle = float(row["angle_deg"])
                hx = float(row["handle_x"])
                hy = float(row["handle_y"])
                hdx = float(row["head_x"])
                hdy = float(row["head_y"])
                quality = float(row["quality"])
                keypoints.append((frame_idx, detected, (cx, cy, angle, hx, hy, hdx, hdy, quality)))
            else:
                keypoints.append((frame_idx, detected, None))
    return keypoints


def _interpolate_gaps(data: np.ndarray, max_gap: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interpolate short gaps in 1D array.
    Returns: (interpolated_data, is_interpolated_mask)
    """
    result = data.copy()
    is_interp = np.zeros(len(data), dtype=bool)
    
    valid_mask = ~np.isnan(data)
    if valid_mask.sum() < 2:
        return result, is_interp
    
    valid_indices = np.where(valid_mask)[0]
    
    for i in range(len(valid_indices) - 1):
        start_idx = valid_indices[i]
        end_idx = valid_indices[i + 1]
        gap_size = end_idx - start_idx - 1
        
        if gap_size > 0 and gap_size <= max_gap:
            start_val = data[start_idx]
            end_val = data[end_idx]
            for j in range(1, gap_size + 1):
                alpha = j / (gap_size + 1)
                result[start_idx + j] = start_val * (1 - alpha) + end_val * alpha
                is_interp[start_idx + j] = True
    
    return result, is_interp


def _smooth_savgol(data: np.ndarray, window_length: int = 11, polyorder: int = 2) -> np.ndarray:
    """
    Apply Savitzky-Golay filter to smooth data.
    """
    valid_mask = ~np.isnan(data)
    if valid_mask.sum() < window_length:
        return data
    
    if window_length % 2 == 0:
        window_length += 1
    
    window_length = min(window_length, valid_mask.sum())
    if window_length < polyorder + 2:
        return data
    
    result = data.copy()
    valid_indices = np.where(valid_mask)[0]
    valid_data = data[valid_mask]
    
    try:
        smoothed = savgol_filter(valid_data, window_length, polyorder, mode='nearest')
        result[valid_mask] = smoothed
    except:
        pass
    
    return result


def _process_trajectory(keypoints_list: List[Tuple], max_gap: int = 10, smooth_window: int = 11) -> List[TrajectoryPoint]:
    """
    Process raw keypoints: interpolate gaps, smooth trajectory.
    """
    n_frames = len(keypoints_list)
    
    cx_arr = np.full(n_frames, np.nan)
    cy_arr = np.full(n_frames, np.nan)
    angle_arr = np.full(n_frames, np.nan)
    hx_arr = np.full(n_frames, np.nan)
    hy_arr = np.full(n_frames, np.nan)
    hdx_arr = np.full(n_frames, np.nan)
    hdy_arr = np.full(n_frames, np.nan)
    quality_arr = np.full(n_frames, np.nan)
    
    for i, (frame_idx, detected, kp_data) in enumerate(keypoints_list):
        if detected == 1 and kp_data is not None:
            cx, cy, angle, hx, hy, hdx, hdy, quality = kp_data
            cx_arr[i] = cx
            cy_arr[i] = cy
            angle_arr[i] = angle
            hx_arr[i] = hx
            hy_arr[i] = hy
            hdx_arr[i] = hdx
            hdy_arr[i] = hdy
            quality_arr[i] = quality
    
    print(f"Processing trajectory: {np.sum(~np.isnan(cx_arr))} valid points")
    
    # Interpolate gaps
    cx_interp, cx_is_interp = _interpolate_gaps(cx_arr, max_gap)
    cy_interp, cy_is_interp = _interpolate_gaps(cy_arr, max_gap)
    angle_interp, angle_is_interp = _interpolate_gaps(angle_arr, max_gap)
    hx_interp, hx_is_interp = _interpolate_gaps(hx_arr, max_gap)
    hy_interp, hy_is_interp = _interpolate_gaps(hy_arr, max_gap)
    hdx_interp, hdx_is_interp = _interpolate_gaps(hdx_arr, max_gap)
    hdy_interp, hdy_is_interp = _interpolate_gaps(hdy_arr, max_gap)
    
    is_interp = cx_is_interp | cy_is_interp
    
    print(f"After interpolation: {np.sum(~np.isnan(cx_interp))} valid points")
    
    # Smooth trajectory
    cx_smooth = _smooth_savgol(cx_interp, smooth_window, 2)
    cy_smooth = _smooth_savgol(cy_interp, smooth_window, 2)
    angle_smooth = _smooth_savgol(angle_interp, smooth_window, 2)
    hx_smooth = _smooth_savgol(hx_interp, smooth_window, 2)
    hy_smooth = _smooth_savgol(hy_interp, smooth_window, 2)
    hdx_smooth = _smooth_savgol(hdx_interp, smooth_window, 2)
    hdy_smooth = _smooth_savgol(hdy_interp, smooth_window, 2)
    
    # Build result
    results = []
    for i in range(n_frames):
        if not np.isnan(cx_smooth[i]):
            confidence = 1.0 if not is_interp[i] else 0.5
            results.append(TrajectoryPoint(
                frame_index=i,
                center_x=float(cx_smooth[i]),
                center_y=float(cy_smooth[i]),
                angle_deg=float(angle_smooth[i]),
                handle_x=float(hx_smooth[i]),
                handle_y=float(hy_smooth[i]),
                head_x=float(hdx_smooth[i]),
                head_y=float(hdy_smooth[i]),
                smoothed=True,
                interpolated=bool(is_interp[i]),
                confidence=confidence,
            ))
        else:
            results.append(TrajectoryPoint(
                frame_index=i,
                center_x=None,
                center_y=None,
                angle_deg=None,
                handle_x=None,
                handle_y=None,
                head_x=None,
                head_y=None,
                smoothed=False,
                interpolated=False,
                confidence=0.0,
            ))
    
    return results


def _write_trajectory_csv(path: str, rows: List[TrajectoryPoint]) -> None:
    _ensure_parent_dir(path)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame", "center_x", "center_y", "angle_deg", "handle_x", "handle_y", "head_x", "head_y", "smoothed", "interpolated", "confidence"])
        for r in rows:
            w.writerow([
                r.frame_index,
                r.center_x,
                r.center_y,
                r.angle_deg,
                r.handle_x,
                r.handle_y,
                r.head_x,
                r.head_y,
                int(r.smoothed),
                int(r.interpolated),
                r.confidence,
            ])


def _write_report(
    path: str,
    total_frames: int,
    raw_valid: int,
    smooth_valid: int,
    interpolated_count: int,
    coverage: float,
    pass_threshold: float,
) -> None:
    _ensure_parent_dir(path)
    passed = coverage >= pass_threshold
    with open(path, "w", encoding="utf-8") as f:
        f.write("=== Racket Tracking & Smoothing - Quality Report ===\n\n")
        f.write(f"Total frames: {total_frames}\n")
        f.write(f"Raw valid points: {raw_valid}\n")
        f.write(f"After interpolation: {smooth_valid}\n")
        f.write(f"Interpolated frames: {interpolated_count}\n")
        f.write(f"Coverage: {coverage:.4f} ({coverage*100:.2f}%)\n")
        f.write(f"Pass threshold: {pass_threshold:.4f} ({pass_threshold*100:.2f}%)\n\n")
        f.write(f"PASSED: {passed}\n")


def _open_video(video_path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    return cap


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Smooth racket trajectory and interpolate gaps (Phase 9.3)."
    )

    parser.add_argument("--video", default=DEFAULT_VIDEO_PATH, help="Input video path.")
    parser.add_argument("--keypoints-csv", default=DEFAULT_KEYPOINTS_CSV, help="Input keypoints CSV from phase 9.2.")
    parser.add_argument("--out-csv", default=DEFAULT_OUT_CSV, help="Output smooth trajectory CSV path.")
    parser.add_argument("--out-report", default=DEFAULT_OUT_REPORT, help="Output report path.")
    parser.add_argument("--no-debug-video", action="store_true", help="Disable writing debug video.")
    parser.add_argument("--out-debug-video", default=DEFAULT_OUT_DEBUG_VIDEO, help="Debug video output path.")
    parser.add_argument("--max-gap", type=int, default=10, help="Maximum gap size (frames) to interpolate.")
    parser.add_argument("--smooth-window", type=int, default=11, help="Smoothing window length (must be odd).")
    parser.add_argument("--pass-threshold", type=float, default=0.70, help="Minimum acceptable coverage rate.")
    parser.add_argument("--progress-interval", type=int, default=100, help="Print progress every N frames.")

    args = parser.parse_args()

    _file_must_exist(args.video, "Input video")
    _file_must_exist(args.keypoints_csv, "Keypoints CSV")

    print(f"\n{'='*60}")
    print("RACKET TRACKING & SMOOTHING - PHASE 9.3")
    print(f"{'='*60}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Video: {args.video}")
    print(f"Keypoints CSV: {args.keypoints_csv}")
    print(f"Max gap for interpolation: {args.max_gap} frames")
    print(f"Smoothing window: {args.smooth_window}")
    print(f"Pass threshold: {args.pass_threshold:.2%}")
    print(f"{'='*60}\n")

    keypoints_list = _read_keypoints_csv(args.keypoints_csv)
    total_frames = len(keypoints_list)
    raw_valid = sum(1 for _, det, _ in keypoints_list if det == 1)

    print(f"Loaded {total_frames} frames, {raw_valid} raw valid keypoints.\n")

    trajectory = _process_trajectory(keypoints_list, args.max_gap, args.smooth_window)

    smooth_valid = sum(1 for t in trajectory if t.center_x is not None)
    interpolated_count = sum(1 for t in trajectory if t.interpolated)
    coverage = smooth_valid / total_frames if total_frames > 0 else 0.0

    print(f"\nTrajectory processing complete:")
    print(f"  Valid points after smoothing: {smooth_valid}/{total_frames}")
    print(f"  Interpolated frames: {interpolated_count}")
    print(f"  Coverage: {coverage:.2%}\n")

    _write_trajectory_csv(args.out_csv, trajectory)
    _write_report(args.out_report, total_frames, raw_valid, smooth_valid, interpolated_count, coverage, args.pass_threshold)

    # Generate debug video
    cap = _open_video(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 30.0

    writer = None
    if not args.no_debug_video:
        out_path = str(args.out_debug_video).strip()
        _ensure_parent_dir(out_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open VideoWriter for: {out_path}")
        print(f"Debug video will be saved to: {out_path}\n")

    print("Rendering debug video...")
    frame_idx = 0
    for traj_point in trajectory:
        ret, frame = cap.read()
        if not ret:
            break

        vis = frame.copy()

        if traj_point.center_x is not None:
            cx = int(traj_point.center_x)
            cy = int(traj_point.center_y)
            hx = int(traj_point.handle_x)
            hy = int(traj_point.handle_y)
            hdx = int(traj_point.head_x)
            hdy = int(traj_point.head_y)

            if traj_point.interpolated:
                line_color = (0, 165, 255)
                status_text = "INTERPOLATED"
                status_color = (0, 165, 255)
            else:
                line_color = (0, 255, 0)
                status_text = "TRACKED"
                status_color = (0, 255, 0)

            cv2.line(vis, (hx, hy), (hdx, hdy), line_color, 3)
            cv2.circle(vis, (cx, cy), 6, (255, 0, 0), -1)
            cv2.circle(vis, (hx, hy), 5, (0, 255, 0), -1)
            cv2.circle(vis, (hdx, hdy), 5, (0, 0, 255), -1)

            label = f"Angle: {traj_point.angle_deg:.1f} | Conf: {traj_point.confidence:.2f}"
            cv2.putText(vis, label, (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            status_text = "NO TRACK"
            status_color = (0, 0, 255)

        cv2.putText(vis, f"Frame: {frame_idx}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, f"Status: {status_text}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2, cv2.LINE_AA)

        if writer is not None:
            writer.write(vis)

        if args.progress_interval > 0 and (frame_idx + 1) % args.progress_interval == 0:
            print(f"  Frame {frame_idx + 1}/{total_frames}")

        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()

    print(f"\n{'='*60}")
    print("TRACKING & SMOOTHING COMPLETE")
    print(f"{'='*60}")
    print(f"Total frames: {total_frames}")
    print(f"Raw valid keypoints: {raw_valid}")
    print(f"After smoothing & interpolation: {smooth_valid}")
    print(f"Interpolated frames: {interpolated_count}")
    print(f"Coverage: {coverage:.4f} ({coverage*100:.2f}%)")
    print(f"Pass threshold: {args.pass_threshold:.4f} ({args.pass_threshold*100:.2f}%)")
    print(f"PASSED: {coverage >= args.pass_threshold}")
    print(f"\nOutputs:")
    print(f"  CSV: {args.out_csv}")
    print(f"  Report: {args.out_report}")
    if not args.no_debug_video:
        print(f"  Debug video: {args.out_debug_video}")
    print(f"{'='*60}\n")

    return 0 if coverage >= args.pass_threshold else 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\nERROR: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
