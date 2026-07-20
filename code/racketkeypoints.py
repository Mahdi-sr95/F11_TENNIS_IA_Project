import argparse
import csv
import os
import sys
from dataclasses import dataclass
from typing import Optional, Tuple, List

import cv2
import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

DEFAULT_VIDEO_PATH = os.path.join(PROJECT_ROOT, "data", "clips", "input_video.mp4")
DEFAULT_DETECTIONS_CSV = os.path.join(PROJECT_ROOT, "results", "racketpose", "racket2d.csv")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "racketpose")
DEFAULT_OUT_CSV = os.path.join(DEFAULT_OUTPUT_DIR, "racket_keypoints_2d.csv")
DEFAULT_OUT_REPORT = os.path.join(DEFAULT_OUTPUT_DIR, "keypoints_quality_report.txt")
DEFAULT_OUT_DEBUG_VIDEO = os.path.join(DEFAULT_OUTPUT_DIR, "overlay_racket_keypoints.mp4")


@dataclass
class KeypointResult:
    frame_index: int
    detected: int
    center_x: Optional[float]
    center_y: Optional[float]
    angle_deg: Optional[float]
    handle_x: Optional[float]
    handle_y: Optional[float]
    head_x: Optional[float]
    head_y: Optional[float]
    quality: Optional[float]


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _file_must_exist(path: str, what: str) -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{what} not found: {path}")


def _extract_racket_axis_pca(
    frame: np.ndarray, bbox: Tuple[float, float, float, float]
) -> Optional[Tuple[float, float, float, float, float, float, float, float]]:
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    
    h, w = frame.shape[:2]
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))
    
    if x2 <= x1 or y2 <= y1:
        return None
    
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    edges = cv2.Canny(gray, 50, 150)
    
    combined = cv2.bitwise_or(binary, edges)
    
    coords = np.column_stack(np.where(combined > 0))
    
    if len(coords) < 10:
        return None
    
    coords_float = coords.astype(np.float64)
    
    mean = np.mean(coords_float, axis=0)
    coords_centered = coords_float - mean
    
    cov = np.cov(coords_centered.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    
    sort_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sort_indices]
    eigenvectors = eigenvectors[:, sort_indices]
    
    principal_axis = eigenvectors[:, 0]
    
    quality = float(eigenvalues[0] / (eigenvalues[1] + 1e-6))
    
    cy_local, cx_local = mean
    cx_global = cx_local + x1
    cy_global = cy_local + y1
    
    angle_rad = np.arctan2(principal_axis[1], principal_axis[0])
    angle_deg = np.degrees(angle_rad)
    
    axis_length = max(roi.shape) * 0.4
    
    handle_x = cx_global - axis_length * np.cos(angle_rad)
    handle_y = cy_global - axis_length * np.sin(angle_rad)
    head_x = cx_global + axis_length * np.cos(angle_rad)
    head_y = cy_global + axis_length * np.sin(angle_rad)
    
    return (cx_global, cy_global, angle_deg, handle_x, handle_y, head_x, head_y, quality)


def _read_detections_csv(path: str) -> List[Tuple[int, int, Optional[Tuple[float, float, float, float]]]]:
    detections = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame_idx = int(row["frame"])
            detected = int(row["detected"])
            if detected == 1:
                x1 = float(row["x1"])
                y1 = float(row["y1"])
                x2 = float(row["x2"])
                y2 = float(row["y2"])
                bbox = (x1, y1, x2, y2)
            else:
                bbox = None
            detections.append((frame_idx, detected, bbox))
    return detections


def _write_keypoints_csv(path: str, rows: List[KeypointResult]) -> None:
    _ensure_parent_dir(path)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame", "detected", "center_x", "center_y", "angle_deg", "handle_x", "handle_y", "head_x", "head_y", "quality"])
        for r in rows:
            w.writerow([
                r.frame_index,
                r.detected,
                r.center_x,
                r.center_y,
                r.angle_deg,
                r.handle_x,
                r.handle_y,
                r.head_x,
                r.head_y,
                r.quality,
            ])


def _write_report(
    path: str,
    total_detections: int,
    successful_keypoints: int,
    success_rate: float,
    pass_threshold: float,
) -> None:
    _ensure_parent_dir(path)
    passed = success_rate >= pass_threshold
    with open(path, "w", encoding="utf-8") as f:
        f.write("=== Racket Keypoints (2D) - Quality Report ===\n\n")
        f.write(f"Total detected frames (from 9.1): {total_detections}\n")
        f.write(f"Successful keypoint extraction: {successful_keypoints}\n")
        f.write(f"Success rate: {success_rate:.4f} ({success_rate*100:.2f}%)\n")
        f.write(f"Pass threshold: {pass_threshold:.4f} ({pass_threshold*100:.2f}%)\n\n")
        f.write(f"PASSED: {passed}\n")


def _open_video(video_path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    return cap


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract 2D keypoints/orientation from detected racket bboxes."
    )

    parser.add_argument("--video", default=DEFAULT_VIDEO_PATH, help="Input video path.")
    parser.add_argument("--detections-csv", default=DEFAULT_DETECTIONS_CSV, help="Input detections CSV from phase 9.1.")
    parser.add_argument("--out-csv", default=DEFAULT_OUT_CSV, help="Output keypoints CSV path.")
    parser.add_argument("--out-report", default=DEFAULT_OUT_REPORT, help="Output report path.")
    parser.add_argument("--no-debug-video", action="store_true", help="Disable writing debug video.")
    parser.add_argument("--out-debug-video", default=DEFAULT_OUT_DEBUG_VIDEO, help="Debug video output path.")
    parser.add_argument("--pass-threshold", type=float, default=0.70, help="Minimum acceptable success rate.")
    parser.add_argument("--progress-interval", type=int, default=100, help="Print progress every N frames.")

    args = parser.parse_args()

    _file_must_exist(args.video, "Input video")
    _file_must_exist(args.detections_csv, "Detections CSV")

    print(f"\n{'='*60}")
    print("RACKET KEYPOINTS EXTRACTION (2D) - PHASE 9.2")
    print(f"{'='*60}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Video: {args.video}")
    print(f"Detections CSV: {args.detections_csv}")
    print(f"Pass threshold: {args.pass_threshold:.2%}")
    print(f"{'='*60}\n")

    detections = _read_detections_csv(args.detections_csv)
    total_frames = len(detections)
    total_detections = sum(1 for _, det, _ in detections if det == 1)

    print(f"Loaded {total_frames} frames, {total_detections} detections.\n")

    cap = _open_video(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 30.0

    results: List[KeypointResult] = []
    successful_keypoints = 0

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

    print("Processing frames for keypoint extraction...")
    frame_idx = 0
    for det_frame_idx, detected, bbox in detections:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx != det_frame_idx:
            print(f"Warning: frame index mismatch at {frame_idx} vs {det_frame_idx}")

        if detected == 0 or bbox is None:
            results.append(KeypointResult(
                frame_index=frame_idx,
                detected=0,
                center_x=None,
                center_y=None,
                angle_deg=None,
                handle_x=None,
                handle_y=None,
                head_x=None,
                head_y=None,
                quality=None,
            ))
        else:
            axis_data = _extract_racket_axis_pca(frame, bbox)
            if axis_data is not None:
                cx, cy, angle, hx, hy, hdx, hdy, quality = axis_data
                results.append(KeypointResult(
                    frame_index=frame_idx,
                    detected=1,
                    center_x=cx,
                    center_y=cy,
                    angle_deg=angle,
                    handle_x=hx,
                    handle_y=hy,
                    head_x=hdx,
                    head_y=hdy,
                    quality=quality,
                ))
                successful_keypoints += 1
            else:
                results.append(KeypointResult(
                    frame_index=frame_idx,
                    detected=0,
                    center_x=None,
                    center_y=None,
                    angle_deg=None,
                    handle_x=None,
                    handle_y=None,
                    head_x=None,
                    head_y=None,
                    quality=None,
                ))

        if writer is not None:
            vis = frame.copy()
            kp = results[-1]

            if kp.detected == 1 and kp.center_x is not None:
                cx, cy = int(kp.center_x), int(kp.center_y)
                hx, hy = int(kp.handle_x), int(kp.handle_y)
                hdx, hdy = int(kp.head_x), int(kp.head_y)

                cv2.line(vis, (hx, hy), (hdx, hdy), (0, 255, 255), 3)
                cv2.circle(vis, (cx, cy), 6, (255, 0, 0), -1)
                cv2.circle(vis, (hx, hy), 6, (0, 255, 0), -1)
                cv2.circle(vis, (hdx, hdy), 6, (0, 0, 255), -1)

                label = f"Angle: {kp.angle_deg:.1f} deg"
                cv2.putText(vis, label, (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

                status_text = "KEYPOINTS OK"
                status_color = (0, 255, 0)
            else:
                status_text = "NO KEYPOINTS"
                status_color = (0, 0, 255)

            cv2.putText(vis, f"Frame: {frame_idx}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(vis, f"Status: {status_text}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2, cv2.LINE_AA)
            cv2.putText(vis, f"Success: {successful_keypoints}/{frame_idx+1}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2, cv2.LINE_AA)

            writer.write(vis)

        if args.progress_interval > 0 and (frame_idx + 1) % args.progress_interval == 0:
            current_rate = successful_keypoints / total_detections if total_detections > 0 else 0.0
            print(f"  Frame {frame_idx + 1}: Keypoints {successful_keypoints} / {total_detections} detected ({current_rate:.2%})")

        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()

    success_rate = successful_keypoints / total_detections if total_detections > 0 else 0.0

    _write_keypoints_csv(args.out_csv, results)
    _write_report(args.out_report, total_detections, successful_keypoints, success_rate, args.pass_threshold)

    print(f"\n{'='*60}")
    print("KEYPOINT EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"Total detected frames: {total_detections}")
    print(f"Successful keypoint extraction: {successful_keypoints}")
    print(f"Success rate: {success_rate:.4f} ({success_rate*100:.2f}%)")
    print(f"Pass threshold: {args.pass_threshold:.4f} ({args.pass_threshold*100:.2f}%)")
    print(f"PASSED: {success_rate >= args.pass_threshold}")
    print(f"\nOutputs:")
    print(f"  CSV: {args.out_csv}")
    print(f"  Report: {args.out_report}")
    if not args.no_debug_video:
        print(f"  Debug video: {args.out_debug_video}")
    print(f"{'='*60}\n")

    return 0 if success_rate >= args.pass_threshold else 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\nERROR: {str(e)}", file=sys.stderr)
        sys.exit(1)
