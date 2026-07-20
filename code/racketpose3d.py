import argparse
import csv
import os
import sys
from dataclasses import dataclass
from typing import Optional, List, Tuple
import json

import cv2
import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

DEFAULT_VIDEO_PATH = os.path.join(PROJECT_ROOT, "data", "clips", "input_video.mp4")
DEFAULT_TRAJECTORY_CSV = os.path.join(PROJECT_ROOT, "results", "racketpose", "racket_trajectory_smooth.csv")
DEFAULT_COURT_POINTS_JSON = os.path.join(PROJECT_ROOT, "results", "intermediate", "court_calib", "court_points.json")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "racketpose")
DEFAULT_OUT_CSV = os.path.join(DEFAULT_OUTPUT_DIR, "racket_pose_3d.csv")
DEFAULT_OUT_REPORT = os.path.join(DEFAULT_OUTPUT_DIR, "pose3d_quality_report.txt")
DEFAULT_OUT_DEBUG_VIDEO = os.path.join(DEFAULT_OUTPUT_DIR, "overlay_racket_3d.mp4")


@dataclass
class Pose3D:
    frame_index: int
    x: Optional[float]
    y: Optional[float]
    z: Optional[float]
    qw: Optional[float]
    qx: Optional[float]
    qy: Optional[float]
    qz: Optional[float]
    confidence: float
    valid: bool


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _file_must_exist(path: str, what: str) -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{what} not found: {path}")


def _load_homography_from_court_points(json_path: str) -> np.ndarray:
    """
    Load court points and compute homography from image to world (court).
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract image points and world points
    points_image = data.get("points_image", {})
    points_world = data.get("points_world", {})
    
    if not points_image or not points_world:
        raise ValueError("JSON file must contain 'points_image' and 'points_world'")
    
    img_points = []
    world_points = []
    
    # Iterate through all available points
    for key in points_image.keys():
        if key in points_world:
            img_coord = points_image[key]
            world_coord = points_world[key]
            
            # points_image: [x, y] in pixels
            # points_world: [x, y] in meters
            img_points.append([float(img_coord[0]), float(img_coord[1])])
            world_points.append([float(world_coord[0]), float(world_coord[1])])
    
    if len(img_points) < 4:
        raise ValueError(f"Not enough court points found. Need at least 4, got {len(img_points)}")
    
    img_points = np.array(img_points, dtype=np.float32)
    world_points = np.array(world_points, dtype=np.float32)
    
    print(f"Loaded {len(img_points)} court points:")
    print(f"  Image points shape: {img_points.shape}")
    print(f"  World points shape: {world_points.shape}")
    
    # Compute homography: world -> image
    H_world2img, mask = cv2.findHomography(world_points, img_points, cv2.RANSAC, 5.0)
    
    if H_world2img is None:
        raise RuntimeError("Failed to compute homography")
    
    # We need image -> world for projection
    H_img2world = np.linalg.inv(H_world2img)
    
    print(f"\nComputed homography (image to world):")
    print(H_img2world)
    print()
    
    return H_img2world


def _read_trajectory_csv(path: str) -> List[Tuple]:
    trajectory = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame_idx = int(row["frame"])
            if row["center_x"] and row["center_x"] != "":
                cx = float(row["center_x"])
                cy = float(row["center_y"])
                angle = float(row["angle_deg"])
                hx = float(row["handle_x"])
                hy = float(row["handle_y"])
                hdx = float(row["head_x"])
                hdy = float(row["head_y"])
                confidence = float(row["confidence"])
                trajectory.append((frame_idx, (cx, cy, angle, hx, hy, hdx, hdy, confidence)))
            else:
                trajectory.append((frame_idx, None))
    return trajectory


def _rotation_matrix_to_quaternion(R: np.ndarray) -> Tuple[float, float, float, float]:
    """Convert 3x3 rotation matrix to quaternion (w,x,y,z)."""
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    
    return (w, x, y, z)


def _estimate_racket_3d_pose(
    center_2d: np.ndarray,
    handle_2d: np.ndarray,
    head_2d: np.ndarray,
    H_img2world: np.ndarray,
    assumed_height: float = 1.0,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Estimate 3D position and orientation of racket.
    
    Returns: (position_3d, quaternion) or None
    """
    # Project center to ground plane to get x,y
    center_homo = np.array([center_2d[0], center_2d[1], 1.0])
    ground_homo = H_img2world @ center_homo
    ground_homo /= ground_homo[2]
    
    x_ground = ground_homo[0]
    y_ground = ground_homo[1]
    
    # Assume racket is at a certain height
    z = assumed_height
    
    position_3d = np.array([x_ground, y_ground, z])
    
    # Project handle and head to ground plane to get 3D direction
    handle_homo = np.array([handle_2d[0], handle_2d[1], 1.0])
    handle_ground_homo = H_img2world @ handle_homo
    handle_ground_homo /= handle_ground_homo[2]
    
    head_homo = np.array([head_2d[0], head_2d[1], 1.0])
    head_ground_homo = H_img2world @ head_homo
    head_ground_homo /= head_ground_homo[2]
    
    # 3D direction in ground plane
    axis_3d_ground = head_ground_homo[:2] - handle_ground_homo[:2]
    axis_length = np.linalg.norm(axis_3d_ground)
    
    if axis_length < 1e-6:
        return None
    
    axis_3d_ground_norm = axis_3d_ground / axis_length
    
    # Create rotation matrix (simplified - racket axis in ground plane)
    z_axis = np.array([axis_3d_ground_norm[0], axis_3d_ground_norm[1], 0.0])
    z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-6)
    
    x_axis = np.array([-z_axis[1], z_axis[0], 0.0])
    x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-6)
    
    y_axis = np.array([0.0, 0.0, 1.0])
    
    R = np.column_stack([x_axis, y_axis, z_axis])
    
    quat = _rotation_matrix_to_quaternion(R)
    
    return (position_3d, quat)


def _process_3d_poses(
    trajectory_list: List[Tuple],
    H_img2world: np.ndarray,
    assumed_height: float = 1.0,
) -> List[Pose3D]:
    """Process all trajectory points to 3D poses."""
    results = []
    
    for frame_idx, traj_data in trajectory_list:
        if traj_data is None:
            results.append(Pose3D(
                frame_index=frame_idx,
                x=None, y=None, z=None,
                qw=None, qx=None, qy=None, qz=None,
                confidence=0.0,
                valid=False,
            ))
            continue
        
        cx, cy, angle, hx, hy, hdx, hdy, confidence = traj_data
        
        center_2d = np.array([cx, cy])
        handle_2d = np.array([hx, hy])
        head_2d = np.array([hdx, hdy])
        
        try:
            pose_result = _estimate_racket_3d_pose(
                center_2d, handle_2d, head_2d, H_img2world, assumed_height
            )
            
            if pose_result is not None:
                position_3d, quat = pose_result
                results.append(Pose3D(
                    frame_index=frame_idx,
                    x=float(position_3d[0]),
                    y=float(position_3d[1]),
                    z=float(position_3d[2]),
                    qw=float(quat[0]),
                    qx=float(quat[1]),
                    qy=float(quat[2]),
                    qz=float(quat[3]),
                    confidence=confidence,
                    valid=True,
                ))
            else:
                results.append(Pose3D(
                    frame_index=frame_idx,
                    x=None, y=None, z=None,
                    qw=None, qx=None, qy=None, qz=None,
                    confidence=0.0,
                    valid=False,
                ))
        except Exception:
            results.append(Pose3D(
                frame_index=frame_idx,
                x=None, y=None, z=None,
                qw=None, qx=None, qy=None, qz=None,
                confidence=0.0,
                valid=False,
            ))
    
    return results


def _write_pose3d_csv(path: str, rows: List[Pose3D]) -> None:
    _ensure_parent_dir(path)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame", "x", "y", "z", "qw", "qx", "qy", "qz", "confidence", "valid"])
        for r in rows:
            w.writerow([
                r.frame_index,
                r.x, r.y, r.z,
                r.qw, r.qx, r.qy, r.qz,
                r.confidence,
                int(r.valid),
            ])


def _write_report(
    path: str,
    total_frames: int,
    trajectory_valid: int,
    pose_valid: int,
    success_rate: float,
    pass_threshold: float,
) -> None:
    _ensure_parent_dir(path)
    passed = success_rate >= pass_threshold
    with open(path, "w", encoding="utf-8") as f:
        f.write("=== Racket 3D Pose Estimation - Quality Report ===\n\n")
        f.write(f"Total frames: {total_frames}\n")
        f.write(f"Trajectory valid frames: {trajectory_valid}\n")
        f.write(f"3D pose valid frames: {pose_valid}\n")
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
        description="Estimate 3D pose (position + orientation) of tennis racket (Phase 9.4)."
    )

    parser.add_argument("--video", default=DEFAULT_VIDEO_PATH, help="Input video path.")
    parser.add_argument("--trajectory-csv", default=DEFAULT_TRAJECTORY_CSV, help="Input smooth trajectory CSV.")
    parser.add_argument("--court-points", default=DEFAULT_COURT_POINTS_JSON, help="Court points JSON file.")
    parser.add_argument("--out-csv", default=DEFAULT_OUT_CSV, help="Output 3D pose CSV path.")
    parser.add_argument("--out-report", default=DEFAULT_OUT_REPORT, help="Output report path.")
    parser.add_argument("--no-debug-video", action="store_true", help="Disable debug video.")
    parser.add_argument("--out-debug-video", default=DEFAULT_OUT_DEBUG_VIDEO, help="Debug video path.")
    parser.add_argument("--assumed-height", type=float, default=1.0, help="Assumed racket height (meters).")
    parser.add_argument("--pass-threshold", type=float, default=0.70, help="Min success rate.")
    parser.add_argument("--progress-interval", type=int, default=100, help="Progress print interval.")

    args = parser.parse_args()

    _file_must_exist(args.video, "Input video")
    _file_must_exist(args.trajectory_csv, "Trajectory CSV")
    _file_must_exist(args.court_points, "Court points JSON")

    print(f"\n{'='*60}")
    print("RACKET 3D POSE ESTIMATION - PHASE 9.4")
    print(f"{'='*60}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Video: {args.video}")
    print(f"Trajectory CSV: {args.trajectory_csv}")
    print(f"Court points: {args.court_points}")
    print(f"Assumed height: {args.assumed_height}m")
    print(f"Pass threshold: {args.pass_threshold:.2%}")
    print(f"{'='*60}\n")

    # Compute homography from court points
    H_img2world = _load_homography_from_court_points(args.court_points)

    # Load trajectory
    trajectory_list = _read_trajectory_csv(args.trajectory_csv)
    total_frames = len(trajectory_list)
    trajectory_valid = sum(1 for _, data in trajectory_list if data is not None)

    print(f"Loaded {total_frames} frames, {trajectory_valid} with valid trajectory.\n")

    # Process 3D poses
    print("Computing 3D poses...")
    poses = _process_3d_poses(trajectory_list, H_img2world, args.assumed_height)

    pose_valid = sum(1 for p in poses if p.valid)
    success_rate = pose_valid / trajectory_valid if trajectory_valid > 0 else 0.0

    print(f"\n3D pose computation complete:")
    print(f"  Valid 3D poses: {pose_valid}/{trajectory_valid} trajectory points")
    print(f"  Success rate: {success_rate:.2%}\n")

    _write_pose3d_csv(args.out_csv, poses)
    _write_report(args.out_report, total_frames, trajectory_valid, pose_valid, success_rate, args.pass_threshold)

    print(f"Outputs written:")
    print(f"  CSV: {args.out_csv}")
    print(f"  Report: {args.out_report}\n")

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

    if writer is not None:
        print("Rendering debug video...")
        frame_idx = 0
        for pose in poses:
            ret, frame = cap.read()
            if not ret:
                break

            vis = frame.copy()

            if pose.valid:
                status_text = "3D POSE OK"
                status_color = (0, 255, 0)
                info_text = f"Pos: ({pose.x:.2f}, {pose.y:.2f}, {pose.z:.2f})m"
            else:
                status_text = "NO 3D POSE"
                status_color = (0, 0, 255)
                info_text = ""

            cv2.putText(vis, f"Frame: {frame_idx}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(vis, f"Status: {status_text}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2, cv2.LINE_AA)
            if info_text:
                cv2.putText(vis, info_text, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)

            writer.write(vis)

            if args.progress_interval > 0 and (frame_idx + 1) % args.progress_interval == 0:
                print(f"  Frame {frame_idx + 1}/{total_frames}")

            frame_idx += 1

        writer.release()

    cap.release()

    print(f"\n{'='*60}")
    print("3D POSE ESTIMATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total frames: {total_frames}")
    print(f"Trajectory valid: {trajectory_valid}")
    print(f"3D pose valid: {pose_valid}")
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
        import traceback
        traceback.print_exc()
        sys.exit(1)
