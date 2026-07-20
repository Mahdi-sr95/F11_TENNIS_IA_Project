"""
Racket 3D Pose Visualization - Generate Plots (Phase 9.5)
Creates trajectory plots for position and orientation analysis.
"""

import argparse
import csv
import os
import sys
import json

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

DEFAULT_POSE_CSV = os.path.join(PROJECT_ROOT, "results", "racketpose", "racket_pose_3d.csv")
DEFAULT_COURT_JSON = os.path.join(PROJECT_ROOT, "results", "intermediate", "court_calib", "court_points.json")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "racketpose")


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _file_must_exist(path: str, what: str) -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{what} not found: {path}")


def _read_pose_csv(path: str) -> dict:
    """Read 3D pose CSV and return dict with arrays."""
    frames = []
    x_vals = []
    y_vals = []
    z_vals = []
    qw_vals = []
    qx_vals = []
    qy_vals = []
    qz_vals = []
    confidence_vals = []
    
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            valid = int(row['valid'])
            if valid:
                frames.append(int(row['frame']))
                x_vals.append(float(row['x']))
                y_vals.append(float(row['y']))
                z_vals.append(float(row['z']))
                qw_vals.append(float(row['qw']))
                qx_vals.append(float(row['qx']))
                qy_vals.append(float(row['qy']))
                qz_vals.append(float(row['qz']))
                confidence_vals.append(float(row['confidence']))
    
    return {
        'frames': np.array(frames),
        'x': np.array(x_vals),
        'y': np.array(y_vals),
        'z': np.array(z_vals),
        'qw': np.array(qw_vals),
        'qx': np.array(qx_vals),
        'qy': np.array(qy_vals),
        'qz': np.array(qz_vals),
        'confidence': np.array(confidence_vals)
    }


def _load_court_dimensions(json_path: str) -> dict:
    """Load court dimensions from JSON."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('court_dimensions', {
        'length': 23.77,
        'width_doubles': 10.97,
        'width_singles': 8.23,
        'service_line_distance': 6.4
    })


def create_3d_trajectory_plot(data: dict, output_path: str):
    """Create 3D scatter plot of racket trajectory."""
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Color by frame number
    colors = data['frames']
    scatter = ax.scatter(data['x'], data['y'], data['z'], 
                        c=colors, cmap='viridis', s=30, alpha=0.6)
    
    ax.set_xlabel('X (m) - Court Width', fontsize=12)
    ax.set_ylabel('Y (m) - Court Length', fontsize=12)
    ax.set_zlabel('Z (m) - Height', fontsize=12)
    ax.set_title('Racket 3D Trajectory', fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label('Frame Index', fontsize=11)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    _ensure_parent_dir(output_path)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 3D trajectory plot saved: {output_path}")


def create_position_over_time_plot(data: dict, output_path: str):
    """Create position (x, y, z) over time plots."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    # X position
    axes[0].plot(data['frames'], data['x'], 'r-', linewidth=1.5, alpha=0.7)
    axes[0].set_ylabel('X (m)', fontsize=11)
    axes[0].set_title('Racket Position Over Time', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    axes[0].axhline(y=10.97, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    
    # Y position
    axes[1].plot(data['frames'], data['y'], 'g-', linewidth=1.5, alpha=0.7)
    axes[1].set_ylabel('Y (m)', fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    axes[1].axhline(y=23.77, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    
    # Z position
    axes[2].plot(data['frames'], data['z'], 'b-', linewidth=1.5, alpha=0.7)
    axes[2].set_ylabel('Z (m)', fontsize=11)
    axes[2].set_xlabel('Frame', fontsize=11)
    axes[2].grid(True, alpha=0.3)
    axes[2].axhline(y=1.0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    
    plt.tight_layout()
    
    _ensure_parent_dir(output_path)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Position over time plot saved: {output_path}")


def create_orientation_over_time_plot(data: dict, output_path: str):
    """Create orientation (quaternion) over time plots."""
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    
    # qw
    axes[0].plot(data['frames'], data['qw'], 'r-', linewidth=1.5, alpha=0.7)
    axes[0].set_ylabel('qw', fontsize=11)
    axes[0].set_title('Racket Orientation (Quaternion) Over Time', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    
    # qx
    axes[1].plot(data['frames'], data['qx'], 'g-', linewidth=1.5, alpha=0.7)
    axes[1].set_ylabel('qx', fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    
    # qy
    axes[2].plot(data['frames'], data['qy'], 'b-', linewidth=1.5, alpha=0.7)
    axes[2].set_ylabel('qy', fontsize=11)
    axes[2].grid(True, alpha=0.3)
    axes[2].axhline(y=0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    
    # qz
    axes[3].plot(data['frames'], data['qz'], 'm-', linewidth=1.5, alpha=0.7)
    axes[3].set_ylabel('qz', fontsize=11)
    axes[3].set_xlabel('Frame', fontsize=11)
    axes[3].grid(True, alpha=0.3)
    axes[3].axhline(y=0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    
    plt.tight_layout()
    
    _ensure_parent_dir(output_path)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Orientation over time plot saved: {output_path}")


def create_topdown_trajectory_plot(data: dict, court_dims: dict, output_path: str):
    """Create top-down view of trajectory on court."""
    fig, ax = plt.subplots(figsize=(8, 16))
    
    court_width = court_dims.get('width_doubles', 10.97)
    court_length = court_dims.get('length', 23.77)
    service_dist = court_dims.get('service_line_distance', 6.4)
    
    # Draw court boundaries
    ax.plot([0, court_width, court_width, 0, 0], 
            [0, 0, court_length, court_length, 0], 
            'k-', linewidth=2, label='Court boundary')
    
    # Service lines
    ax.plot([0, court_width], [service_dist, service_dist], 
            'k--', linewidth=1, alpha=0.6)
    ax.plot([0, court_width], [court_length - service_dist, court_length - service_dist], 
            'k--', linewidth=1, alpha=0.6)
    
    # Net
    ax.plot([0, court_width], [court_length/2, court_length/2], 
            'k-', linewidth=2.5, label='Net')
    
    # Center line
    ax.plot([court_width/2, court_width/2], [0, court_length], 
            'k--', linewidth=1, alpha=0.6)
    
    # Trajectory
    colors = data['frames']
    scatter = ax.scatter(data['x'], data['y'], c=colors, cmap='hot', 
                        s=25, alpha=0.6, edgecolors='k', linewidth=0.3)
    
    ax.set_xlabel('X (m) - Court Width', fontsize=12)
    ax.set_ylabel('Y (m) - Court Length', fontsize=12)
    ax.set_title('Racket Trajectory - Top-Down View', fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=10)
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Frame Index', fontsize=11)
    
    _ensure_parent_dir(output_path)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Top-down trajectory plot saved: {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate plots for racket 3D pose analysis (Phase 9.5)."
    )
    
    parser.add_argument("--pose-csv", default=DEFAULT_POSE_CSV, 
                       help="Input racket_pose_3d.csv path.")
    parser.add_argument("--court-json", default=DEFAULT_COURT_JSON, 
                       help="Court points JSON path.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, 
                       help="Output directory for plots.")
    
    args = parser.parse_args()
    
    _file_must_exist(args.pose_csv, "Pose CSV")
    _file_must_exist(args.court_json, "Court JSON")
    
    print(f"\n{'='*60}")
    print("RACKET 3D POSE PLOTS GENERATION - PHASE 9.5")
    print(f"{'='*60}")
    print(f"Input CSV: {args.pose_csv}")
    print(f"Court JSON: {args.court_json}")
    print(f"Output dir: {args.output_dir}")
    print(f"{'='*60}\n")
    
    # Load data
    print("Loading pose data...")
    data = _read_pose_csv(args.pose_csv)
    print(f"Loaded {len(data['frames'])} valid poses.\n")
    
    print("Loading court dimensions...")
    court_dims = _load_court_dimensions(args.court_json)
    print(f"Court: {court_dims['length']}m x {court_dims['width_doubles']}m\n")
    
    # Generate plots
    print("Generating plots...")
    
    output_3d = os.path.join(args.output_dir, "trajectory_3d.png")
    create_3d_trajectory_plot(data, output_3d)
    
    output_position = os.path.join(args.output_dir, "position_over_time.png")
    create_position_over_time_plot(data, output_position)
    
    output_orientation = os.path.join(args.output_dir, "orientation_over_time.png")
    create_orientation_over_time_plot(data, output_orientation)
    
    output_topdown = os.path.join(args.output_dir, "topdown_trajectory.png")
    create_topdown_trajectory_plot(data, court_dims, output_topdown)
    
    print(f"\n{'='*60}")
    print("PLOTS GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"All plots saved to: {args.output_dir}")
    print(f"{'='*60}\n")
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\nERROR: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
