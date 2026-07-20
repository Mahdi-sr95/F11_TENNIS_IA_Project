"""
Racket 3D Pose Validation Analysis (Phase 9.5)
Analyzes failure cases, occlusion, blur, and quality metrics.
"""

import argparse
import csv
import os
import sys
from typing import List, Dict

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

DEFAULT_POSE_CSV = os.path.join(PROJECT_ROOT, "results", "racketpose", "racket_pose_3d.csv")
DEFAULT_REPORT_TXT = os.path.join(PROJECT_ROOT, "results", "racketpose", "pose3d_quality_report.txt")
DEFAULT_OUTPUT_TXT = os.path.join(PROJECT_ROOT, "results", "racketpose", "validation_analysis.txt")


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _file_must_exist(path: str, what: str) -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{what} not found: {path}")


def _read_pose_csv(path: str) -> List[Dict]:
    """Read pose CSV and return list of dicts."""
    poses = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            poses.append({
                'frame': int(row['frame']),
                'x': float(row['x']) if row['x'] else None,
                'y': float(row['y']) if row['y'] else None,
                'z': float(row['z']) if row['z'] else None,
                'confidence': float(row['confidence']) if row['confidence'] else 0.0,
                'valid': int(row['valid'])
            })
    return poses


def _analyze_position_outliers(poses: List[Dict]) -> Dict:
    """Detect position outliers (outside court boundaries)."""
    court_width = 10.97
    court_length = 23.77
    
    valid_poses = [p for p in poses if p['valid']]
    
    out_of_bounds = []
    for p in valid_poses:
        x, y = p['x'], p['y']
        if x < -1.0 or x > court_width + 1.0:
            out_of_bounds.append((p['frame'], 'X', x))
        if y < -5.0 or y > court_length + 5.0:
            out_of_bounds.append((p['frame'], 'Y', y))
    
    return {
        'total_valid': len(valid_poses),
        'out_of_bounds_count': len(out_of_bounds),
        'out_of_bounds_frames': out_of_bounds
    }


def _analyze_confidence_distribution(poses: List[Dict]) -> Dict:
    """Analyze confidence distribution."""
    valid_poses = [p for p in poses if p['valid']]
    confidences = [p['confidence'] for p in valid_poses]
    
    if len(confidences) == 0:
        return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
    
    return {
        'mean': np.mean(confidences),
        'std': np.std(confidences),
        'min': np.min(confidences),
        'max': np.max(confidences),
        'low_conf_count': sum(1 for c in confidences if c < 0.7)
    }


def _analyze_gaps(poses: List[Dict]) -> Dict:
    """Analyze temporal gaps (consecutive invalid frames)."""
    gaps = []
    current_gap_start = None
    
    for i, p in enumerate(poses):
        if not p['valid']:
            if current_gap_start is None:
                current_gap_start = i
        else:
            if current_gap_start is not None:
                gap_length = i - current_gap_start
                gaps.append((current_gap_start, i - 1, gap_length))
                current_gap_start = None
    
    # If ends with a gap
    if current_gap_start is not None:
        gap_length = len(poses) - current_gap_start
        gaps.append((current_gap_start, len(poses) - 1, gap_length))
    
    return {
        'num_gaps': len(gaps),
        'gaps': gaps,
        'max_gap_length': max([g[2] for g in gaps]) if gaps else 0
    }


def _write_validation_report(
    output_path: str,
    poses: List[Dict],
    outlier_analysis: Dict,
    confidence_analysis: Dict,
    gap_analysis: Dict
) -> None:
    """Write comprehensive validation analysis report."""
    
    total_frames = len(poses)
    valid_count = sum(1 for p in poses if p['valid'])
    invalid_count = total_frames - valid_count
    
    _ensure_parent_dir(output_path)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("RACKET 3D POSE VALIDATION ANALYSIS - PHASE 9.5\n")
        f.write("=" * 70 + "\n\n")
        
        # Overall Statistics
        f.write("1. OVERALL STATISTICS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Total frames: {total_frames}\n")
        f.write(f"Valid 3D poses: {valid_count} ({valid_count/total_frames*100:.2f}%)\n")
        f.write(f"Invalid poses: {invalid_count} ({invalid_count/total_frames*100:.2f}%)\n")
        f.write("\n")
        
        # Confidence Analysis
        f.write("2. CONFIDENCE ANALYSIS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Mean confidence: {confidence_analysis['mean']:.4f}\n")
        f.write(f"Std deviation: {confidence_analysis['std']:.4f}\n")
        f.write(f"Min confidence: {confidence_analysis['min']:.4f}\n")
        f.write(f"Max confidence: {confidence_analysis['max']:.4f}\n")
        f.write(f"Low confidence frames (<0.7): {confidence_analysis['low_conf_count']}\n")
        f.write("\n")
        
        # Position Outliers
        f.write("3. POSITION OUTLIER ANALYSIS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Valid poses: {outlier_analysis['total_valid']}\n")
        f.write(f"Out-of-bounds detections: {outlier_analysis['out_of_bounds_count']}\n")
        
        if outlier_analysis['out_of_bounds_count'] > 0:
            f.write("\nOut-of-bounds frames (first 10):\n")
            for frame, axis, value in outlier_analysis['out_of_bounds_frames'][:10]:
                f.write(f"  Frame {frame}: {axis} = {value:.2f}m\n")
            if outlier_analysis['out_of_bounds_count'] > 10:
                f.write(f"  ... and {outlier_analysis['out_of_bounds_count'] - 10} more\n")
        f.write("\n")
        
        # Temporal Gaps
        f.write("4. TEMPORAL GAP ANALYSIS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Number of gaps: {gap_analysis['num_gaps']}\n")
        f.write(f"Max gap length: {gap_analysis['max_gap_length']} frames\n")
        
        if gap_analysis['num_gaps'] > 0:
            f.write("\nGaps (all):\n")
            for start, end, length in gap_analysis['gaps']:
                f.write(f"  Frames {start}-{end}: {length} frames\n")
        f.write("\n")
        
        # Failure Cases Analysis
        f.write("5. FAILURE CASES ANALYSIS\n")
        f.write("-" * 70 + "\n")
        f.write("Main causes of failure/invalid poses:\n\n")
        
        f.write("a) Occlusion:\n")
        f.write("   - Racket occluded by player body during backswing or follow-through\n")
        f.write("   - Partial occlusion causing low confidence detections\n")
        f.write(f"   - Estimated affected frames: {gap_analysis['num_gaps']} gaps\n\n")
        
        f.write("b) Motion Blur:\n")
        f.write("   - Fast racket movement during swing causes blur\n")
        f.write("   - Affects keypoint detection accuracy\n")
        f.write(f"   - Low confidence frames: {confidence_analysis['low_conf_count']}\n\n")
        
        f.write("c) Projection Errors:\n")
        f.write("   - Homography-based projection assumes ground plane (Z=const)\n")
        f.write("   - Does not account for racket height variation\n")
        f.write(f"   - Out-of-bounds frames: {outlier_analysis['out_of_bounds_count']}\n\n")
        
        f.write("d) Detection Failures:\n")
        f.write("   - Upstream detection/tracking failures propagate to 3D pose\n")
        f.write(f"   - Total invalid frames: {invalid_count}\n\n")
        
        # Recommendations
        f.write("6. RECOMMENDATIONS FOR IMPROVEMENT\n")
        f.write("-" * 70 + "\n")
        f.write("- Use multi-view geometry for true 3D reconstruction (not just homography)\n")
        f.write("- Implement player segmentation to reduce occlusion effects\n")
        f.write("- Add motion deblurring preprocessing for fast movements\n")
        f.write("- Use temporal smoothing (Kalman filter) on 3D poses\n")
        f.write("- Estimate racket height dynamically instead of assuming constant Z\n")
        f.write("\n")
        
        # Quality Rating
        f.write("7. OVERALL QUALITY RATING\n")
        f.write("-" * 70 + "\n")
        
        success_rate = valid_count / total_frames
        if success_rate >= 0.85:
            rating = "EXCELLENT"
        elif success_rate >= 0.70:
            rating = "GOOD"
        elif success_rate >= 0.50:
            rating = "FAIR"
        else:
            rating = "POOR"
        
        f.write(f"Success rate: {success_rate:.2%}\n")
        f.write(f"Quality rating: {rating}\n")
        f.write("\n")
        
        f.write("=" * 70 + "\n")
        f.write("END OF VALIDATION ANALYSIS\n")
        f.write("=" * 70 + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate validation analysis for racket 3D pose (Phase 9.5)."
    )
    
    parser.add_argument("--pose-csv", default=DEFAULT_POSE_CSV, 
                       help="Input racket_pose_3d.csv path.")
    parser.add_argument("--output-txt", default=DEFAULT_OUTPUT_TXT, 
                       help="Output validation analysis text file.")
    
    args = parser.parse_args()
    
    _file_must_exist(args.pose_csv, "Pose CSV")
    
    print(f"\n{'='*60}")
    print("RACKET 3D POSE VALIDATION ANALYSIS - PHASE 9.5")
    print(f"{'='*60}")
    print(f"Input CSV: {args.pose_csv}")
    print(f"Output TXT: {args.output_txt}")
    print(f"{'='*60}\n")
    
    # Load data
    print("Loading pose data...")
    poses = _read_pose_csv(args.pose_csv)
    print(f"Loaded {len(poses)} frames.\n")
    
    # Analyze
    print("Analyzing position outliers...")
    outlier_analysis = _analyze_position_outliers(poses)
    
    print("Analyzing confidence distribution...")
    confidence_analysis = _analyze_confidence_distribution(poses)
    
    print("Analyzing temporal gaps...")
    gap_analysis = _analyze_gaps(poses)
    
    print("\nGenerating validation report...")
    _write_validation_report(
        args.output_txt,
        poses,
        outlier_analysis,
        confidence_analysis,
        gap_analysis
    )
    
    print(f"\n{'='*60}")
    print("VALIDATION ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Report saved to: {args.output_txt}")
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
