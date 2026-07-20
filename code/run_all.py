#!/usr/bin/env python3
"""
Complete Tennis Analysis Pipeline
Runs all processing steps from video to final output with bounces

Steps:
1. Ball detection and tracking
2. Bounce detection
3. Video visualization with bounce markers
"""

import sys
from pathlib import Path


def check_required_files():
    """Check if required input files exist."""
    print("\n" + "="*70)
    print("CHECKING REQUIRED FILES")
    print("="*70)

    required = {
        'input_video.mp4': 'Input video file',
    }

    missing = []
    for file, desc in required.items():
        if Path(file).exists():
            print(f"✅ {desc}: {file}")
        else:
            print(f"❌ {desc}: {file} - NOT FOUND")
            missing.append(file)

    if missing:
        print(f"\n❌ ERROR: Missing {len(missing)} required file(s)")
        print("\nPlease ensure 'input_video.mp4' exists in the current directory.")
        return False

    print("\n✅ All required files found")
    return True


def run_ball_tracking():
    """Run ball detection and tracking."""
    print("\n" + "="*70)
    print("STEP 1/3: BALL DETECTION & TRACKING")
    print("="*70)

    try:
        # Import and run ball tracker
        print("\nRunning ball tracker...")
        import ball_track


        # This should create ball_tracks_interpolated.csv
        print("\n✅ Ball tracking completed")
        print("   Output: ball_tracks_interpolated.csv")
        return True

    except Exception as e:
        print(f"\n❌ ERROR in ball tracking: {e}")
        return False


def run_bounce_detection():
    """Run bounce detection."""
    print("\n" + "="*70)
    print("STEP 2/3: BOUNCE DETECTION")
    print("="*70)

    # Check if trajectory file exists
    if not Path('ball_tracks_interpolated.csv').exists():
        print("\n❌ ERROR: ball_tracks_interpolated.csv not found")
        print("   Please run ball tracking first")
        return False

    try:
        # Import and run bounce detector
        print("\nRunning bounce detector...")
        import bounce_detection


        # This should create bounce_frames.json
        print("\n✅ Bounce detection completed")
        print("   Output: bounce_frames.json")
        return True

    except Exception as e:
        print(f"\n❌ ERROR in bounce detection: {e}")
        return False


def run_visualization():
    """Run video visualization."""
    print("\n" + "="*70)
    print("STEP 3/3: VIDEO VISUALIZATION")
    print("="*70)

    # Check required files
    required_files = [
        'input_video.mp4',
        'ball_tracks_interpolated.csv',
        'bounce_frames.json'
    ]

    missing = [f for f in required_files if not Path(f).exists()]

    if missing:
        print(f"\n❌ ERROR: Missing required files:")
        for f in missing:
            print(f"   - {f}")
        return False

    try:
        # Import and run visualizer
        print("\nRunning visualizer...")
        import bounce_detection


        # This should create output_with_bounces.mp4
        print("\n✅ Visualization completed")
        print("   Output: output_with_bounces.mp4")
        return True

    except Exception as e:
        print(f"\n❌ ERROR in visualization: {e}")
        import traceback
        traceback.print_exc()
        return False


def print_final_summary():
    """Print final summary of all outputs."""
    print("\n" + "="*70)
    print("PIPELINE COMPLETE - SUMMARY")
    print("="*70)

    outputs = {
        'ball_tracks_interpolated.csv': 'Ball trajectory data',
        'bounce_frames.json': 'Bounce detection results',
        'output_with_bounces.mp4': 'Final video with bounce markers'
    }

    print("\nGenerated Files:")
    for file, desc in outputs.items():
        if Path(file).exists():
            size = Path(file).stat().st_size
            size_mb = size / (1024 * 1024)
            print(f"  ✅ {file}")
            print(f"     {desc} ({size_mb:.2f} MB)")
        else:
            print(f"  ❌ {file} - NOT CREATED")

    print("\n" + "="*70)


def main():
    """Run complete pipeline."""
    print("\n" + "="*70)
    print("TENNIS ANALYSIS PIPELINE")
    print("Complete ball tracking and bounce detection")
    print("="*70)

    # Check input files
    if not check_required_files():
        sys.exit(1)

    # Step 1: Ball tracking
    if not run_ball_tracking():
        print("\n❌ Pipeline stopped: Ball tracking failed")
        sys.exit(1)

    # Step 2: Bounce detection
    if not run_bounce_detection():
        print("\n❌ Pipeline stopped: Bounce detection failed")
        sys.exit(1)

    # Step 3: Visualization
    if not run_visualization():
        print("\n⚠️  Warning: Visualization failed, but results are available")

    # Final summary
    print_final_summary()

    print("\n✅ PIPELINE EXECUTION COMPLETE")
    print("\nTo view results:")
    print("  - Ball trajectory: ball_tracks_interpolated.csv")
    print("  - Bounce data: bounce_frames.json")
    print("  - Final video: output_with_bounces.mp4")
    print()


if __name__ == "__main__":
    main()
