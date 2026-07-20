#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
TENNIS BOUNCE VISUALIZATION - SIMPLE & ACCURATE
================================================================================

Creates output video with bounce markers at EXACT specified frames.
No detection algorithms - just marking the frames you identified.

Bounce frames: 137, 228, 323, 408, 474, 585, 686, 759, 845

Author: Tennis Analysis Team
Date: January 2026
Version: 3.0 (SIMPLIFIED)
================================================================================
"""

import cv2
import pandas as pd
from pathlib import Path


# ============================================================================
# CONFIGURATION
# ============================================================================

# Project paths
PROJECT_ROOT = Path(r"C:/Users/Ost/Desktop/F11_TENNIS")

# Input files
INPUT_VIDEO = PROJECT_ROOT / "data" / "Clips" / "input_video.mp4"
BALL_TRAJECTORY = PROJECT_ROOT / "ball_tracks_interpolated.csv"

# Output file
OUTPUT_VIDEO = PROJECT_ROOT / "results" / "final" / "output_with_bounces.mp4"

# EXACT bounce frames (manually identified by you)
BOUNCE_FRAMES = [137, 228, 323, 408, 474, 585, 686, 759, 845]


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def create_bounce_video():
    """
    Create video with bounce markers at exact specified frames.
    
    This function:
    1. Reads input video frame by frame
    2. Draws ball trajectory (green circle) from CSV
    3. Draws bounce marker (red circles + text) ONLY at specified frames
    4. Saves output video
    """
    
    print("\n" + "="*70)
    print("TENNIS BOUNCE VISUALIZATION")
    print("="*70)
    print(f"\nBounce frames to mark: {BOUNCE_FRAMES}")
    print(f"Total bounces: {len(BOUNCE_FRAMES)}")
    print("="*70)
    
    # Check input files exist
    if not INPUT_VIDEO.exists():
        print(f"\n❌ ERROR: Input video not found!")
        print(f"   Expected: {INPUT_VIDEO}")
        return False
    
    if not BALL_TRAJECTORY.exists():
        print(f"\n❌ ERROR: Ball trajectory not found!")
        print(f"   Expected: {BALL_TRAJECTORY}")
        return False
    
    print(f"\n✅ Input video: {INPUT_VIDEO.name}")
    print(f"✅ Ball trajectory: {BALL_TRAJECTORY.name}")
    
    # Load ball trajectory
    print("\n📂 Loading ball trajectory...")
    trajectory = pd.read_csv(BALL_TRAJECTORY)
    trajectory = trajectory[trajectory['final_x'].notna() & trajectory['final_y'].notna()].copy()
    print(f"✅ Loaded {len(trajectory)} ball positions")
    
    # Open input video
    print("\n📂 Opening input video...")
    cap = cv2.VideoCapture(str(INPUT_VIDEO))
    
    if not cap.isOpened():
        print("❌ ERROR: Cannot open input video")
        return False
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"✅ Video properties:")
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps}")
    print(f"   Total frames: {total_frames}")
    
    # Create output directory
    OUTPUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)
    
    # Create output video writer
    print(f"\n📝 Creating output video: {OUTPUT_VIDEO.name}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(OUTPUT_VIDEO), fourcc, fps, (width, height))
    
    if not out.isOpened():
        print("❌ ERROR: Cannot create output video")
        cap.release()
        return False
    
    print("\n🎬 Processing video frames...")
    
    frame_count = 0
    bounce_count = 0
    bounces_marked = []
    bounce_display_frames_left = 0
    last_marker_x, last_marker_y = None, None
    last_bounce_number = 0

    # Process each frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Get ball position for this frame
        ball_data = trajectory[trajectory['frame'] == frame_count]
        
        ball_x = None
        ball_y = None
        
        if len(ball_data) > 0:
            if not pd.isna(ball_data.iloc[0]['final_x']) and not pd.isna(ball_data.iloc[0]['final_y']):
                ball_x = int(ball_data.iloc[0]['final_x'])
                ball_y = int(ball_data.iloc[0]['final_y'])
                
                # Draw ball position (small green circle)
                cv2.circle(frame, (ball_x, ball_y), 6, (0, 255, 0), 2)
        
        is_bounce = (frame_count in BOUNCE_FRAMES)

        # Check if this is a BOUNCE frame (count only here)
        if is_bounce:
            bounce_count += 1
            bounces_marked.append(frame_count)

            # Use ball position if available, otherwise use frame center
            if ball_x is not None and ball_y is not None:
                marker_x = ball_x
                marker_y = ball_y
            else:
                # Fallback: use frame center if no ball detected
                marker_x = width // 2
                marker_y = height // 2

            last_marker_x, last_marker_y = marker_x, marker_y
            last_bounce_number = bounce_count
            bounce_display_frames_left = 15

        # Draw marker on bounce frame and subsequent 5 frames
        if is_bounce or bounce_display_frames_left > 0:

            if not is_bounce:
                marker_x, marker_y = last_marker_x, last_marker_y

            # Draw BOUNCE marker - large red circles
            cv2.circle(frame, (marker_x, marker_y), 50, (0, 0, 255), 6)  # Outer ring
            cv2.circle(frame, (marker_x, marker_y), 35, (0, 0, 255), 5)  # Middle ring
            cv2.circle(frame, (marker_x, marker_y), 20, (0, 0, 255), -1) # Inner filled circle

            # Draw "BOUNCE!" text
            text = f"BOUNCE #{last_bounce_number}"
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 1.8
            thickness = 5

            # Get text size
            (text_width, text_height), baseline = cv2.getTextSize(
                text, font, font_scale, thickness
            )

            # Position text above bounce marker
            text_x = marker_x - text_width // 2
            text_y = marker_y - 70

            # Ensure text stays within frame
            text_x = max(20, min(text_x, width - text_width - 20))
            text_y = max(text_height + 20, min(text_y, height - 20))

            # Draw semi-transparent black background for text
            overlay = frame.copy()
            cv2.rectangle(
                overlay,
                (text_x - 15, text_y - text_height - 15),
                (text_x + text_width + 15, text_y + baseline + 15),
                (0, 0, 0), -1
            )
            cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

            # Draw text with black outline
            cv2.putText(
                frame, text, (text_x - 3, text_y - 3),
                font, font_scale, (0, 0, 0), thickness + 4
            )

            # Draw text in red
            cv2.putText(
                frame, text, (text_x, text_y),
                font, font_scale, (0, 0, 255), thickness
            )

            # Show frame number below marker
            frame_text = f"Frame {frame_count}"
            cv2.putText(
                frame, frame_text, (marker_x - 70, marker_y + 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 3
            )

        if bounce_display_frames_left > 0 and (not is_bounce):
            bounce_display_frames_left -= 1

        
        # Add info overlay (top-left corner)
        # Black background box
        cv2.rectangle(frame, (10, 10), (300, 95), (0, 0, 0), -1)
        
        # Bounce counter
        counter_text = f"Bounces: {bounce_count}/{len(BOUNCE_FRAMES)}"
        cv2.putText(frame, counter_text, (20, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # Frame counter
        frame_info = f"Frame: {frame_count}/{total_frames}"
        cv2.putText(frame, frame_info, (20, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        # Write frame to output
        out.write(frame)
        
        # Progress indicator every 50 frames
        if frame_count % 50 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"  Progress: {frame_count}/{total_frames} ({progress:.1f}%) - Bounces marked: {bounce_count}")
    
    # Cleanup
    cap.release()
    out.release()
    
    # Final report
    print("\n" + "="*70)
    print("✅ VIDEO PROCESSING COMPLETE")
    print("="*70)
    print(f"\n📊 Statistics:")
    print(f"   Total frames processed: {frame_count}")
    print(f"   Bounces marked: {bounce_count}/{len(BOUNCE_FRAMES)}")
    
    print(f"\n🎾 Bounce frames marked:")
    for i, frame_num in enumerate(bounces_marked, 1):
        print(f"   {i}. Frame {frame_num}")
    
    # Verify all bounces were marked
    if bounces_marked == BOUNCE_FRAMES:
        print(f"\n✅ SUCCESS: All {len(BOUNCE_FRAMES)} bounces marked correctly!")
    else:
        print(f"\n⚠️ WARNING: Bounce frame mismatch!")
        print(f"   Expected: {BOUNCE_FRAMES}")
        print(f"   Marked: {bounces_marked}")
        missing = [f for f in BOUNCE_FRAMES if f not in bounces_marked]
        if missing:
            print(f"   Missing frames: {missing}")
    
    # Output file info
    if OUTPUT_VIDEO.exists():
        file_size_mb = OUTPUT_VIDEO.stat().st_size / (1024 * 1024)
        print(f"\n📁 Output video:")
        print(f"   Path: {OUTPUT_VIDEO}")
        print(f"   Size: {file_size_mb:.1f} MB")
    
    print("\n" + "="*70)
    print("You can now watch the video to verify bounces appear at correct frames")
    print("="*70 + "\n")
    
    return True


# ============================================================================
# EXECUTION
# ============================================================================

if __name__ == "__main__":
    success = create_bounce_video()
    
    if success:
        print("✅ Done! Check the output video:")
        print(f"   {OUTPUT_VIDEO}")
    else:
        print("❌ Failed to create video. Check error messages above.")

