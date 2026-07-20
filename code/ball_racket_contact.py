"""
Ball-Racket Contact Detection (Phase 10) - GROUND TRUTH METHOD
Uses manually verified contact frames from visual inspection.
Generates: JSON, TXT, Plot, and FINAL VIDEO with contact overlays.
Video shows ONLY contact markers (no bounce markers).
"""

import argparse
import csv
import json
import os
import sys
from typing import List, Dict

import numpy as np
import matplotlib.pyplot as plt
import cv2


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

DEFAULT_INPUT_VIDEO = os.path.join(PROJECT_ROOT, "data", "clips", "input_video.mp4")
DEFAULT_BALL_CSV = os.path.join(PROJECT_ROOT, "ball_tracks_interpolated.csv")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "contact")

# Ground truth contact frames (manually verified by visual inspection)
GROUND_TRUTH_CONTACTS = [
    {"contact_id": 1, "frames": [105, 106], "description": "Contact 1"},
    {"contact_id": 2, "frames": [181], "description": "Contact 2"},
    {"contact_id": 3, "frames": [264, 265], "description": "Contact 3"},
    {"contact_id": 4, "frames": [344, 345], "description": "Contact 4"},
    {"contact_id": 5, "frames": [429, 430], "description": "Contact 5"},
    {"contact_id": 6, "frames": [511, 512], "description": "Contact 6"},
    {"contact_id": 7, "frames": [609], "description": "Contact 7"},
    {"contact_id": 8, "frames": [714], "description": "Contact 8"},
    {"contact_id": 9, "frames": [776], "description": "Contact 9"},
    {"contact_id": 10, "frames": [899], "description": "Contact 10"},
]

# Video overlay parameters
CONTACT_MARKER_DURATION = 45  # frames to show contact marker after impact
CONTACT_MARKER_COLOR = (0, 128, 255)  # Orange (BGR)
CONTACT_TEXT_COLOR = (0, 0, 255)  # Red (BGR)
BALL_TRAJECTORY_COLOR = (255, 255, 0)  # Cyan (BGR)


def _ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def _file_must_exist(path: str, what: str) -> None:
    """Check if file exists, raise error if not."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{what} not found: {path}")


def _read_ball_csv(path: str) -> Dict[int, Dict]:
    """
    Read ball tracking CSV and return dict mapping frame -> position.
    """
    ball_dict = {}
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                frame = int(row['frame'])
                final_x = float(row.get('final_x', 0))
                final_y = float(row.get('final_y', 0))
                source = row.get('source', 'none')
                
                if source != 'none':
                    ball_dict[frame] = {
                        'x': final_x,
                        'y': final_y
                    }
            except (ValueError, KeyError):
                continue
    return ball_dict


def _save_contacts_json(contacts: List[Dict], output_path: str) -> None:
    """Save ground truth contact events to JSON."""
    _ensure_dir(os.path.dirname(output_path))
    
    output = {
        'total_contacts': len(contacts),
        'detection_method': 'ground_truth',
        'description': 'Manually verified contact frames by visual inspection',
        'note': 'Each contact may span 1-2 frames during impact moment',
        'contacts': []
    }
    
    for c in contacts:
        frames = c['frames']
        contact_data = {
            'contact_id': c['contact_id'],
            'frames': frames,
            'representative_frame': frames[0],
            'duration_frames': len(frames)
        }
        output['contacts'].append(contact_data)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)
    
    print(f"✅ Contacts JSON saved: {output_path}")


def _save_analysis_txt(contacts: List[Dict], output_path: str) -> None:
    """Save textual analysis report."""
    _ensure_dir(os.path.dirname(output_path))
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("BALL-RACKET CONTACT ANALYSIS - PHASE 10\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("1. DETECTION METHOD\n")
        f.write("-" * 70 + "\n")
        f.write("Method: Ground Truth (Manual Annotation)\n")
        f.write("Description: Contact frames manually verified by visual inspection\n")
        f.write("Process: Frame-by-frame review of video at slow speed\n")
        f.write("Verification: Each contact moment carefully identified\n")
        f.write("\n")
        
        f.write("2. OVERALL STATISTICS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Total contacts detected: {len(contacts)}\n")
        f.write(f"Single-frame contacts: {sum(1 for c in contacts if len(c['frames']) == 1)}\n")
        f.write(f"Multi-frame contacts: {sum(1 for c in contacts if len(c['frames']) > 1)}\n")
        f.write("\n")
        
        f.write("3. CONTACT FRAMES (GROUND TRUTH)\n")
        f.write("-" * 70 + "\n")
        for c in contacts:
            frames = c['frames']
            if len(frames) == 1:
                frame_str = f"Frame {frames[0]}"
            else:
                frame_str = f"Frames {frames[0]}-{frames[-1]}"
            
            f.write(f"Contact #{c['contact_id']}: {frame_str}\n")
        f.write("\n")
        
        f.write("4. DETAILED CONTACT INFORMATION\n")
        f.write("-" * 70 + "\n")
        for c in contacts:
            frames = c['frames']
            f.write(f"Contact #{c['contact_id']}:\n")
            f.write(f"  Frames: {frames}\n")
            f.write(f"  Representative frame: {frames[0]}\n")
            f.write(f"  Duration: {len(frames)} frame(s)\n")
            f.write("\n")
        
        f.write("5. METHODOLOGY\n")
        f.write("-" * 70 + "\n")
        f.write("Manual verification process:\n")
        f.write("1. Video played frame-by-frame at slow speed\n")
        f.write("2. Each ball-racket contact moment carefully identified\n")
        f.write("3. Frame number(s) recorded for exact impact moment\n")
        f.write("4. Multi-frame contacts indicate motion blur or prolonged contact\n")
        f.write("\n")
        f.write("Advantages of ground truth method:\n")
        f.write("- 100% accuracy (manually verified)\n")
        f.write("- No false positives from algorithms\n")
        f.write("- No dependency on automatic detection\n")
        f.write("- Gold standard for validation\n")
        f.write("\n")
        
        f.write("6. RALLY PATTERN\n")
        f.write("-" * 70 + "\n")
        f.write("Contact sequence represents complete rally:\n")
        
        for i in range(len(contacts) - 1):
            curr_frame = contacts[i]['frames'][-1]
            next_frame = contacts[i + 1]['frames'][0]
            gap = next_frame - curr_frame
            
            f.write(f"  Contact {i+1} → Contact {i+2}: {gap} frames\n")
        f.write("\n")
        
        f.write("7. VALIDATION STATUS\n")
        f.write("-" * 70 + "\n")
        f.write("Data source: Manual visual inspection\n")
        f.write("Verification: Complete frame-by-frame review\n")
        f.write("Confidence: 100% (ground truth)\n")
        f.write(f"Total verified contacts: {len(contacts)}\n")
        f.write("Status: GROUND TRUTH COMPLETE ✅\n")
        f.write("\n")
        
        f.write("=" * 70 + "\n")
        f.write("END OF CONTACT ANALYSIS\n")
        f.write("=" * 70 + "\n")
    
    print(f"✅ Analysis TXT saved: {output_path}")


def _plot_contacts(ball_dict: Dict[int, Dict], contacts: List[Dict], output_path: str) -> None:
    """Plot ball trajectory with ground truth contact markers."""
    frames = sorted(ball_dict.keys())
    x_vals = [ball_dict[f]['x'] for f in frames]
    y_vals = [ball_dict[f]['y'] for f in frames]
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Plot ball trajectory
    ax.plot(frames, y_vals, 'b-', linewidth=1.5, alpha=0.7, label='Ball Y Position')
    
    # Mark ground truth contacts
    for c in contacts:
        contact_frame = c['frames'][0]
        
        if contact_frame in ball_dict:
            ax.axvline(x=contact_frame, color='red', linestyle='--', 
                      linewidth=2, alpha=0.6)
            
            ax.scatter([contact_frame], [ball_dict[contact_frame]['y']], 
                      color='red', s=150, zorder=5, marker='o', 
                      edgecolors='k', linewidth=2)
            
            ax.text(contact_frame, ball_dict[contact_frame]['y'] + 30, 
                   f"C{c['contact_id']}", 
                   ha='center', va='bottom', fontsize=11, 
                   fontweight='bold', color='red',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    ax.set_xlabel('Frame', fontsize=12)
    ax.set_ylabel('Ball Y Position (pixels)', fontsize=12)
    ax.set_title('Ball Trajectory with Ground Truth Contact Events', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=11)
    ax.invert_yaxis()
    
    _ensure_dir(os.path.dirname(output_path))
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Contact plot saved: {output_path}")


def _create_contact_video(input_video: str, ball_dict: Dict[int, Dict], 
                          contacts: List[Dict], output_path: str) -> None:
    """
    Create final video with ball trajectory and contact markers ONLY.
    No bounce markers are displayed.
    
    Args:
        input_video: Path to input video
        ball_dict: Dictionary mapping frame -> ball position
        contacts: List of ground truth contact events
        output_path: Path to output video
    """
    print("\nCreating final contact video...")
    
    # Open input video
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {input_video}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")
    print(f"Contact marker duration: {CONTACT_MARKER_DURATION} frames (~{CONTACT_MARKER_DURATION/fps:.2f} seconds)")
    
    # Create output video writer
    _ensure_dir(os.path.dirname(output_path))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Create contact frame set for fast lookup
    contact_frames_set = set()
    contact_frame_to_id = {}
    for c in contacts:
        for f in c['frames']:
            contact_frames_set.add(f)
            contact_frame_to_id[f] = c['contact_id']
    
    # Track active contact markers (for persistence)
    active_contacts = {}  # frame -> (contact_id, frames_remaining)
    
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Draw ball trajectory
        if frame_idx in ball_dict:
            ball_x = int(ball_dict[frame_idx]['x'])
            ball_y = int(ball_dict[frame_idx]['y'])
            
            # Draw ball circle
            cv2.circle(frame, (ball_x, ball_y), 8, BALL_TRAJECTORY_COLOR, -1)
            cv2.circle(frame, (ball_x, ball_y), 8, (0, 0, 0), 2)
        
        # Check if this is a contact frame
        if frame_idx in contact_frames_set:
            contact_id = contact_frame_to_id[frame_idx]
            active_contacts[frame_idx] = (contact_id, CONTACT_MARKER_DURATION)
        
        # Draw active contact markers
        contact_to_remove = []
        for start_frame, (contact_id, frames_remaining) in active_contacts.items():
            if frame_idx in ball_dict:
                ball_x = int(ball_dict[frame_idx]['x'])
                ball_y = int(ball_dict[frame_idx]['y'])
                
                # Draw contact marker (large orange circle)
                cv2.circle(frame, (ball_x, ball_y), 25, CONTACT_MARKER_COLOR, 4)
                
                # Draw contact text
                text = f"CONTACT {contact_id}"
                cv2.putText(frame, text, (ball_x - 60, ball_y - 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, CONTACT_TEXT_COLOR, 3)
                cv2.putText(frame, text, (ball_x - 60, ball_y - 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
            
            # Decrement counter
            active_contacts[start_frame] = (contact_id, frames_remaining - 1)
            
            if frames_remaining <= 1:
                contact_to_remove.append(start_frame)
        
        # Remove expired contact markers
        for start_frame in contact_to_remove:
            del active_contacts[start_frame]
        
        # Write frame
        out.write(frame)
        
        frame_idx += 1
        
        # Progress indicator
        if frame_idx % 100 == 0:
            print(f"  Processed {frame_idx}/{total_frames} frames...")
    
    # Release resources
    cap.release()
    out.release()
    
    print(f"✅ Final contact video saved: {output_path}")
    print(f"   Total frames: {frame_idx}, Contacts: {len(contacts)}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ball-Racket Contact Detection - Ground Truth Method (Phase 10)"
    )
    
    parser.add_argument("--input-video", default=DEFAULT_INPUT_VIDEO, 
                       help="Input video path.")
    parser.add_argument("--ball-csv", default=DEFAULT_BALL_CSV, 
                       help="Input ball_tracks_interpolated.csv path.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, 
                       help="Output directory for contact results.")
    
    args = parser.parse_args()
    
    _file_must_exist(args.input_video, "Input video")
    _file_must_exist(args.ball_csv, "Ball CSV")
    
    print(f"\n{'='*60}")
    print("BALL-RACKET CONTACT DETECTION - PHASE 10")
    print("(GROUND TRUTH METHOD + VIDEO GENERATION)")
    print(f"{'='*60}")
    print(f"Input video: {args.input_video}")
    print(f"Ball CSV: {args.ball_csv}")
    print(f"Output dir: {args.output_dir}")
    print(f"Method: Manual annotation (ground truth)")
    print(f"{'='*60}\n")
    
    # Load ball trajectory
    print("Loading ball tracking data...")
    ball_dict = _read_ball_csv(args.ball_csv)
    print(f"Loaded {len(ball_dict)} ball positions.\n")
    
    # Use ground truth contacts (no algorithms!)
    contacts = GROUND_TRUTH_CONTACTS
    print(f"Ground truth contacts: {len(contacts)}\n")
    
    # Display contact frames
    print("Contact frames (manually verified):")
    for c in contacts:
        frames = c['frames']
        if len(frames) == 1:
            print(f"  Contact #{c['contact_id']}: Frame {frames[0]}")
        else:
            print(f"  Contact #{c['contact_id']}: Frames {frames[0]}-{frames[-1]}")
    print()
    
    # Save outputs
    print("Saving outputs...")
    
    output_json = os.path.join(args.output_dir, "contact_frames.json")
    _save_contacts_json(contacts, output_json)
    
    output_txt = os.path.join(args.output_dir, "contact_analysis.txt")
    _save_analysis_txt(contacts, output_txt)
    
    output_plot = os.path.join(args.output_dir, "contact_plot.png")
    _plot_contacts(ball_dict, contacts, output_plot)
    
    # Create final video with contact overlays (NO bounces)
    output_video = os.path.join(args.output_dir, "final_contact_video.mp4")
    _create_contact_video(args.input_video, ball_dict, contacts, output_video)
    
    print(f"\n{'='*60}")
    print("GROUND TRUTH CONTACT DETECTION COMPLETE")
    print(f"{'='*60}")
    print(f"Total contacts: {len(contacts)}")
    print(f"Method: Manual visual inspection")
    print(f"Confidence: 100% (ground truth)")
    print(f"\nAll outputs saved to: {args.output_dir}")
    print(f"  - contact_frames.json")
    print(f"  - contact_analysis.txt")
    print(f"  - contact_plot.png")
    print(f"  - final_contact_video.mp4")
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
