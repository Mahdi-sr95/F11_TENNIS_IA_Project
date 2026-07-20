#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================
ball_track_with_minicourt.py - FINAL VERSION
Tennis ball tracking with mini court visualization
============================================================
"""

import os
import json
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# ============================================================
# PATHS
# ============================================================

VIDEO_PATH = os.path.join("data", "input_video.mp4")
CALIB_DIR = os.path.join("results", "intermediate", "court_calib")
BALL_TRACKS_CSV = os.path.join("results", "intermediate", "ball_tracks.csv")
COURT_JSON = os.path.join(CALIB_DIR, "court_points.json")
H_IMG2COURT = os.path.join(CALIB_DIR, "H_img2court.npy")
H_COURT2IMG = os.path.join(CALIB_DIR, "H_court2img.npy")
OUTPUT_DIR = os.path.join("results", "final")
OUTPUT_VIDEO = os.path.join(OUTPUT_DIR, "output_with_minicourt.mp4")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("🎾 TENNIS BALL TRACKING WITH MINI COURT")
print("=" * 70)

# Load calibration
print(f"\n📂 Loading calibration: {COURT_JSON}")
with open(COURT_JSON, 'r') as f:
    calib_data = json.load(f)

# Fix: Support both "court_dimensions" and "court_dimensions_m"
if "court_dimensions_m" in calib_data:
    dims = calib_data["court_dimensions_m"]
elif "court_dimensions" in calib_data:
    dims = calib_data["court_dimensions"]
else:
    # Default ITF values
    dims = {
        "length": 23.77,
        "width_doubles": 10.97,
        "width_singles": 8.23,
        "service_line_distance": 6.40
    }

COURT_LENGTH = dims["length"]
COURT_WIDTH = dims["width_doubles"]
SERVICE_DIST = dims["service_line_distance"]

print(f"✓ Court dimensions: {COURT_WIDTH}m x {COURT_LENGTH}m")

# Load homography
print(f"📂 Loading homography: {H_IMG2COURT}")
H_img2court = np.load(H_IMG2COURT)
print(f"✓ H_img2court shape: {H_img2court.shape}")

print(f"📂 Loading inverse homography: {H_COURT2IMG}")
H_court2img = np.load(H_COURT2IMG)
print(f"✓ H_court2img shape: {H_court2img.shape}")

# Load ball tracks
print(f"📂 Loading ball tracks: {BALL_TRACKS_CSV}")
if not os.path.exists(BALL_TRACKS_CSV):
    raise FileNotFoundError(
        f"❌ Ball tracks not found!\n"
        f"   Please run: python code/ball_track.py first\n"
        f"   Missing file: {BALL_TRACKS_CSV}"
    )

df_ball = pd.read_csv(BALL_TRACKS_CSV)
print(f"✓ Loaded {len(df_ball)} frames with ball data")

# ============================================================
# MINI COURT SETTINGS
# ============================================================

MINI_WIDTH = 300
MINI_HEIGHT = int(MINI_WIDTH * (COURT_LENGTH / COURT_WIDTH))
MINI_PADDING = 20
MINI_X = 50
MINI_Y = 50

print(f"\n🎨 Mini court size: {MINI_WIDTH}x{MINI_HEIGHT} pixels")

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def world_to_mini(x_world, y_world):
    """Convert world coordinates (meters) to mini court pixels"""
    x_mini = int((x_world / COURT_WIDTH) * MINI_WIDTH)
    y_mini = int((y_world / COURT_LENGTH) * MINI_HEIGHT)
    return x_mini, y_mini

def draw_mini_court(canvas):
    """Draw tennis court on mini canvas"""
    # Background
    canvas[:] = (50, 120, 50)  # Green
    
    # Outer rectangle (doubles)
    cv2.rectangle(canvas, (0, 0), (MINI_WIDTH-1, MINI_HEIGHT-1), (255, 255, 255), 2)
    
    # Singles sidelines
    singles_margin = (COURT_WIDTH - 8.23) / 2.0
    singles_x_left = int((singles_margin / COURT_WIDTH) * MINI_WIDTH)
    singles_x_right = int(((COURT_WIDTH - singles_margin) / COURT_WIDTH) * MINI_WIDTH)
    
    cv2.line(canvas, (singles_x_left, 0), (singles_x_left, MINI_HEIGHT), (255, 255, 255), 1)
    cv2.line(canvas, (singles_x_right, 0), (singles_x_right, MINI_HEIGHT), (255, 255, 255), 1)
    
    # Service lines
    service_y_near = int((SERVICE_DIST / COURT_LENGTH) * MINI_HEIGHT)
    service_y_far = int(((COURT_LENGTH - SERVICE_DIST) / COURT_LENGTH) * MINI_HEIGHT)
    
    cv2.line(canvas, (0, service_y_near), (MINI_WIDTH, service_y_near), (255, 255, 255), 1)
    cv2.line(canvas, (0, service_y_far), (MINI_WIDTH, service_y_far), (255, 255, 255), 1)
    
    # Net line
    net_y = MINI_HEIGHT // 2
    cv2.line(canvas, (0, net_y), (MINI_WIDTH, net_y), (255, 255, 255), 2)
    
    # Center service line
    center_x = MINI_WIDTH // 2
    cv2.line(canvas, (center_x, service_y_near), (center_x, service_y_far), (255, 255, 255), 1)
    
    return canvas

def image_to_world(x_img, y_img):
    """Convert image pixel to world coordinates (meters)"""
    pt_img = np.array([x_img, y_img, 1.0])
    pt_world = H_img2court @ pt_img
    pt_world /= pt_world[2]
    return pt_world[0], pt_world[1]

# ============================================================
# PROCESS VIDEO
# ============================================================

print(f"\n🎬 Opening video: {VIDEO_PATH}")
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise IOError(f"Cannot open video: {VIDEO_PATH}")

fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"✓ Video info: {width}x{height} @ {fps} FPS, {total_frames} frames")

# Setup video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

print(f"\n💾 Output video: {OUTPUT_VIDEO}")
print(f"\n🔄 Processing frames...")

# Ball trail storage
ball_trail = []
MAX_TRAIL_LENGTH = 30

# Process frames
frame_idx = 0
pbar = tqdm(total=total_frames, desc="Rendering", unit="frame")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Get ball position for this frame
    ball_row = df_ball[df_ball['frame'] == frame_idx]
    
    # Create mini court
    mini_court = np.zeros((MINI_HEIGHT, MINI_WIDTH, 3), dtype=np.uint8)
    draw_mini_court(mini_court)
    
    # Draw ball on mini court
    if not ball_row.empty and ball_row.iloc[0]['visible'] == 1:
        x_img = ball_row.iloc[0]['x']
        y_img = ball_row.iloc[0]['y']
        
        # Convert to world coordinates
        x_world, y_world = image_to_world(x_img, y_img)
        
        # Clamp to court bounds
        x_world = np.clip(x_world, 0, COURT_WIDTH)
        y_world = np.clip(y_world, 0, COURT_LENGTH)
        
        # Convert to mini court pixels
        x_mini, y_mini = world_to_mini(x_world, y_world)
        
        # Add to trail
        ball_trail.append((x_mini, y_mini))
        if len(ball_trail) > MAX_TRAIL_LENGTH:
            ball_trail.pop(0)
        
        # Draw trail
        for i in range(1, len(ball_trail)):
            alpha = i / len(ball_trail)
            thickness = int(1 + alpha * 2)
            color = (0, int(100 + alpha * 155), int(100 + alpha * 155))
            cv2.line(mini_court, ball_trail[i-1], ball_trail[i], color, thickness)
        
        # Draw current ball
        cv2.circle(mini_court, (x_mini, y_mini), 5, (0, 255, 255), -1)
        cv2.circle(mini_court, (x_mini, y_mini), 7, (255, 255, 255), 1)
        
        # Draw ball on main frame
        cv2.circle(frame, (int(x_img), int(y_img)), 8, (0, 255, 255), -1)
        cv2.circle(frame, (int(x_img), int(y_img)), 12, (255, 255, 255), 2)
    
    # Create semi-transparent overlay for mini court background
    overlay = frame.copy()
    x1 = MINI_X - MINI_PADDING
    y1 = MINI_Y - MINI_PADDING
    x2 = MINI_X + MINI_WIDTH + MINI_PADDING
    y2 = MINI_Y + MINI_HEIGHT + MINI_PADDING
    
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Add white border
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
    
    # Overlay mini court
    frame[MINI_Y:MINI_Y+MINI_HEIGHT, MINI_X:MINI_X+MINI_WIDTH] = mini_court
    
    # Add label
    cv2.putText(frame, "COURT VIEW", (MINI_X, MINI_Y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Add frame counter
    cv2.putText(frame, f"Frame: {frame_idx}/{total_frames}", (width - 250, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Write frame
    out.write(frame)
    
    frame_idx += 1
    pbar.update(1)

pbar.close()
cap.release()
out.release()

print("\n" + "=" * 70)
print("✅ VIDEO PROCESSING COMPLETE!")
print("=" * 70)
print(f"\n📹 Output saved to: {OUTPUT_VIDEO}")
print(f"📊 Processed {frame_idx} frames")
print(f"\n🎬 You can now play the video with mini court visualization!\n")
