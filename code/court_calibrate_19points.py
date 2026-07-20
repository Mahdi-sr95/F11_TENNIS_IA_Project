#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================
court_calibrate_19points.py - Tennis Court Calibration
============================================================
Full 19-point calibration for tennis court
- 7 points on near half
- 7 points on far half
- 5 points on net line
"""

import os
import json
import glob
import numpy as np
import cv2

# ============================================================
# CONFIGURATION
# ============================================================

FRAMES_DIR = os.path.join("data", "frames")
OUT_DIR = os.path.join("results", "intermediate", "court_calib")
os.makedirs(OUT_DIR, exist_ok=True)

# Tennis court dimensions (ITF standard - meters)
COURT_LENGTH = 23.77
COURT_WIDTH = 10.97      # Doubles
SINGLES_WIDTH = 8.23
SERVICE_LINE_DIST = 6.40
NET_Y = COURT_LENGTH / 2.0

# Derived values
SINGLES_SIDELINE_X = (COURT_WIDTH - SINGLES_WIDTH) / 2.0
CENTER_X = COURT_WIDTH / 2.0

# ============================================================
# WORLD COORDINATES (19 POINTS)
# ============================================================

KEYPOINTS_WORLD = {
    # NEAR HALF (7 points)
    "near_baseline_left": [0.0, 0.0],
    "near_baseline_right": [COURT_WIDTH, 0.0],
    "near_singles_left": [SINGLES_SIDELINE_X, 0.0],
    "near_singles_right": [COURT_WIDTH - SINGLES_SIDELINE_X, 0.0],
    "near_service_left": [0.0, SERVICE_LINE_DIST],
    "near_service_right": [COURT_WIDTH, SERVICE_LINE_DIST],
    "near_service_center": [CENTER_X, SERVICE_LINE_DIST],
    
    # FAR HALF (7 points)
    "far_baseline_left": [0.0, COURT_LENGTH],
    "far_baseline_right": [COURT_WIDTH, COURT_LENGTH],
    "far_singles_left": [SINGLES_SIDELINE_X, COURT_LENGTH],
    "far_singles_right": [COURT_WIDTH - SINGLES_SIDELINE_X, COURT_LENGTH],
    "far_service_left": [0.0, COURT_LENGTH - SERVICE_LINE_DIST],
    "far_service_right": [COURT_WIDTH, COURT_LENGTH - SERVICE_LINE_DIST],
    "far_service_center": [CENTER_X, COURT_LENGTH - SERVICE_LINE_DIST],
    
    # NET LINE (5 points)
    "net_left": [0.0, NET_Y],
    "net_singles_left": [SINGLES_SIDELINE_X, NET_Y],
    "net_center": [CENTER_X, NET_Y],
    "net_singles_right": [COURT_WIDTH - SINGLES_SIDELINE_X, NET_Y],
    "net_right": [COURT_WIDTH, NET_Y],
}

# Ordered list of all 19 points
POINT_ORDER = [
    # Near half
    "near_baseline_left",
    "near_baseline_right",
    "near_singles_left",
    "near_singles_right",
    "near_service_left",
    "near_service_right",
    "near_service_center",
    # Far half
    "far_baseline_left",
    "far_baseline_right",
    "far_singles_left",
    "far_singles_right",
    "far_service_left",
    "far_service_right",
    "far_service_center",
    # Net
    "net_left",
    "net_singles_left",
    "net_center",
    "net_singles_right",
    "net_right",
]

# ============================================================
# POINT DESCRIPTIONS
# ============================================================

POINT_DESCRIPTIONS = {
    "near_baseline_left": "LEFT corner of NEAR baseline (doubles line)",
    "near_baseline_right": "RIGHT corner of NEAR baseline (doubles line)",
    "near_singles_left": "LEFT corner of NEAR baseline (singles line)",
    "near_singles_right": "RIGHT corner of NEAR baseline (singles line)",
    "near_service_left": "LEFT corner of NEAR service line",
    "near_service_right": "RIGHT corner of NEAR service line",
    "near_service_center": "CENTER mark on NEAR service line",
    
    "far_baseline_left": "LEFT corner of FAR baseline (doubles line)",
    "far_baseline_right": "RIGHT corner of FAR baseline (doubles line)",
    "far_singles_left": "LEFT corner of FAR baseline (singles line)",
    "far_singles_right": "RIGHT corner of FAR baseline (singles line)",
    "far_service_left": "LEFT corner of FAR service line",
    "far_service_right": "RIGHT corner of FAR service line",
    "far_service_center": "CENTER mark on FAR service line",
    
    "net_left": "NET line at LEFT doubles sideline",
    "net_singles_left": "NET line at LEFT singles sideline",
    "net_center": "CENTER mark on NET line",
    "net_singles_right": "NET line at RIGHT singles sideline",
    "net_right": "NET line at RIGHT doubles sideline",
}

# ============================================================
# COURT LINES TO DRAW
# ============================================================

COURT_LINES = [
    # Outer rectangle (doubles)
    ["near_baseline_left", "near_baseline_right"],
    ["near_baseline_right", "far_baseline_right"],
    ["far_baseline_right", "far_baseline_left"],
    ["far_baseline_left", "near_baseline_left"],
    
    # Singles sidelines
    ["near_singles_left", "far_singles_left"],
    ["near_singles_right", "far_singles_right"],
    
    # Service lines
    ["near_service_left", "near_service_right"],
    ["far_service_left", "far_service_right"],
    
    # Center line (service boxes)
    ["near_service_center", "far_service_center"],
    
    # Net line
    ["net_left", "net_right"],
]

# ============================================================
# FIND FRAME
# ============================================================

frame_files = sorted(glob.glob(os.path.join(FRAMES_DIR, "*.jpg"))) + \
              sorted(glob.glob(os.path.join(FRAMES_DIR, "*.png")))

if not frame_files:
    raise FileNotFoundError(f"No frames found in {FRAMES_DIR}")

IMG_PATH = frame_files[0]
print(f"\n{'='*70}")
print(f"🎾 TENNIS COURT CALIBRATION - 19 Point Mode")
print(f"{'='*70}")
print(f"📸 Using frame: {IMG_PATH}\n")

# ============================================================
# LOAD IMAGE
# ============================================================

img = cv2.imread(IMG_PATH)
if img is None:
    raise FileNotFoundError(f"Cannot read: {IMG_PATH}")

H_IMG, W_IMG = img.shape[:2]
print(f"📐 Image size: {W_IMG} x {H_IMG}\n")

# ============================================================
# INSTRUCTION
# ============================================================

print("📋 You will click 19 keypoints in this order:")
print()
for i, name in enumerate(POINT_ORDER, 1):
    section = ""
    if i <= 7:
        section = "[NEAR HALF]"
    elif i <= 14:
        section = "[FAR HALF]"
    else:
        section = "[NET LINE]"
    print(f"  {i:2d}. {section:12} {POINT_DESCRIPTIONS[name]}")

print("\n⌨️  Controls:")
print("  - Left Click: Add point")
print("  - [u]: Undo last point")
print("  - [r]: Reset all")
print("  - [ESC] or [q]: Finish and save")
print("\n🚀 Starting in 2 seconds...")

# ============================================================
# PREVIEW
# ============================================================

preview = img.copy()
cv2.putText(preview, "COURT CALIBRATION - Press any key to start", 
            (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
cv2.namedWindow("Court Calibration", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Court Calibration", 1280, 720)
cv2.imshow("Court Calibration", preview)
cv2.waitKey(2000)

# ============================================================
# GLOBAL STATE
# ============================================================

clicks = []
window_name = "Court Calibration"

# ============================================================
# HELPER: DRAW MINI DIAGRAM
# ============================================================

def draw_mini_diagram(canvas, current_point_name):
    """Draw a miniature court showing which point to click"""
    mini_w = 220
    mini_h = int(mini_w * (COURT_LENGTH / COURT_WIDTH))
    margin = 20
    
    x_start = canvas.shape[1] - mini_w - margin - 10
    y_start = margin + 140
    
    # Semi-transparent background
    overlay = canvas.copy()
    cv2.rectangle(overlay, 
                  (x_start - 10, y_start - 10),
                  (x_start + mini_w + 10, y_start + mini_h + 10),
                  (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)
    
    # Outer rectangle
    cv2.rectangle(canvas, (x_start, y_start), 
                  (x_start + mini_w, y_start + mini_h), (180, 180, 180), 2)
    
    # Singles lines
    singles_x_left = int(x_start + (SINGLES_SIDELINE_X / COURT_WIDTH) * mini_w)
    singles_x_right = int(x_start + ((COURT_WIDTH - SINGLES_SIDELINE_X) / COURT_WIDTH) * mini_w)
    cv2.line(canvas, (singles_x_left, y_start), (singles_x_left, y_start + mini_h), (120, 120, 120), 1)
    cv2.line(canvas, (singles_x_right, y_start), (singles_x_right, y_start + mini_h), (120, 120, 120), 1)
    
    # Service lines
    service_y_near = int(y_start + (SERVICE_LINE_DIST / COURT_LENGTH) * mini_h)
    service_y_far = int(y_start + ((COURT_LENGTH - SERVICE_LINE_DIST) / COURT_LENGTH) * mini_h)
    cv2.line(canvas, (x_start, service_y_near), (x_start + mini_w, service_y_near), (140, 140, 140), 1)
    cv2.line(canvas, (x_start, service_y_far), (x_start + mini_w, service_y_far), (140, 140, 140), 1)
    
    # Net
    net_y = int(y_start + 0.5 * mini_h)
    cv2.line(canvas, (x_start, net_y), (x_start + mini_w, net_y), (140, 140, 140), 1)
    
    # Center line
    center_x = int(x_start + 0.5 * mini_w)
    cv2.line(canvas, (center_x, service_y_near), (center_x, service_y_far), (100, 100, 100), 1)
    
    # Highlight current point
    if current_point_name in KEYPOINTS_WORLD:
        world_pt = KEYPOINTS_WORLD[current_point_name]
        px = int(x_start + (world_pt[0] / COURT_WIDTH) * mini_w)
        py = int(y_start + (world_pt[1] / COURT_LENGTH) * mini_h)
        
        cv2.circle(canvas, (px, py), 7, (0, 255, 255), -1)
        cv2.circle(canvas, (px, py), 11, (0, 255, 0), 2)
        cv2.arrowedLine(canvas, (px + 25, py - 15), (px + 6, py - 4), 
                        (0, 255, 0), 2, tipLength=0.4)

# ============================================================
# HELPER: DRAW UI
# ============================================================

def draw_ui():
    """Draw the calibration UI"""
    canvas = img.copy()
    
    n_total = len(POINT_ORDER)
    n_done = len(clicks)
    
    # Dark header
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (canvas.shape[1], 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, canvas, 0.35, 0, canvas)
    
    if n_done < n_total:
        current_point = POINT_ORDER[n_done]
        
        # Section label
        if n_done < 7:
            section = "[NEAR HALF]"
            section_color = (0, 200, 255)
        elif n_done < 14:
            section = "[FAR HALF]"
            section_color = (255, 150, 0)
        else:
            section = "[NET LINE]"
            section_color = (0, 255, 100)
        
        cv2.putText(canvas, section, (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, section_color, 2, cv2.LINE_AA)
        
        # Main instruction
        msg = f"POINT {n_done+1}/{n_total}: {POINT_DESCRIPTIONS[current_point]}"
        cv2.putText(canvas, msg, (20, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
        
        # Progress bar
        bar_w = 700
        bar_h = 18
        bar_x = 20
        bar_y = 75
        progress = n_done / n_total
        
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), 2)
        if progress > 0:
            cv2.rectangle(canvas, (bar_x, bar_y), 
                          (bar_x + int(bar_w * progress), bar_y + bar_h), (0, 200, 0), -1)
        
        # Controls
        cv2.putText(canvas, "Controls: [u]=Undo [r]=Reset [ESC/q]=Finish", 
                    (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
        
        # Mini diagram
        draw_mini_diagram(canvas, current_point)
        
    else:
        cv2.putText(canvas, "✓ ALL 19 POINTS MARKED! Press ESC or 'q' to save", 
                    (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Draw clicked points
    for i, (x, y) in enumerate(clicks):
        # Determine color based on section
        if i < 7:
            color = (255, 100, 0)  # Blue-ish for near
        elif i < 14:
            color = (0, 100, 255)  # Orange-ish for far
        else:
            color = (0, 255, 0)  # Green for net
        
        # Circle
        cv2.circle(canvas, (int(x), int(y)), 8, color, -1)
        cv2.circle(canvas, (int(x), int(y)), 12, (255, 255, 255), 2)
        
        # Label
        label = str(i + 1)
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        label_x = int(x) + 18
        label_y = int(y) - 18
        
        cv2.rectangle(canvas,
                      (label_x - 3, label_y - label_size[1] - 3),
                      (label_x + label_size[0] + 3, label_y + 3),
                      (0, 0, 0), -1)
        cv2.putText(canvas, label, (label_x, label_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        
        # Connect consecutive points
        if i > 0:
            prev_x, prev_y = clicks[i - 1]
            cv2.line(canvas, (int(prev_x), int(prev_y)), (int(x), int(y)), 
                     (200, 100, 0), 2)
    
    return canvas

# ============================================================
# MOUSE CALLBACK
# ============================================================

def mouse_callback(event, x, y, flags, param):
    """Handle mouse events"""
    global clicks
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicks) < len(POINT_ORDER):
            clicks.append((x, y))
            print(f"✓ Point {len(clicks)}/{len(POINT_ORDER)}: {POINT_ORDER[len(clicks)-1]} at ({x}, {y})")
            
            canvas = draw_ui()
            cv2.imshow(window_name, canvas)

# ============================================================
# MAIN CALIBRATION LOOP
# ============================================================

cv2.setMouseCallback(window_name, mouse_callback)

while True:
    canvas = draw_ui()
    cv2.imshow(window_name, canvas)
    
    key = cv2.waitKey(50) & 0xFF
    
    if key == 27 or key == ord('q'):  # ESC or 'q'
        if len(clicks) == len(POINT_ORDER):
            print("\n✅ Calibration complete!")
            break
        else:
            print(f"\n⚠️  Only {len(clicks)}/{len(POINT_ORDER)} points marked. Continue? (y/n): ", end="")
            response = input().strip().lower()
            if response == 'y':
                break
            else:
                continue
    
    elif key == ord('u'):  # Undo
        if clicks:
            removed = clicks.pop()
            print(f"↶ Undo: Removed point {len(clicks)+1}")
    
    elif key == ord('r'):  # Reset
        clicks = []
        print("⟲ Reset: All points cleared")

cv2.destroyAllWindows()

# ============================================================
# SAVE RESULTS
# ============================================================

if len(clicks) == 0:
    print("\n❌ No points marked. Exiting without saving.")
    exit(0)

print(f"\n{'='*70}")
print("💾 Saving calibration data...")
print(f"{'='*70}\n")

# Prepare data
points_image = {name: list(clicks[i]) for i, name in enumerate(POINT_ORDER[:len(clicks)])}
points_world = {name: KEYPOINTS_WORLD[name] for name in POINT_ORDER[:len(clicks)]}

calibration_data = {
    "image_path": IMG_PATH,
    "image_size": [W_IMG, H_IMG],
    "num_points": len(clicks),
    "points_image": points_image,
    "points_world": points_world,
    "court_dimensions": {
        "length": COURT_LENGTH,
        "width_doubles": COURT_WIDTH,
        "width_singles": SINGLES_WIDTH,
        "service_line_distance": SERVICE_LINE_DIST,
    }
}

# Save JSON
json_path = os.path.join(OUT_DIR, "court_points.json")
with open(json_path, 'w') as f:
    json.dump(calibration_data, f, indent=2)
print(f"✓ Saved: {json_path}")

# Compute homography if we have enough points
if len(clicks) >= 4:
    pts_img = np.array([clicks[i] for i in range(len(clicks))], dtype=np.float32)
    pts_world = np.array([KEYPOINTS_WORLD[POINT_ORDER[i]] for i in range(len(clicks))], dtype=np.float32)
    
    H, mask = cv2.findHomography(pts_img, pts_world, cv2.RANSAC, 5.0)
    
    # Save homography
    H_path = os.path.join(OUT_DIR, "homography.npy")
    np.save(H_path, H)
    print(f"✓ Saved: {H_path}")
    
    # Draw overlay
    overlay = img.copy()
    
    # Draw all lines
    for line in COURT_LINES:
        if line[0] in points_image and line[1] in points_image:
            pt1 = tuple(map(int, points_image[line[0]]))
            pt2 = tuple(map(int, points_image[line[1]]))
            cv2.line(overlay, pt1, pt2, (0, 255, 0), 3)
    
    # Draw points
    for i, (x, y) in enumerate(clicks):
        if i < 7:
            color = (255, 100, 0)
        elif i < 14:
            color = (0, 100, 255)
        else:
            color = (0, 255, 0)
        cv2.circle(overlay, (int(x), int(y)), 10, color, -1)
        cv2.circle(overlay, (int(x), int(y)), 14, (255, 255, 255), 2)
    
    overlay_path = os.path.join(OUT_DIR, "debug_overlay_complete.png")
    cv2.imwrite(overlay_path, overlay)
    print(f"✓ Saved: {overlay_path}")
    
    # Create top-down view
    top_w = 600
    top_h = int(top_w * (COURT_LENGTH / COURT_WIDTH))
    
    topdown = np.zeros((top_h, top_w, 3), dtype=np.uint8)
    topdown[:] = (50, 120, 50)  # Green background
    
    # Warp image
    H_inv = np.linalg.inv(H)
    
    for py in range(top_h):
        for px in range(top_w):
            wx = (px / top_w) * COURT_WIDTH
            wy = (py / top_h) * COURT_LENGTH
            
            img_pt = H_inv @ np.array([wx, wy, 1])
            img_pt /= img_pt[2]
            
            ix, iy = int(img_pt[0]), int(img_pt[1])
            if 0 <= ix < W_IMG and 0 <= iy < H_IMG:
                topdown[py, px] = img[iy, ix]
    
    # Draw lines on top-down
    for line in COURT_LINES:
        if line[0] in points_world and line[1] in points_world:
            pt1_w = points_world[line[0]]
            pt2_w = points_world[line[1]]
            
            pt1_t = (int((pt1_w[0] / COURT_WIDTH) * top_w),
                     int((pt1_w[1] / COURT_LENGTH) * top_h))
            pt2_t = (int((pt2_w[0] / COURT_WIDTH) * top_w),
                     int((pt2_w[1] / COURT_LENGTH) * top_h))
            
            cv2.line(topdown, pt1_t, pt2_t, (255, 255, 255), 2)
    
    topdown_path = os.path.join(OUT_DIR, "topdown.jpg")
    cv2.imwrite(topdown_path, topdown)
    print(f"✓ Saved: {topdown_path}")

print(f"\n{'='*70}")
print("✅ CALIBRATION COMPLETE!")
print(f"{'='*70}\n")
print(f"📂 Output directory: {OUT_DIR}/")
print(f"   - court_points.json")
if len(clicks) >= 4:
    print(f"   - homography.npy")
    print(f"   - debug_overlay_complete.png")
    print(f"   - topdown.jpg")
print()
