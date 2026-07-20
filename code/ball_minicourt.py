#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ball to Court Projection and Mini-Court Visualization
Projects ball positions onto court coordinates and creates mini-court overlay
"""

import json
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from collections import deque


class MiniCourtVisualizer:
    """Project ball to court and visualize on mini-court"""
    
    def __init__(self, court_json_path, ball_csv_path, video_path):
        """
        Args:
            court_json_path: Path to court_points.json
            ball_csv_path: Path to ball_tracks_interpolated.csv
            video_path: Path to input video
        """
        self.court_json_path = Path(court_json_path)
        self.ball_csv_path = Path(ball_csv_path)
        self.video_path = Path(video_path)
        
        print(f"\n{'='*70}")
        print(f"MINI-COURT BALL PROJECTION (with RANSAC)")
        print(f"{'='*70}")
        print(f"Court JSON: {self.court_json_path}")
        print(f"Ball CSV: {self.ball_csv_path}")
        print(f"Video: {self.video_path}")
        print(f"{'='*70}\n")
        
        # Load data
        self._load_court_calibration()
        self._load_ball_tracks()
        
        # ═══════════════════════════════════════════════════════════════
        # 🔧 تغییر: کوچکتر کردن Mini-court به نصف اندازه
        # ═══════════════════════════════════════════════════════════════
        
        # Mini-court settings
        self.minicourt_width = 200  # کاهش از 400 به 200
        self.minicourt_height = int(self.minicourt_width * 23.77 / 10.97)  # 433 pixels
        self.minicourt_margin = 25  # کاهش از 50 به 25
        
        # Trail settings
        self.trail_length = 30  # Show last 30 positions
        self.trail_positions = deque(maxlen=self.trail_length)
    
    def _load_court_calibration(self):
        """Load court calibration from JSON"""
        print("[1/2] Loading court calibration...")
        with open(self.court_json_path, 'r') as f:
            court_data = json.load(f)
        
        # Get court dimensions
        self.court_length = court_data['court_dimensions']['length']
        self.court_width = court_data['court_dimensions']['width_doubles']
        
        # Extract 19 points - handle dict format
        points_image_dict = court_data['points_image']
        points_world_dict = court_data['points_world']
        
        # Convert dict to array (maintain consistent order)
        point_names = [
            'near_baseline_left', 'near_baseline_right',
            'near_singles_left', 'near_singles_right',
            'near_service_left', 'near_service_right', 'near_service_center',
            'far_baseline_left', 'far_baseline_right',
            'far_singles_left', 'far_singles_right',
            'far_service_left', 'far_service_right', 'far_service_center',
            'net_left', 'net_singles_left', 'net_center', 'net_singles_right', 'net_right'
        ]
        
        points_image = []
        points_world = []
        for name in point_names:
            if name in points_image_dict:
                points_image.append(points_image_dict[name])
                points_world.append(points_world_dict[name])
        
        points_image = np.array(points_image, dtype=np.float32)
        points_world = np.array(points_world, dtype=np.float32)
        
        # Compute homography: image -> court (meters) with RANSAC
        self.H_img2court, mask = cv2.findHomography(
            points_image, 
            points_world,
            method=cv2.RANSAC,           # استفاده از RANSAC برای حذف outliers
            ransacReprojThreshold=5.0    # آستانه خطا: 5 پیکسل
        )
        
        # نمایش آمار RANSAC
        if mask is not None:
            inliers = np.sum(mask)
            outliers = len(mask) - inliers
            print(f"   ✓ RANSAC Analysis:")
            print(f"      • Inliers (good points): {inliers}/{len(mask)}")
            print(f"      • Outliers (bad points): {outliers}/{len(mask)}")
            
            # نمایش نقاط outlier
            if outliers > 0:
                print(f"      ⚠️ Outlier points detected:")
                for i, is_inlier in enumerate(mask):
                    if is_inlier == 0:
                        print(f"         - Point {i} ({point_names[i]})")
        else:
            print(f"   ⚠️ RANSAC mask is None - all points used")
        
        self.H_court2img = np.linalg.inv(self.H_img2court)
        
        print(f"   ✓ Loaded {len(points_image)} court points")
        print(f"   ✓ Court dimensions: {self.court_length}m × {self.court_width}m")
    
    def _load_ball_tracks(self):
        """Load ball tracking data"""
        print("[2/2] Loading ball tracks...")
        self.df_ball = pd.read_csv(self.ball_csv_path)
        
        # Add columns for court coordinates
        self.df_ball['court_x'] = np.nan
        self.df_ball['court_y'] = np.nan
        
        # Project all ball positions to court
        for idx, row in self.df_ball.iterrows():
            if pd.notna(row['final_x']) and pd.notna(row['final_y']):
                # Image coordinates
                img_pt = np.array([[row['final_x'], row['final_y']]], dtype=np.float32)
                
                # Transform to court coordinates (meters)
                court_pt = cv2.perspectiveTransform(img_pt.reshape(1, 1, 2), self.H_img2court)
                self.df_ball.at[idx, 'court_x'] = court_pt[0, 0, 0]
                self.df_ball.at[idx, 'court_y'] = court_pt[0, 0, 1]
        
        # Count valid projections
        valid_count = self.df_ball['court_x'].notna().sum()
        print(f"   ✓ Projected {valid_count}/{len(self.df_ball)} ball positions to court")
    
    def _create_mini_court_template(self):
        """Create mini-court template with lines"""
        W = self.minicourt_width
        H = self.minicourt_height
        
        # Create white canvas
        court_img = np.ones((H, W, 3), dtype=np.uint8) * 255
        
        # Scale factor: meters to pixels
        scale_x = W / self.court_width
        scale_y = H / self.court_length
        
        def to_pixel(x_m, y_m):
            """Convert court meters to pixel coordinates"""
            px = int(x_m * scale_x)
            py = int(y_m * scale_y)
            return (px, py)
        
        # Tennis court lines (in meters, ITF standard)
        color_line = (50, 50, 50)  # Dark gray
        thickness = 1  # کاهش ضخامت خطوط برای mini-court کوچکتر
        
        # Outer boundary (doubles)
        pts = np.array([
            to_pixel(0, 0),
            to_pixel(self.court_width, 0),
            to_pixel(self.court_width, self.court_length),
            to_pixel(0, self.court_length)
        ])
        cv2.polylines(court_img, [pts], True, color_line, thickness)
        
        # Singles lines
        singles_margin = 1.37
        pts = np.array([
            to_pixel(singles_margin, 0),
            to_pixel(self.court_width - singles_margin, 0),
            to_pixel(self.court_width - singles_margin, self.court_length),
            to_pixel(singles_margin, self.court_length)
        ])
        cv2.polylines(court_img, [pts], True, color_line, thickness)
        
        # Service lines
        service_dist = 6.4
        
        # Near service line
        cv2.line(court_img,
                to_pixel(0, service_dist),
                to_pixel(self.court_width, service_dist),
                color_line, thickness)
        
        # Far service line
        cv2.line(court_img,
                to_pixel(0, self.court_length - service_dist),
                to_pixel(self.court_width, self.court_length - service_dist),
                color_line, thickness)
        
        # Center service line
        net_y = self.court_length / 2
        cv2.line(court_img,
                to_pixel(self.court_width/2, service_dist),
                to_pixel(self.court_width/2, self.court_length - service_dist),
                color_line, thickness)
        
        # Net line (thicker and darker)
        cv2.line(court_img,
                to_pixel(0, net_y),
                to_pixel(self.court_width, net_y),
                (0, 0, 0), thickness + 1)
        
        return court_img
    
    def _court_to_minicourt(self, court_x, court_y):
        """Convert court coordinates (meters) to mini-court pixel coordinates"""
        W = self.minicourt_width
        H = self.minicourt_height
        
        scale_x = W / self.court_width
        scale_y = H / self.court_length
        
        px = int(court_x * scale_x)
        py = int(court_y * scale_y)
        
        # Clamp to image bounds
        px = max(0, min(W - 1, px))
        py = max(0, min(H - 1, py))
        
        return (px, py)
    
    def process_video(self, output_video_path):
        """Process video and add mini-court overlay"""
        print(f"\n[PROCESSING VIDEO]")
        print(f"Input: {self.video_path}")
        print(f"Output: {output_video_path}")
        
        # Open video
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {self.video_path}")
        
        # Video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"   Resolution: {width}×{height}")
        print(f"   FPS: {fps}")
        print(f"   Total frames: {total_frames}")
        
        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
        
        # Create mini-court template
        minicourt_template = self._create_mini_court_template()
        
        # Position mini-court on video frame (top-right corner)
        margin = self.minicourt_margin
        start_x = width - self.minicourt_width - margin
        start_y = margin
        
        frame_idx = 0
        print(f"\n   Processing frames...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1
            
            # Get ball position for this frame
            ball_row = self.df_ball[self.df_ball['frame'] == frame_idx]
            
            if not ball_row.empty and pd.notna(ball_row.iloc[0]['court_x']):
                court_x = ball_row.iloc[0]['court_x']
                court_y = ball_row.iloc[0]['court_y']
                
                # Add to trail
                self.trail_positions.append((court_x, court_y))
            
            # Create mini-court for this frame
            minicourt = minicourt_template.copy()
            
            # Draw trail
            if len(self.trail_positions) > 1:
                for i in range(1, len(self.trail_positions)):
                    pt1 = self._court_to_minicourt(*self.trail_positions[i-1])
                    pt2 = self._court_to_minicourt(*self.trail_positions[i])
                    
                    # Fade trail (older = lighter)
                    alpha = i / len(self.trail_positions)
                    color = (int(255 * (1 - alpha)), int(165 * (1 - alpha)), 0)  # Orange fade
                    cv2.line(minicourt, pt1, pt2, color, 1)
            
            # Draw current ball position
            if len(self.trail_positions) > 0:
                current_pt = self._court_to_minicourt(*self.trail_positions[-1])
                cv2.circle(minicourt, current_pt, 3, (0, 255, 0), -1)  # Green ball - کوچکتر شده
                cv2.circle(minicourt, current_pt, 4, (0, 0, 0), 1)     # Black border - کوچکتر شده
            
            # Add semi-transparent background for mini-court
            overlay = frame.copy()
            cv2.rectangle(overlay,
                         (start_x - 5, start_y - 5),
                         (start_x + self.minicourt_width + 5,
                          start_y + self.minicourt_height + 5),
                         (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
            
            # Overlay mini-court on frame
            frame[start_y:start_y + self.minicourt_height,
                  start_x:start_x + self.minicourt_width] = minicourt
            
            # Add border
            cv2.rectangle(frame,
                         (start_x, start_y),
                         (start_x + self.minicourt_width, start_y + self.minicourt_height),
                         (255, 255, 255), 1)
            
            # Write frame
            out.write(frame)
            
            # Progress
            if frame_idx % 100 == 0:
                print(f"   Frame {frame_idx}/{total_frames} ({100*frame_idx/total_frames:.1f}%)")
        
        cap.release()
        out.release()
        
        print(f"\n   ✓ Processed {frame_idx} frames")
        print(f"   ✓ Saved to: {output_video_path}")
    
    def create_trajectory_image(self, output_image_path):
        """Create static image showing all ball positions on court"""
        print(f"\n[CREATING TRAJECTORY IMAGE]")
        
        # Create court template
        court_img = self._create_mini_court_template()
        
        # Plot all valid ball positions
        for _, row in self.df_ball.iterrows():
            if pd.notna(row['court_x']) and pd.notna(row['court_y']):
                pt = self._court_to_minicourt(row['court_x'], row['court_y'])
                cv2.circle(court_img, pt, 1, (0, 100, 255), -1)  # Small blue dots
        
        # Mark start position (first valid ball)
        first_valid = self.df_ball[self.df_ball['court_x'].notna()].iloc[0]
        start_pt = self._court_to_minicourt(first_valid['court_x'], first_valid['court_y'])
        cv2.circle(court_img, start_pt, 5, (0, 255, 0), -1)  # Green for start
        cv2.putText(court_img, "START", (start_pt[0] + 10, start_pt[1] + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
        
        # Save image
        cv2.imwrite(str(output_image_path), court_img)
        print(f"   ✓ Saved trajectory image: {output_image_path}")


def main():
    """Main execution"""
    # Paths - CUSTOMIZE THESE
    BASE_DIR = Path(r"C:\Users\Ost\Desktop\F11_TENNIS")
    
    court_json = BASE_DIR / "results" / "intermediate" / "court_calib" / "court_points.json"
    ball_csv = BASE_DIR / "ball_tracks_interpolated.csv"
    video_in = BASE_DIR / "data" / "clips" / "input_video.mp4"
    video_out = BASE_DIR / "output_with_minicourt.mp4"
    trajectory_img = BASE_DIR / "ball_trajectory_on_court.png"
    
    # Initialize visualizer
    viz = MiniCourtVisualizer(court_json, ball_csv, video_in)
    
    # Create trajectory image
    viz.create_trajectory_image(trajectory_img)
    
    # Process video
    viz.process_video(video_out)
    
    print(f"\n{'='*70}")
    print(f"✅ MINI-COURT VISUALIZATION COMPLETE (with RANSAC)")
    print(f"{'='*70}\n")
    print(f"📊 Output files:")
    print(f"   1. Video: {video_out}")
    print(f"   2. Trajectory: {trajectory_img}")
    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
