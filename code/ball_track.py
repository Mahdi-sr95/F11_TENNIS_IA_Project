#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tennis Ball Tracker - YOLO + Kalman Filter
Project: F11_TENNIS
Model: YOLOv5 (yolo5_last.pt)
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import csv


class KalmanFilter2D:
    """Kalman Filter for 2D ball tracking"""
    
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 5
        
        self.initialized = False
    
    def predict(self):
        """Predict next ball position"""
        pred = self.kf.predict()
        return int(pred[0]), int(pred[1])
    
    def update(self, x, y):
        """Update with actual ball position"""
        if not self.initialized:
            self.kf.statePost = np.array([x, y, 0, 0], dtype=np.float32)
            self.initialized = True
        
        measurement = np.array([[x], [y]], dtype=np.float32)
        self.kf.correct(measurement)
        return int(self.kf.statePost[0]), int(self.kf.statePost[1])


class TennisBallTracker:
    """Tennis ball tracker using YOLO and Kalman Filter"""
    
    def __init__(
        self,
        video_path,
        model_path,
        output_csv,
        output_video=None,
        confidence_threshold=0.10,
        show_debug=True
    ):
        self.video_path = Path(video_path)
        self.model_path = Path(model_path)
        self.output_csv = Path(output_csv)
        self.output_video = Path(output_video) if output_video else None
        self.confidence = confidence_threshold
        self.show_debug = show_debug
        
        # Check if files exist
        if not self.video_path.exists():
            raise FileNotFoundError(f"❌ Video not found: {self.video_path}")
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"❌ Model file not found: {self.model_path}")
        
        # Load YOLO model
        print(f"\n✓ Loading YOLO model: {self.model_path.name}")
        self.model = YOLO(str(self.model_path))
        print(f"✓ Model loaded successfully!")
        
        # Initialize Kalman Filter
        self.kf = KalmanFilter2D()
        
        # Store tracking data
        self.tracks = []
        
    def process_video(self):
        """Process video and track ball"""
        cap = cv2.VideoCapture(str(self.video_path))
        
        if not cap.isOpened():
            raise RuntimeError(f"❌ Cannot open video: {self.video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\n📹 Video Information:")
        print(f"   Resolution: {width}x{height}")
        print(f"   FPS: {fps}")
        print(f"   Total Frames: {total_frames}")
        print(f"   Duration: {total_frames/fps:.1f} seconds")
        
        # Create VideoWriter
        out = None
        if self.output_video:
            self.output_video.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(
                str(self.output_video),
                fourcc,
                fps,
                (width, height)
            )
            print(f"✓ Output video will be saved to: {self.output_video}")
        
        frame_num = 0
        detection_count = 0
        
        print(f"\n🎾 Processing frames...\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_num += 1
            
            # Detect ball with YOLO
            results = self.model(frame, conf=self.confidence, verbose=False)
            
            ball_detected = False
            detected_x, detected_y = None, None
            best_conf = 0
            best_box = None
            
            # Find best detection
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    conf = float(box.conf[0])
                    if conf > best_conf:
                        best_conf = conf
                        best_box = box
            
            # Process detection
            if best_box is not None:
                x1, y1, x2, y2 = map(int, best_box.xyxy[0])
                detected_x = (x1 + x2) // 2
                detected_y = (y1 + y2) // 2
                ball_detected = True
                detection_count += 1
                
                # Update Kalman Filter
                kf_x, kf_y = self.kf.update(detected_x, detected_y)
                
                # Draw on frame
                # 1. YOLO detection rectangle (blue)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                # 2. Detection center point (red)
                cv2.circle(frame, (detected_x, detected_y), 5, (0, 0, 255), -1)
                
                # 3. Kalman Filter point (green)
                cv2.circle(frame, (kf_x, kf_y), 7, (0, 255, 0), 2)
                
                # 4. Connection line
                cv2.line(frame, (detected_x, detected_y), (kf_x, kf_y), 
                        (255, 255, 0), 1)
                
                # 5. Confidence text
                if self.show_debug:
                    text = f"Conf: {best_conf:.2f}"
                    cv2.putText(frame, text, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            else:
                # Kalman prediction only
                if self.kf.initialized:
                    kf_x, kf_y = self.kf.predict()
                    # Prediction circle (orange)
                    cv2.circle(frame, (kf_x, kf_y), 7, (0, 165, 255), 2)
                else:
                    kf_x, kf_y = None, None
            
            # Store data
            self.tracks.append({
                'frame': frame_num,
                'detected': ball_detected,
                'det_x': detected_x,
                'det_y': detected_y,
                'kf_x': kf_x if self.kf.initialized else None,
                'kf_y': kf_y if self.kf.initialized else None,
                'confidence': best_conf if ball_detected else 0.0
            })
            
            # Display frame information
            if self.show_debug:
                info_text = f"Frame: {frame_num}/{total_frames} | Detections: {detection_count}"
                cv2.putText(frame, info_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Status
                status = "DETECTED" if ball_detected else "PREDICTED"
                color = (0, 255, 0) if ball_detected else (0, 165, 255)
                cv2.putText(frame, status, (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Write frame
            if out:
                out.write(frame)
            
            # Show progress
            if frame_num % 30 == 0:
                progress = (frame_num / total_frames) * 100
                print(f"   Progress: {progress:.1f}% | Frame {frame_num}/{total_frames} | Detections: {detection_count}")
        
        # Release resources
        cap.release()
        if out:
            out.release()
        
        # Summary
        print(f"\n" + "="*70)
        print(f"✓ Processing Complete!")
        print(f"="*70)
        print(f"   Total Frames Processed: {frame_num}")
        print(f"   Ball Detections: {detection_count}")
        print(f"   Detection Rate: {(detection_count/frame_num)*100:.1f}%")
        print(f"   Missed Frames: {frame_num - detection_count}")
        print(f"="*70)
        
    def save_csv(self):
        """Save results to CSV file"""
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'frame', 'detected', 'det_x', 'det_y',
                'kf_x', 'kf_y', 'confidence'
            ])
            writer.writeheader()
            writer.writerows(self.tracks)
        
        print(f"\n✓ CSV file saved: {self.output_csv}")
        print(f"   Total records: {len(self.tracks)}")
    
    def run(self):
        """Run complete tracking process"""
        self.process_video()
        self.save_csv()


def main():
    """Main function"""
    
    print("=" * 70)
    print("         🎾 TENNIS BALL TRACKER - YOLO + Kalman Filter 🎾")
    print("=" * 70)
    
    # ═══════════════════════════════════════════════════════════════
    # Project Configuration
    # ═══════════════════════════════════════════════════════════════
    
    PROJECT_ROOT = Path(r"C:\Users\Ost\Desktop\F11_TENNIS")
    
    # YOLO model path (exactly as you saved it)
    MODEL_PATH = PROJECT_ROOT / "Model" / "yolo5_last.pt"
    
    # Input and output paths
    VIDEO_PATH = PROJECT_ROOT / "data" / "clips" / "input_video.mp4"
    OUTPUT_CSV = PROJECT_ROOT / "ball_tracks.csv"
    OUTPUT_VIDEO = PROJECT_ROOT / "output_tracking.mp4"
    
    # Settings
    CONFIDENCE_THRESHOLD = 0.10  # Confidence threshold (0.1 to 0.9)
    SHOW_DEBUG = True             # Show debug info on video
    
    # ═══════════════════════════════════════════════════════════════
    
    print(f"\n📋 Configuration:")
    print(f"   Video Input  : {VIDEO_PATH}")
    print(f"   YOLO Model   : {MODEL_PATH}")
    print(f"   CSV Output   : {OUTPUT_CSV}")
    print(f"   Video Output : {OUTPUT_VIDEO}")
    print(f"   Confidence   : {CONFIDENCE_THRESHOLD}")
    print(f"   Debug Mode   : {SHOW_DEBUG}")
    print("=" * 70)
    
    try:
        # Create and run tracker
        tracker = TennisBallTracker(
            video_path=VIDEO_PATH,
            model_path=MODEL_PATH,
            output_csv=OUTPUT_CSV,
            output_video=OUTPUT_VIDEO,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            show_debug=SHOW_DEBUG
        )
        
        tracker.run()
        
        print("\n" + "=" * 70)
        print("✅ SUCCESS - All tasks completed successfully!")
        print("=" * 70)
        print(f"\n📊 Output files:")
        print(f"   1. CSV: {OUTPUT_CSV}")
        print(f"   2. Video: {OUTPUT_VIDEO}")
        print("\n✓ You can now use these files for further analysis.")
        print("=" * 70 + "\n")
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("\n💡 Solution:")
        print("   • Check if all file paths are correct")
        print("   • Make sure the model and video files exist")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
