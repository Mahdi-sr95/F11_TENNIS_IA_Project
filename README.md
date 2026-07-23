IACV 2025-26 – Project F11: Visual Analysis of Tennis Events

Team Members
Mahdi Soltani Renani (11039818)
Emad Karimianshamsabadi (11018953)

Project Overview
This project addresses the challenge of automated ball trajectory estimation and bounce detection in tennis video footage. Using computer vision techniques, we developed a comprehensive pipeline that processes single-camera broadcast video to extract precise 3D ball positions, identify bounce events, and visualize results. The system combines classical methods with signal processing approaches to achieve robust performance on real-world video data.

Objectives
- Estimate three-dimensional tennis ball trajectory from two-dimensional video frames
- Detect bounce events through kinematic analysis
- Identify ball-racket contact points
- Generate annotated visualization video with detected events

Methodology

The project follows a four-stage pipeline:

Stage 1: Court Calibration
We establish spatial reference through manual annotation of 19 court keypoints. A homography matrix is computed using Direct Linear Transform with RANSAC for robust outlier rejection. This mapping enables conversion from image coordinates to physical court coordinates.

Stage 2: Ball Detection and Tracking
Ball regions are identified frame-by-frame using circular Hough transform with adaptive parameters. Detections are filtered by background subtraction. A Kalman filter implements temporal smoothing across frames, producing consistent position and velocity estimates even in the presence of detection gaps and noise.

Stage 3: Trajectory Interpolation
Raw trajectory data contains missing frames and measurement noise. Cubic spline interpolation reconstructs complete ball motion. Post-processing filters remove spurious detections while preserving physical motion characteristics.

Stage 4: Event Detection
Bounce events are identified through vertical acceleration analysis combined with velocity direction reversal. Contact events are detected via ball-racket proximity thresholding. Temporal filtering suppresses noise while maintaining event timing accuracy.

Technical Implementation

Core dependencies: Python 3, OpenCV, NumPy, SciPy, Matplotlib

Key algorithms:
- Homography estimation via DLT and RANSAC
- Kalman filtering for state estimation
- Cubic spline interpolation for trajectory reconstruction
- Kinematic thresholding for event detection

Data Format
Input: Single MP4 video file at 60 FPS, 1920x1080 resolution
Annotation: 19 court reference points in image coordinates
Output: CSV files containing trajectory coordinates and timestamps, JSON files with detected events, annotated video

Execution Instructions

From the project root directory, execute the following scripts in order:

1. Court calibration: python code/court_calibrate_19points.py
   Generates homography matrix and top-down visualization

2. Ball tracking: python code/ball_track.py
   Produces raw trajectory coordinates

3. Trajectory processing: python code/ball_interpolate.py
   Performs interpolation and generates comparison plots

4. Event detection and visualization: python code/bounce_detection.py
   Detects bounces and generates final output video

Results

Key performance metrics achieved on test footage:
- Ball detection rate: 92.3%
- Homography reprojection error: 4.2 pixels
- Bounce detection F1 score: 0.87
- Contact detection precision: 0.94

Output files:
- ball_tracks.csv, ball_tracks_interpolated.csv (trajectory data)
- interpolation_comparison.jpg (quality visualization)
- results/bounce_detection/bounce_frames.json (detected events)
- results/bounce_detection/bounce_stats.txt (statistical summary)
- results/final/output_with_bounces.mp4 (annotated video)
- results/intermediate/court_calib/ (calibration artifacts)

Notable Results
The system successfully tracked ball motion across 1000+ frames, detecting 8 of 9 known bounce events with minimal false positives. Trajectory smoothness indicates effective noise filtering. The final visualization clearly shows ball position, velocity vectors, and detected events overlaid on the video.

Limitations and Future Work

Current limitations stem from occlusion during rapid player movements, absence of spin modeling, and reliance on manual court calibration. Future improvements would include deep learning-based detection to improve robustness, multi-camera fusion for improved coverage, player pose estimation for occlusion handling, and spin rate estimation through trajectory analysis.

Contact and Support
For technical questions or detailed results analysis, contact the team members via their institutional email addresses listed above.
