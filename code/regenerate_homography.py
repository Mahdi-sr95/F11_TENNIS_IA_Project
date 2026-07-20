#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================
regenerate_homography.py
Regenerate H_img2court.npy and H_court2img.npy from court_points.json
============================================================
"""

import os
import json
import numpy as np
import cv2

# Paths
CALIB_DIR = os.path.join("results", "intermediate", "court_calib")
JSON_PATH = os.path.join(CALIB_DIR, "court_points.json")
H_IMG2COURT_PATH = os.path.join(CALIB_DIR, "H_img2court.npy")
H_COURT2IMG_PATH = os.path.join(CALIB_DIR, "H_court2img.npy")

print("=" * 70)
print("🔄 REGENERATING HOMOGRAPHY MATRICES")
print("=" * 70)

# Check if JSON exists
if not os.path.exists(JSON_PATH):
    raise FileNotFoundError(f"❌ court_points.json not found at: {JSON_PATH}")

# Load calibration data
print(f"\n📂 Loading: {JSON_PATH}")
with open(JSON_PATH, 'r') as f:
    calib_data = json.load(f)

points_image = calib_data['points_image']
points_world = calib_data['points_world']
num_points = calib_data['num_points']

print(f"✓ Loaded {num_points} calibration points")

# Prepare arrays
pts_img = []
pts_world = []

for key in points_image.keys():
    pts_img.append(points_image[key])
    pts_world.append(points_world[key])

pts_img = np.array(pts_img, dtype=np.float32)
pts_world = np.array(pts_world, dtype=np.float32)

print(f"\n🔢 Image points shape: {pts_img.shape}")
print(f"🔢 World points shape: {pts_world.shape}")

# Compute homography: Image -> Court (world)
print("\n⚙️  Computing H_img2court...")
H_img2court, mask = cv2.findHomography(pts_img, pts_world, cv2.RANSAC, 5.0)
inliers = np.sum(mask)
print(f"✓ Homography computed with {inliers}/{num_points} inliers")

# Compute inverse: Court -> Image
print("⚙️  Computing H_court2img...")
H_court2img = np.linalg.inv(H_img2court)
print("✓ Inverse homography computed")

# Save matrices
print(f"\n💾 Saving: {H_IMG2COURT_PATH}")
np.save(H_IMG2COURT_PATH, H_img2court)
print(f"✓ Saved H_img2court.npy")

print(f"\n💾 Saving: {H_COURT2IMG_PATH}")
np.save(H_COURT2IMG_PATH, H_court2img)
print(f"✓ Saved H_court2img.npy")

# Verify by loading
print("\n🔍 Verifying saved files...")
H_test = np.load(H_IMG2COURT_PATH)
print(f"✓ H_img2court.npy verified (shape: {H_test.shape})")

H_test2 = np.load(H_COURT2IMG_PATH)
print(f"✓ H_court2img.npy verified (shape: {H_test2.shape})")

print("\n" + "=" * 70)
print("✅ HOMOGRAPHY MATRICES REGENERATED SUCCESSFULLY!")
print("=" * 70)
print(f"\n📁 Output files:")
print(f"   - {H_IMG2COURT_PATH}")
print(f"   - {H_COURT2IMG_PATH}")
print("\n🚀 Now you can run: python code/ball_track.py\n")
