import os
import cv2

video_path = os.path.join("data", "clips", "IA_cut.mp4")
out_dir = os.path.join("data", "frames")
os.makedirs(out_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
i = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    i += 1
    out_path = os.path.join(out_dir, f"frame_{i:06d}.jpg")
    cv2.imwrite(out_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

cap.release()
print("Saved frames:", i)
