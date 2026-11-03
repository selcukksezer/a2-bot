# -*- coding: utf-8 -*-
import cv2
import os

# Relative paths
video_path = os.path.expanduser("~/Downloads/route.mp4")
out_dir = os.path.join(".", "video_frames")
os.makedirs(out_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

count = 0
for sec in range(0, int(total / fps), 3):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(sec * fps))
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(os.path.join(out_dir, f"frame_{sec:02d}s.jpg"), frame)
        count += 1
cap.release()
print(f"Extracted {count} frames to {out_dir}")
