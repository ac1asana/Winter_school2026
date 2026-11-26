"""
Tools for Watchful Eyes
Date: 25-11-2026
"""
import cv2
import os

def extract_frames_from_mp4(video_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    video = cv2.VideoCapture(video_path)
    
    if not video.isOpened():
        print("Error: Could not open video.")
        return

    frame_count = 0
    while True:
        success, frame = video.read()
        
        if not success:
            break
            
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
        
        cv2.imwrite(frame_filename, frame)
        
        frame_count += 1

    video.release()
    print(f"Successfully extracted {frame_count} frames.")


extract_frames_from_mp4(video_path ="./Recording/50ml_honey_success.mp4", output_folder = "extracted_frames/honey")

