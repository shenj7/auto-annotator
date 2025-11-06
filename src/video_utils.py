import cv2
import os


def extract_frames(video_path, output_dir, max_frames):
    """
    Extract frames from a video and save them to the output directory.
    
    Args:
        video_path: Path to the input video file
        output_dir: Directory to save extracted frames
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Error: Could not open video file {video_path}")
    
    frame_count = 0
    max_frames = 100
    
    print(f"Extracting first {max_frames} frames from {video_path}...")
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Save frame
        frame_filename = os.path.join(output_dir, f"frame_{frame_count:06d}.jpg")
        cv2.imwrite(frame_filename, frame)
        
        frame_count += 1
        
        if frame_count % 10 == 0:
            print(f"Extracted {frame_count}/{max_frames} frames...")
    
    cap.release()
    print(f"Finished! Extracted {frame_count} frames to {output_dir}")

