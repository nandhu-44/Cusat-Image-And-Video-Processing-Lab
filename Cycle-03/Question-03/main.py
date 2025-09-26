import cv2
import numpy as np
import os

def apply_processing_to_frame(frame, processing_type='smooth'):
    """
    Apply processing to a single frame

    Parameters:
    frame: Input frame
    processing_type: Type of processing ('smooth', 'sharpen', 'grayscale', 'blur')

    Returns:
    Processed frame
    """
    if processing_type == 'smooth':
        # Gaussian smoothing
        processed = cv2.GaussianBlur(frame, (5, 5), 0)

    elif processing_type == 'sharpen':
        # Sharpening filter
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        processed = cv2.filter2D(frame, -1, kernel)

    elif processing_type == 'grayscale':
        # Convert to grayscale
        processed = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)  # Keep 3 channels for video

    elif processing_type == 'blur':
        # Average blur
        processed = cv2.blur(frame, (5, 5))

    else:
        processed = frame

    return processed

def create_processed_video(input_video_path, output_video_path, processing_type='smooth'):
    """
    Read video, apply processing to frames, and save as new video

    Parameters:
    input_video_path: Path to input video
    output_video_path: Path to save processed video
    processing_type: Type of processing to apply
    """
    # Open input video
    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        print(f"Error: Could not open input video {input_video_path}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Input video properties:")
    print(f"FPS: {fps}")
    print(f"Resolution: {width}x{height}")
    print(f"Total frames: {total_frames}")
    print(f"Applying {processing_type} processing...")

    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 codec
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    if not out.isOpened():
        print("Error: Could not create output video file")
        cap.release()
        return

    frame_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Apply processing
        processed_frame = apply_processing_to_frame(frame, processing_type)

        # Write processed frame to output video
        out.write(processed_frame)

        frame_count += 1

        # Print progress
        if frame_count % 100 == 0:
            print(f"Processed {frame_count}/{total_frames} frames")

    # Release resources
    cap.release()
    out.release()

    print(f"Video processing complete!")
    print(f"Processed video saved as: {output_video_path}")

def main():
    # Note: You'll need to provide input video path
    input_video = "../sample_video.mp4"  # Replace with actual video path
    output_video = "processed_video.mp4"

    # Choose processing type
    processing_types = ['smooth', 'sharpen', 'grayscale', 'blur']
    processing_type = 'smooth'  # Change this to try different effects

    try:
        create_processed_video(input_video, output_video, processing_type)
    except Exception as e:
        print(f"Error: {e}")
        print("Please ensure you have a video file at the specified path.")

if __name__ == "__main__":
    main()
