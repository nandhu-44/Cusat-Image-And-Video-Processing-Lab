import cv2
import os

def extract_frames(video_path, output_dir, interval=30):
    """
    Extract frames from video at regular intervals

    Parameters:
    video_path: Path to the video file
    output_dir: Directory to save extracted frames
    interval: Extract every 'interval' frames
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video properties:")
    print(f"Total frames: {total_frames}")
    print(f"FPS: {fps}")
    print(f"Resolution: {width}x{height}")

    frame_count = 0
    extracted_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Extract frame at specified intervals
        if frame_count % interval == 0:
            frame_filename = os.path.join(output_dir, f"frame_{extracted_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            extracted_count += 1
            print(f"Extracted frame {extracted_count} at position {frame_count}")

        frame_count += 1

    cap.release()
    print(f"Extraction complete. {extracted_count} frames extracted.")

def main():
    # Note: You'll need to provide a video file path
    # For demonstration, we'll use a placeholder
    video_path = "../sample_video.mp4"  # Replace with actual video path
    output_dir = "extracted_frames"

    try:
        extract_frames(video_path, output_dir, interval=30)  # Extract every 30 frames
    except Exception as e:
        print(f"Error: {e}")
        print("Please ensure you have a video file at the specified path.")

if __name__ == "__main__":
    main()
