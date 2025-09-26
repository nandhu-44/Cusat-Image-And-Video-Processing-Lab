import cv2
import numpy as np

def apply_filter_to_frame(frame, filter_type='smooth'):
    """
    Apply image processing filter to a single frame

    Parameters:
    frame: Input video frame
    filter_type: Type of filter to apply ('smooth', 'sharpen', 'edge')

    Returns:
    Processed frame
    """
    if filter_type == 'smooth':
        # Gaussian blur for smoothing
        processed = cv2.GaussianBlur(frame, (5, 5), 0)

    elif filter_type == 'sharpen':
        # Sharpening using Laplacian
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        processed = cv2.filter2D(frame, -1, kernel)

    elif filter_type == 'edge':
        # Edge detection using Canny
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        processed = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    else:
        processed = frame  # No processing

    return processed

def process_video_realtime(video_path, filter_type='smooth'):
    """
    Process video frames in real-time and display

    Parameters:
    video_path: Path to the video file
    filter_type: Type of filter to apply
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Processing video with {filter_type} filter...")
    print(f"Press 'q' to quit, 's' to switch to smooth, 'h' to sharpen, 'e' for edges")

    current_filter = filter_type

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Apply current filter
        processed_frame = apply_filter_to_frame(frame, current_filter)

        # Display original and processed frames
        cv2.imshow('Original', frame)
        cv2.imshow('Processed', processed_frame)

        # Check for key presses
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s'):
            current_filter = 'smooth'
            print("Switched to smooth filter")
        elif key == ord('h'):
            current_filter = 'sharpen'
            print("Switched to sharpen filter")
        elif key == ord('e'):
            current_filter = 'edge'
            print("Switched to edge detection")

    cap.release()
    cv2.destroyAllWindows()

def main():
    # Note: You'll need to provide a video file path
    video_path = "../sample_video.mp4"  # Replace with actual video path

    try:
        process_video_realtime(video_path, 'smooth')
    except Exception as e:
        print(f"Error: {e}")
        print("Please ensure you have a video file at the specified path.")

if __name__ == "__main__":
    main()
