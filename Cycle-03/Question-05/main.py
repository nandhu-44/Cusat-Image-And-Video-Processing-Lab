import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_binary_image_with_lines():
    """
    Create a binary image with various line segments for testing
    """
    # Create a blank image
    image = np.zeros((400, 400), dtype=np.uint8)
    
    # Draw horizontal lines
    cv2.line(image, (50, 100), (350, 100), 255, 2)
    cv2.line(image, (100, 300), (300, 300), 255, 2)
    
    # Draw vertical lines
    cv2.line(image, (100, 50), (100, 350), 255, 2)
    cv2.line(image, (300, 100), (300, 250), 255, 2)
    
    # Draw diagonal lines
    cv2.line(image, (50, 50), (200, 200), 255, 2)
    cv2.line(image, (250, 150), (350, 350), 255, 2)
    cv2.line(image, (350, 50), (200, 200), 255, 2)
    
    return image

def detect_lines_hough(binary_image, rho=1, theta=np.pi/180, threshold=100, 
                       min_line_length=50, max_line_gap=10):
    """
    Detect line segments using Hough Transform
    
    Parameters:
    binary_image: Binary input image
    rho: Distance resolution in pixels
    theta: Angle resolution in radians
    threshold: Minimum number of votes
    min_line_length: Minimum line length
    max_line_gap: Maximum gap between line segments
    
    Returns:
    lines: Detected lines
    """
    # Use Probabilistic Hough Line Transform
    lines = cv2.HoughLinesP(binary_image, rho, theta, threshold, 
                            minLineLength=min_line_length, 
                            maxLineGap=max_line_gap)
    
    return lines

def detect_lines_standard_hough(binary_image, rho=1, theta=np.pi/180, threshold=100):
    """
    Detect lines using Standard Hough Transform
    
    Parameters:
    binary_image: Binary input image
    rho: Distance resolution in pixels
    theta: Angle resolution in radians
    threshold: Minimum number of votes
    
    Returns:
    lines: Detected lines in polar coordinates (rho, theta)
    """
    lines = cv2.HoughLines(binary_image, rho, theta, threshold)
    
    return lines

def draw_hough_lines_probabilistic(image, lines):
    """
    Draw detected lines from Probabilistic Hough Transform
    """
    result = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    return result

def draw_hough_lines_standard(image, lines):
    """
    Draw detected lines from Standard Hough Transform
    """
    result = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)
    
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            
            # Calculate endpoints of the line
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            
            cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    return result

def visualize_hough_space(binary_image, rho=1, theta_res=np.pi/180):
    """
    Visualize the Hough accumulator space
    """
    # Calculate Hough transform accumulator
    lines = cv2.HoughLines(binary_image, rho, theta_res, 1)
    
    # Create accumulator visualization
    height, width = binary_image.shape
    diagonal = int(np.sqrt(height**2 + width**2))
    
    # Create accumulator array
    num_rhos = int(2 * diagonal / rho)
    num_thetas = int(np.pi / theta_res)
    accumulator = np.zeros((num_rhos, num_thetas), dtype=np.uint32)
    
    # Get edge points
    edge_points = np.argwhere(binary_image > 0)
    
    # Vote in accumulator
    for y, x in edge_points:
        for theta_idx in range(num_thetas):
            theta = theta_idx * theta_res
            rho_val = x * np.cos(theta) + y * np.sin(theta)
            rho_idx = int((rho_val + diagonal) / rho)
            
            if 0 <= rho_idx < num_rhos:
                accumulator[rho_idx, theta_idx] += 1
    
    return accumulator

def main():
    # Create a test binary image with line segments
    print("Creating binary image with line segments...")
    binary_image = create_binary_image_with_lines()
    
    print("Binary image created!")
    print(f"Image shape: {binary_image.shape}")
    
    # Detect lines using Probabilistic Hough Transform
    print("\nDetecting lines using Probabilistic Hough Transform...")
    lines_prob = detect_lines_hough(binary_image, rho=1, theta=np.pi/180, 
                                    threshold=50, min_line_length=30, max_line_gap=10)
    
    if lines_prob is not None:
        print(f"Detected {len(lines_prob)} line segments")
    else:
        print("No lines detected")
    
    # Detect lines using Standard Hough Transform
    print("\nDetecting lines using Standard Hough Transform...")
    lines_standard = detect_lines_standard_hough(binary_image, rho=1, 
                                                 theta=np.pi/180, threshold=100)
    
    if lines_standard is not None:
        print(f"Detected {len(lines_standard)} lines")
    else:
        print("No lines detected")
    
    # Draw detected lines
    result_prob = draw_hough_lines_probabilistic(binary_image, lines_prob)
    result_standard = draw_hough_lines_standard(binary_image, lines_standard)
    
    # Visualize Hough accumulator space
    print("\nVisualizing Hough accumulator space...")
    accumulator = visualize_hough_space(binary_image)
    
    # Display results
    plt.figure(figsize=(18, 10))
    
    # Original binary image
    plt.subplot(2, 4, 1)
    plt.imshow(binary_image, cmap='gray')
    plt.title('Original Binary Image')
    plt.axis('off')
    
    # Edge detection (Canny)
    edges = cv2.Canny(binary_image, 50, 150)
    plt.subplot(2, 4, 2)
    plt.imshow(edges, cmap='gray')
    plt.title('Edge Detection')
    plt.axis('off')
    
    # Probabilistic Hough Transform result
    plt.subplot(2, 4, 3)
    plt.imshow(cv2.cvtColor(result_prob, cv2.COLOR_BGR2RGB))
    plt.title(f'Probabilistic Hough\n({len(lines_prob) if lines_prob is not None else 0} segments)')
    plt.axis('off')
    
    # Standard Hough Transform result
    plt.subplot(2, 4, 4)
    plt.imshow(cv2.cvtColor(result_standard, cv2.COLOR_BGR2RGB))
    plt.title(f'Standard Hough\n({len(lines_standard) if lines_standard is not None else 0} lines)')
    plt.axis('off')
    
    # Hough accumulator space
    plt.subplot(2, 4, 5)
    plt.imshow(accumulator, cmap='hot', aspect='auto')
    plt.title('Hough Accumulator Space')
    plt.xlabel('Theta (degrees)')
    plt.ylabel('Rho (pixels)')
    plt.colorbar()
    
    # Both methods overlaid
    plt.subplot(2, 4, 6)
    combined = cv2.cvtColor(binary_image.copy(), cv2.COLOR_GRAY2BGR)
    if lines_prob is not None:
        for line in lines_prob:
            x1, y1, x2, y2 = line[0]
            cv2.line(combined, (x1, y1), (x2, y2), (0, 255, 0), 2)
    plt.imshow(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
    plt.title('Line Segments Detected')
    plt.axis('off')
    
    # Line statistics
    plt.subplot(2, 4, 7)
    if lines_prob is not None:
        lengths = []
        angles = []
        for line in lines_prob:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            lengths.append(length)
            angles.append(angle)
        
        plt.hist(lengths, bins=20, alpha=0.7, color='blue')
        plt.title('Line Length Distribution')
        plt.xlabel('Length (pixels)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No lines detected', ha='center', va='center')
        plt.axis('off')
    
    plt.subplot(2, 4, 8)
    if lines_prob is not None and len(angles) > 0:
        plt.hist(angles, bins=20, alpha=0.7, color='red')
        plt.title('Line Angle Distribution')
        plt.xlabel('Angle (degrees)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No lines detected', ha='center', va='center')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print line details
    if lines_prob is not None:
        print("\nDetected Line Segments:")
        print("Index | Start (x1,y1) | End (x2,y2) | Length | Angle")
        print("-" * 65)
        for i, line in enumerate(lines_prob[:10]):  # Show first 10 lines
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            print(f"{i:5d} | ({x1:3d},{y1:3d}) | ({x2:3d},{y2:3d}) | {length:6.1f} | {angle:6.1f}°")

if __name__ == "__main__":
    main()