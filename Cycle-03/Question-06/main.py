import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_blob_image():
    """
    Create an image with small, non-overlapping blobs
    """
    # Create blank image
    image = np.zeros((400, 400), dtype=np.uint8)
    
    # Add blobs with different intensities
    blob_positions = [
        (80, 80, 30, 200),    # (x, y, radius, intensity)
        (200, 100, 25, 180),
        (320, 80, 35, 220),
        (100, 220, 28, 190),
        (250, 250, 32, 210),
        (320, 300, 30, 185),
        (80, 320, 26, 195),
    ]
    
    for x, y, radius, intensity in blob_positions:
        cv2.circle(image, (x, y), radius, intensity, -1)
    
    # Add some noise
    noise = np.random.normal(0, 10, image.shape)
    image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    return image

def threshold_segmentation_manual(image, threshold_value):
    """
    Segment blobs using manual thresholding
    
    Parameters:
    image: Input grayscale image
    threshold_value: Threshold value
    
    Returns:
    Binary segmented image
    """
    _, segmented = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
    return segmented

def threshold_segmentation_otsu(image):
    """
    Segment blobs using Otsu's automatic thresholding
    
    Parameters:
    image: Input grayscale image
    
    Returns:
    Binary segmented image and threshold value
    """
    threshold_value, segmented = cv2.threshold(image, 0, 255, 
                                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return segmented, threshold_value

def threshold_segmentation_adaptive(image, block_size=11, C=2):
    """
    Segment blobs using adaptive thresholding
    
    Parameters:
    image: Input grayscale image
    block_size: Size of pixel neighborhood
    C: Constant subtracted from mean
    
    Returns:
    Binary segmented image
    """
    segmented = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, block_size, C)
    return segmented

def label_and_count_blobs(binary_image):
    """
    Label connected components and count blobs
    
    Parameters:
    binary_image: Binary segmented image
    
    Returns:
    Number of blobs, labeled image, stats
    """
    num_labels, labeled_image, stats, centroids = cv2.connectedComponentsWithStats(
        binary_image, connectivity=8)
    
    # Subtract 1 to exclude background
    num_blobs = num_labels - 1
    
    return num_blobs, labeled_image, stats, centroids

def colorize_labels(labeled_image, num_labels):
    """
    Create a color image from labeled components
    """
    # Create color map
    colors = np.random.randint(0, 255, size=(num_labels + 1, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]  # Background is black
    
    # Map labels to colors
    colored = colors[labeled_image]
    
    return colored

def analyze_blobs(stats, centroids):
    """
    Analyze blob properties
    
    Parameters:
    stats: Component statistics from connectedComponentsWithStats
    centroids: Centroids of components
    
    Returns:
    Dictionary of blob properties
    """
    blob_info = []
    
    # Skip background (index 0)
    for i in range(1, len(stats)):
        blob = {
            'id': i,
            'area': stats[i, cv2.CC_STAT_AREA],
            'left': stats[i, cv2.CC_STAT_LEFT],
            'top': stats[i, cv2.CC_STAT_TOP],
            'width': stats[i, cv2.CC_STAT_WIDTH],
            'height': stats[i, cv2.CC_STAT_HEIGHT],
            'centroid_x': centroids[i][0],
            'centroid_y': centroids[i][1]
        }
        blob_info.append(blob)
    
    return blob_info

def main():
    # Create test image with blobs
    print("Creating image with non-overlapping blobs...")
    image = create_blob_image()
    
    print("Image created!")
    print(f"Image shape: {image.shape}")
    print(f"Pixel value range: [{np.min(image)}, {np.max(image)}]")
    
    # Manual thresholding with different values
    print("\nPerforming manual thresholding...")
    threshold_values = [100, 150, 175]
    manual_results = []
    
    for thresh in threshold_values:
        segmented = threshold_segmentation_manual(image, thresh)
        num_blobs, labeled, stats, centroids = label_and_count_blobs(segmented)
        manual_results.append((thresh, segmented, labeled, num_blobs))
        print(f"  Threshold {thresh}: {num_blobs} blobs detected")
    
    # Otsu's thresholding
    print("\nPerforming Otsu's automatic thresholding...")
    segmented_otsu, otsu_threshold = threshold_segmentation_otsu(image)
    num_blobs_otsu, labeled_otsu, stats_otsu, centroids_otsu = label_and_count_blobs(segmented_otsu)
    print(f"  Otsu's threshold: {otsu_threshold:.1f}")
    print(f"  Blobs detected: {num_blobs_otsu}")
    
    # Adaptive thresholding
    print("\nPerforming adaptive thresholding...")
    segmented_adaptive = threshold_segmentation_adaptive(image, block_size=21, C=5)
    num_blobs_adaptive, labeled_adaptive, stats_adaptive, centroids_adaptive = label_and_count_blobs(segmented_adaptive)
    print(f"  Blobs detected: {num_blobs_adaptive}")
    
    # Analyze blobs from Otsu's method
    blob_info = analyze_blobs(stats_otsu, centroids_otsu)
    
    # Display results
    plt.figure(figsize=(18, 12))
    
    # Original image
    plt.subplot(3, 4, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    # Histogram
    plt.subplot(3, 4, 2)
    plt.hist(image.flatten(), bins=50, alpha=0.7, color='blue')
    plt.axvline(x=otsu_threshold, color='red', linestyle='--', label=f"Otsu's: {otsu_threshold:.1f}")
    for thresh in threshold_values:
        plt.axvline(x=thresh, linestyle=':', alpha=0.5, label=f'Manual: {thresh}')
    plt.title('Image Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Manual thresholding results
    for i, (thresh, segmented, labeled, num_blobs) in enumerate(manual_results):
        plt.subplot(3, 4, 3 + i)
        plt.imshow(segmented, cmap='gray')
        plt.title(f'Manual Threshold: {thresh}\n{num_blobs} blobs')
        plt.axis('off')
    
    # Otsu's thresholding
    plt.subplot(3, 4, 6)
    plt.imshow(segmented_otsu, cmap='gray')
    plt.title(f"Otsu's Threshold: {otsu_threshold:.1f}\n{num_blobs_otsu} blobs")
    plt.axis('off')
    
    # Adaptive thresholding
    plt.subplot(3, 4, 7)
    plt.imshow(segmented_adaptive, cmap='gray')
    plt.title(f'Adaptive Threshold\n{num_blobs_adaptive} blobs')
    plt.axis('off')
    
    # Labeled components (Otsu)
    plt.subplot(3, 4, 8)
    colored_otsu = colorize_labels(labeled_otsu, num_blobs_otsu)
    plt.imshow(colored_otsu)
    plt.title('Labeled Blobs (Otsu)')
    plt.axis('off')
    
    # Blobs with centroids
    plt.subplot(3, 4, 9)
    result_with_centroids = cv2.cvtColor(segmented_otsu.copy(), cv2.COLOR_GRAY2BGR)
    for blob in blob_info:
        cx, cy = int(blob['centroid_x']), int(blob['centroid_y'])
        cv2.circle(result_with_centroids, (cx, cy), 5, (0, 0, 255), -1)
        cv2.putText(result_with_centroids, str(blob['id']), (cx + 10, cy),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    plt.imshow(cv2.cvtColor(result_with_centroids, cv2.COLOR_BGR2RGB))
    plt.title('Blob Centroids')
    plt.axis('off')
    
    # Blob size distribution
    plt.subplot(3, 4, 10)
    areas = [blob['area'] for blob in blob_info]
    plt.bar(range(1, len(areas) + 1), areas, color='skyblue')
    plt.title('Blob Area Distribution')
    plt.xlabel('Blob ID')
    plt.ylabel('Area (pixels)')
    plt.grid(True, alpha=0.3)
    
    # Comparison of methods
    plt.subplot(3, 4, 11)
    methods = ['Manual\n(100)', 'Manual\n(150)', 'Manual\n(175)', 'Otsu', 'Adaptive']
    blob_counts = [manual_results[0][3], manual_results[1][3], manual_results[2][3], 
                   num_blobs_otsu, num_blobs_adaptive]
    plt.bar(methods, blob_counts, color=['lightcoral', 'lightcoral', 'lightcoral', 
                                         'lightgreen', 'lightblue'])
    plt.title('Blob Count Comparison')
    plt.ylabel('Number of Blobs')
    plt.grid(True, alpha=0.3)
    
    # Blob overlay on original
    plt.subplot(3, 4, 12)
    overlay = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(segmented_otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title('Blob Boundaries')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print blob analysis
    print("\nBlob Analysis (Otsu's Method):")
    print("ID | Area  | Center (x,y)    | Size (w×h)")
    print("-" * 50)
    for blob in blob_info:
        print(f"{blob['id']:2d} | {blob['area']:5d} | "
              f"({blob['centroid_x']:6.1f}, {blob['centroid_y']:6.1f}) | "
              f"{blob['width']:3d}×{blob['height']:3d}")
    
    print(f"\nTotal blobs detected: {len(blob_info)}")
    print(f"Average blob area: {np.mean(areas):.1f} pixels")
    print(f"Blob area range: [{np.min(areas)}, {np.max(areas)}] pixels")

if __name__ == "__main__":
    main()