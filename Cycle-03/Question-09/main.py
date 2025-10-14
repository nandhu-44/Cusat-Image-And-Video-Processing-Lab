import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

def create_blob_binary_image():
    """
    Create a binary image with small blobs
    """
    image = np.zeros((400, 400), dtype=np.uint8)
    
    # Add circular blobs
    blob_positions = [
        (80, 80, 30),
        (200, 100, 25),
        (320, 80, 35),
        (100, 220, 28),
        (250, 250, 32),
        (320, 300, 30),
        (80, 320, 26),
        (180, 180, 22),
    ]
    
    for x, y, radius in blob_positions:
        cv2.circle(image, (x, y), radius, 255, -1)
    
    return image

def compute_distance_transform(binary_image):
    """
    Compute Euclidean distance transform
    
    Parameters:
    binary_image: Binary input image
    
    Returns:
    Distance transform
    """
    dist_transform = cv2.distanceTransform(binary_image, cv2.DIST_L2, 5)
    return dist_transform

def watershed_segmentation(binary_image, min_distance=10):
    """
    Segment blobs using watershed transform
    
    Parameters:
    binary_image: Binary input image
    min_distance: Minimum distance between peaks (blob centers)
    
    Returns:
    Labeled image, number of blobs
    """
    # Compute distance transform
    dist_transform = compute_distance_transform(binary_image)
    
    # Find local maxima (peaks) - these will be markers
    # Use morphological dilation to find peaks
    local_max = ndimage.maximum_filter(dist_transform, size=min_distance) == dist_transform
    
    # Remove background
    local_max = local_max & (binary_image > 0)
    
    # Label the local maxima
    markers, num_markers = ndimage.label(local_max)
    
    # Apply watershed transform
    # Convert to 8-bit for watershed
    dist_transform_8u = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Invert distance transform for watershed (watershed works on basins)
    dist_transform_inv = 255 - dist_transform_8u
    
    # Apply watershed
    markers_copy = markers.copy()
    labels = cv2.watershed(cv2.cvtColor(dist_transform_inv, cv2.COLOR_GRAY2BGR), markers_copy)
    
    # Watershed marks boundaries as -1
    labels[labels == -1] = 0
    
    return labels, num_markers

def watershed_segmentation_markers(binary_image, sure_fg_threshold=0.7):
    """
    Watershed segmentation with automatic marker generation
    
    Parameters:
    binary_image: Binary input image
    sure_fg_threshold: Threshold ratio for sure foreground (0-1)
    
    Returns:
    Labeled image, number of segments
    """
    # Distance transform
    dist_transform = cv2.distanceTransform(binary_image, cv2.DIST_L2, 5)
    
    # Threshold to get sure foreground
    threshold_value = sure_fg_threshold * dist_transform.max()
    _, sure_fg = cv2.threshold(dist_transform, threshold_value, 255, cv2.THRESH_BINARY)
    sure_fg = sure_fg.astype(np.uint8)
    
    # Find sure background (dilate binary image)
    kernel = np.ones((3, 3), np.uint8)
    sure_bg = cv2.dilate(binary_image, kernel, iterations=3)
    
    # Find unknown region
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Label sure foreground
    _, markers = cv2.connectedComponents(sure_fg)
    
    # Add 1 to all labels (background becomes 1 instead of 0)
    markers = markers + 1
    
    # Mark unknown region as 0
    markers[unknown == 255] = 0
    
    # Apply watershed
    img_color = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(img_color, markers)
    
    # Get number of segments (exclude background and boundaries)
    num_segments = len(np.unique(markers)) - 2  # -1 for background, -1 for boundaries
    
    return markers, num_segments, sure_fg, sure_bg, unknown

def colorize_watershed(labels):
    """
    Create a colored visualization of watershed results
    """
    # Create color map
    num_labels = labels.max()
    colors = np.random.randint(0, 255, size=(num_labels + 1, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]  # Background is black
    
    # Create colored image
    colored = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
    
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            if labels[i, j] > 0:
                colored[i, j] = colors[labels[i, j]]
    
    # Mark boundaries in white
    colored[labels == -1] = [255, 255, 255]
    
    return colored

def analyze_watershed_regions(labels, original_image):
    """
    Analyze properties of segmented regions
    """
    region_info = []
    unique_labels = np.unique(labels)
    
    for label in unique_labels:
        if label <= 0:  # Skip background and boundaries
            continue
        
        mask = (labels == label).astype(np.uint8)
        area = np.sum(mask)
        
        if area > 0:
            coords = np.argwhere(mask > 0)
            cy, cx = coords.mean(axis=0)
            
            region_info.append({
                'label': label,
                'area': area,
                'centroid': (cx, cy)
            })
    
    return region_info

def main():
    # Create binary image with blobs
    print("Creating binary image with blobs...")
    binary_image = create_blob_binary_image()
    
    print("Binary image created!")
    print(f"Image shape: {binary_image.shape}")
    
    # Method 1: Simple watershed with distance peaks
    print("\nMethod 1: Watershed with distance transform peaks...")
    labels1, num_blobs1 = watershed_segmentation(binary_image, min_distance=15)
    print(f"  Detected {num_blobs1} blobs")
    
    # Method 2: Watershed with automatic markers
    print("\nMethod 2: Watershed with automatic marker generation...")
    labels2, num_blobs2, sure_fg, sure_bg, unknown = watershed_segmentation_markers(binary_image, sure_fg_threshold=0.5)
    print(f"  Detected {num_blobs2} blobs")
    
    # Method 3: Watershed with different threshold
    print("\nMethod 3: Watershed with higher threshold...")
    labels3, num_blobs3, sure_fg3, sure_bg3, unknown3 = watershed_segmentation_markers(binary_image, sure_fg_threshold=0.7)
    print(f"  Detected {num_blobs3} blobs")
    
    # Compute distance transform for visualization
    dist_transform = compute_distance_transform(binary_image)
    
    # Analyze regions
    region_info = analyze_watershed_regions(labels2, binary_image)
    
    # Colorize results
    colored1 = colorize_watershed(labels1)
    colored2 = colorize_watershed(labels2)
    colored3 = colorize_watershed(labels3)
    
    # Display results
    plt.figure(figsize=(18, 14))
    
    # Original binary image
    plt.subplot(4, 4, 1)
    plt.imshow(binary_image, cmap='gray')
    plt.title('Original Binary Image')
    plt.axis('off')
    
    # Distance transform
    plt.subplot(4, 4, 2)
    plt.imshow(dist_transform, cmap='jet')
    plt.title('Distance Transform')
    plt.colorbar()
    plt.axis('off')
    
    # Sure foreground (Method 2)
    plt.subplot(4, 4, 3)
    plt.imshow(sure_fg, cmap='gray')
    plt.title('Sure Foreground\n(Markers)')
    plt.axis('off')
    
    # Unknown region
    plt.subplot(4, 4, 4)
    plt.imshow(unknown, cmap='gray')
    plt.title('Unknown Region')
    plt.axis('off')
    
    # Method 1 results
    plt.subplot(4, 4, 5)
    plt.imshow(colored1)
    plt.title(f'Method 1: Distance Peaks\n{num_blobs1} blobs')
    plt.axis('off')
    
    plt.subplot(4, 4, 6)
    boundary1 = binary_image.copy()
    boundary1[labels1 == -1] = 128
    plt.imshow(boundary1, cmap='gray')
    plt.title('Boundaries (Method 1)')
    plt.axis('off')
    
    # Method 2 results
    plt.subplot(4, 4, 7)
    plt.imshow(colored2)
    plt.title(f'Method 2: Auto Markers (T=0.5)\n{num_blobs2} blobs')
    plt.axis('off')
    
    plt.subplot(4, 4, 8)
    boundary2 = binary_image.copy()
    boundary2[labels2 == -1] = 128
    plt.imshow(boundary2, cmap='gray')
    plt.title('Boundaries (Method 2)')
    plt.axis('off')
    
    # Method 3 results
    plt.subplot(4, 4, 9)
    plt.imshow(colored3)
    plt.title(f'Method 3: Auto Markers (T=0.7)\n{num_blobs3} blobs')
    plt.axis('off')
    
    plt.subplot(4, 4, 10)
    boundary3 = binary_image.copy()
    boundary3[labels3 == -1] = 128
    plt.imshow(boundary3, cmap='gray')
    plt.title('Boundaries (Method 3)')
    plt.axis('off')
    
    # Overlay with centroids
    plt.subplot(4, 4, 11)
    overlay = cv2.cvtColor(binary_image.copy(), cv2.COLOR_GRAY2BGR)
    overlay[labels2 == -1] = [0, 255, 0]  # Green boundaries
    
    for region in region_info:
        cx, cy = int(region['centroid'][0]), int(region['centroid'][1])
        cv2.circle(overlay, (cx, cy), 5, (0, 0, 255), -1)
        cv2.putText(overlay, str(region['label']), (cx + 10, cy),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title('Boundaries & Centroids')
    plt.axis('off')
    
    # Blob area distribution
    plt.subplot(4, 4, 12)
    areas = [region['area'] for region in region_info]
    plt.bar(range(1, len(areas) + 1), areas, color='skyblue')
    plt.title('Blob Area Distribution')
    plt.xlabel('Blob ID')
    plt.ylabel('Area (pixels)')
    plt.grid(True, alpha=0.3)
    
    # Comparison of methods
    plt.subplot(4, 4, 13)
    methods = ['Distance\nPeaks', 'Auto\n(T=0.5)', 'Auto\n(T=0.7)']
    blob_counts = [num_blobs1, num_blobs2, num_blobs3]
    plt.bar(methods, blob_counts, color=['lightcoral', 'lightgreen', 'lightblue'])
    plt.title('Blob Count Comparison')
    plt.ylabel('Number of Blobs')
    plt.grid(True, alpha=0.3)
    
    # 3D visualization of distance transform
    from mpl_toolkits.mplot3d import Axes3D
    ax = plt.subplot(4, 4, 14, projection='3d')
    
    # Downsample for visualization
    step = 10
    x = np.arange(0, dist_transform.shape[1], step)
    y = np.arange(0, dist_transform.shape[0], step)
    X, Y = np.meshgrid(x, y)
    Z = dist_transform[::step, ::step]
    
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    ax.set_title('Distance Transform (3D)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Distance')
    
    # Histogram of blob sizes
    plt.subplot(4, 4, 15)
    if areas:
        plt.hist(areas, bins=15, alpha=0.7, color='purple')
        plt.title('Blob Size Distribution')
        plt.xlabel('Area (pixels)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
    
    # Process steps visualization
    plt.subplot(4, 4, 16)
    steps_img = np.hstack([
        cv2.resize(binary_image, (100, 100)),
        cv2.resize(sure_fg, (100, 100)),
        cv2.resize(unknown, (100, 100)),
        cv2.resize(boundary2, (100, 100))
    ])
    plt.imshow(steps_img, cmap='gray')
    plt.title('Process: Input → Markers → Unknown → Result')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print region analysis
    print("\nBlob Analysis (Method 2):")
    print("ID | Area  | Centroid (x,y)")
    print("-" * 40)
    for region in region_info:
        print(f"{region['label']:2d} | {region['area']:5d} | "
              f"({region['centroid'][0]:6.1f}, {region['centroid'][1]:6.1f})")
    
    print(f"\nTotal blobs detected: {len(region_info)}")
    if areas:
        print(f"Average blob area: {np.mean(areas):.1f} pixels")
        print(f"Blob area range: [{np.min(areas)}, {np.max(areas)}] pixels")
    
    print("\nWatershed Transform Summary:")
    print(f"  Method 1 (Distance Peaks): {num_blobs1} blobs")
    print(f"  Method 2 (Auto T=0.5): {num_blobs2} blobs")
    print(f"  Method 3 (Auto T=0.7): {num_blobs3} blobs")

if __name__ == "__main__":
    main()