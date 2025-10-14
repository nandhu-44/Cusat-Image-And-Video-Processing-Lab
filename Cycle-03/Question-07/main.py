import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

def create_blob_image():
    """
    Create an image with small, non-overlapping blobs
    """
    image = np.zeros((400, 400), dtype=np.uint8)
    
    # Add blobs with different intensities
    blob_positions = [
        (80, 80, 30, 200),
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

def region_growing(image, seed_point, threshold=20):
    """
    Segment a region using region growing algorithm
    
    Parameters:
    image: Input grayscale image
    seed_point: Starting point (x, y) for region growing
    threshold: Maximum intensity difference for region membership
    
    Returns:
    Binary mask of the segmented region
    """
    height, width = image.shape
    segmented = np.zeros((height, width), dtype=np.uint8)
    visited = np.zeros((height, width), dtype=bool)
    
    # Queue for BFS
    queue = deque([seed_point])
    seed_value = image[seed_point[1], seed_point[0]]
    
    # 8-connectivity neighbors
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    
    while queue:
        x, y = queue.popleft()
        
        if visited[y, x]:
            continue
        
        visited[y, x] = True
        current_value = image[y, x]
        
        # Check if pixel belongs to region
        if abs(int(current_value) - int(seed_value)) <= threshold:
            segmented[y, x] = 255
            
            # Add neighbors to queue
            for dx, dy in neighbors:
                nx, ny = x + dx, y + dy
                
                if 0 <= nx < width and 0 <= ny < height and not visited[ny, nx]:
                    queue.append((nx, ny))
    
    return segmented

def region_growing_multiple_seeds(image, threshold=20):
    """
    Segment multiple regions using automatic seed detection
    
    Parameters:
    image: Input grayscale image
    threshold: Maximum intensity difference for region membership
    
    Returns:
    Labeled image with different regions
    """
    # Apply thresholding to find potential regions
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find connected components to get seed points
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    height, width = image.shape
    segmented = np.zeros((height, width), dtype=np.uint8)
    labeled_regions = np.zeros((height, width), dtype=np.int32)
    
    # Grow regions from each centroid
    region_id = 1
    for i in range(1, num_labels):  # Skip background
        cx, cy = int(centroids[i][0]), int(centroids[i][1])
        
        # Grow region from this seed
        region_mask = region_growing(image, (cx, cy), threshold)
        
        # Add to labeled image
        labeled_regions[region_mask > 0] = region_id
        segmented[region_mask > 0] = 255
        region_id += 1
    
    return segmented, labeled_regions, num_labels - 1

def adaptive_region_growing(image, threshold_ratio=0.15):
    """
    Region growing with adaptive threshold based on local statistics
    
    Parameters:
    image: Input grayscale image
    threshold_ratio: Ratio of local standard deviation to use as threshold
    
    Returns:
    Segmented image and labeled regions
    """
    # Find seed points
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    height, width = image.shape
    labeled_regions = np.zeros((height, width), dtype=np.int32)
    
    region_id = 1
    for i in range(1, num_labels):
        cx, cy = int(centroids[i][0]), int(centroids[i][1])
        
        # Calculate local statistics for adaptive threshold
        local_region = image[max(0, cy-20):min(height, cy+20), 
                           max(0, cx-20):min(width, cx+20)]
        local_std = np.std(local_region)
        adaptive_threshold = max(10, int(local_std * threshold_ratio))
        
        # Grow region with adaptive threshold
        region_mask = region_growing(image, (cx, cy), adaptive_threshold)
        labeled_regions[region_mask > 0] = region_id
        region_id += 1
    
    return labeled_regions

def colorize_labels(labeled_image, num_labels):
    """
    Create a color image from labeled components
    """
    colors = np.random.randint(0, 255, size=(num_labels + 1, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]  # Background is black
    colored = colors[labeled_image]
    return colored

def analyze_regions(labeled_image, original_image):
    """
    Analyze properties of segmented regions
    """
    region_info = []
    num_regions = np.max(labeled_image)
    
    for region_id in range(1, num_regions + 1):
        mask = (labeled_image == region_id).astype(np.uint8)
        
        # Calculate properties
        area = np.sum(mask)
        coords = np.argwhere(mask > 0)
        
        if len(coords) > 0:
            min_y, min_x = coords.min(axis=0)
            max_y, max_x = coords.max(axis=0)
            
            # Calculate centroid
            cy, cx = coords.mean(axis=0)
            
            # Calculate mean intensity
            mean_intensity = np.mean(original_image[mask > 0])
            
            region_info.append({
                'id': region_id,
                'area': area,
                'centroid': (cx, cy),
                'bounds': (min_x, min_y, max_x - min_x, max_y - min_y),
                'mean_intensity': mean_intensity
            })
    
    return region_info

def main():
    # Create test image with blobs
    print("Creating image with non-overlapping blobs...")
    image = create_blob_image()
    
    print("Image created!")
    print(f"Image shape: {image.shape}")
    
    # Segment using region growing with different thresholds
    print("\nPerforming region growing segmentation...")
    threshold_values = [15, 25, 35]
    
    results = []
    for threshold in threshold_values:
        segmented, labeled, num_regions = region_growing_multiple_seeds(image, threshold)
        results.append((threshold, segmented, labeled, num_regions))
        print(f"  Threshold {threshold}: {num_regions} regions detected")
    
    # Adaptive region growing
    print("\nPerforming adaptive region growing...")
    labeled_adaptive = adaptive_region_growing(image, threshold_ratio=0.2)
    num_adaptive = np.max(labeled_adaptive)
    print(f"  Adaptive method: {num_adaptive} regions detected")
    
    # Analyze regions (using middle threshold)
    region_info = analyze_regions(results[1][2], image)
    
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
    plt.title('Image Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Region growing results with different thresholds
    for i, (threshold, segmented, labeled, num_regions) in enumerate(results):
        # Binary segmentation
        plt.subplot(3, 4, 3 + i)
        plt.imshow(segmented, cmap='gray')
        plt.title(f'Threshold: {threshold}\n{num_regions} regions')
        plt.axis('off')
        
        # Colored labels
        plt.subplot(3, 4, 6 + i)
        colored = colorize_labels(labeled, num_regions)
        plt.imshow(colored)
        plt.title(f'Labeled Regions (T={threshold})')
        plt.axis('off')
    
    # Adaptive region growing
    plt.subplot(3, 4, 9)
    colored_adaptive = colorize_labels(labeled_adaptive, num_adaptive)
    plt.imshow(colored_adaptive)
    plt.title(f'Adaptive Growing\n{num_adaptive} regions')
    plt.axis('off')
    
    # Regions with boundaries
    plt.subplot(3, 4, 10)
    result_overlay = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)
    segmented_best = results[1][1]  # Use middle threshold
    contours, _ = cv2.findContours(segmented_best, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result_overlay, contours, -1, (0, 255, 0), 2)
    
    # Draw centroids
    for region in region_info:
        cx, cy = int(region['centroid'][0]), int(region['centroid'][1])
        cv2.circle(result_overlay, (cx, cy), 5, (0, 0, 255), -1)
        cv2.putText(result_overlay, str(region['id']), (cx + 10, cy),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    plt.imshow(cv2.cvtColor(result_overlay, cv2.COLOR_BGR2RGB))
    plt.title('Region Boundaries & Centroids')
    plt.axis('off')
    
    # Region size distribution
    plt.subplot(3, 4, 11)
    areas = [region['area'] for region in region_info]
    plt.bar(range(1, len(areas) + 1), areas, color='skyblue')
    plt.title('Region Area Distribution')
    plt.xlabel('Region ID')
    plt.ylabel('Area (pixels)')
    plt.grid(True, alpha=0.3)
    
    # Comparison of methods
    plt.subplot(3, 4, 12)
    methods = [f'T={results[0][0]}', f'T={results[1][0]}', f'T={results[2][0]}', 'Adaptive']
    region_counts = [results[0][3], results[1][3], results[2][3], num_adaptive]
    plt.bar(methods, region_counts, color=['lightcoral', 'lightgreen', 'lightblue', 'gold'])
    plt.title('Region Count Comparison')
    plt.ylabel('Number of Regions')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print region analysis
    print("\nRegion Analysis (Threshold=25):")
    print("ID | Area  | Centroid (x,y)   | Mean Intensity")
    print("-" * 55)
    for region in region_info:
        print(f"{region['id']:2d} | {region['area']:5d} | "
              f"({region['centroid'][0]:6.1f}, {region['centroid'][1]:6.1f}) | "
              f"{region['mean_intensity']:6.1f}")
    
    print(f"\nTotal regions detected: {len(region_info)}")
    if areas:
        print(f"Average region area: {np.mean(areas):.1f} pixels")
        print(f"Region area range: [{np.min(areas)}, {np.max(areas)}] pixels")

if __name__ == "__main__":
    main()