import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_test_image():
    """
    Create an image suitable for quad-tree segmentation
    """
    image = np.zeros((256, 256), dtype=np.uint8)
    
    # Create regions with different intensities
    image[0:128, 0:128] = 50      # Top-left
    image[0:128, 128:256] = 150   # Top-right
    image[128:256, 0:128] = 200   # Bottom-left
    image[128:256, 128:256] = 100 # Bottom-right
    
    # Add some texture to regions
    image[20:60, 20:60] = 80
    image[40:80, 160:200] = 180
    image[160:200, 40:80] = 220
    image[180:220, 180:220] = 130
    
    # Add small details
    image[30:40, 30:40] = 120
    image[170:180, 170:180] = 150
    
    # Add some noise
    noise = np.random.normal(0, 5, image.shape)
    image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    return image

class QuadTreeNode:
    """
    Node in the quad-tree structure
    """
    def __init__(self, x, y, width, height, mean_value):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.mean_value = mean_value
        self.children = []
        self.is_leaf = True
    
    def split(self):
        """Mark this node as non-leaf"""
        self.is_leaf = False

def calculate_variance(image, x, y, width, height):
    """
    Calculate variance of a region
    """
    region = image[y:y+height, x:x+width]
    return np.var(region)

def calculate_mean(image, x, y, width, height):
    """
    Calculate mean value of a region
    """
    region = image[y:y+height, x:x+width]
    return np.mean(region)

def split_region(image, x, y, width, height, min_size, threshold):
    """
    Recursively split a region based on variance threshold
    
    Parameters:
    image: Input image
    x, y: Top-left corner of region
    width, height: Size of region
    min_size: Minimum dimension for splitting
    threshold: Variance threshold for homogeneity
    
    Returns:
    QuadTreeNode representing the region
    """
    mean_value = calculate_mean(image, x, y, width, height)
    node = QuadTreeNode(x, y, width, height, mean_value)
    
    # Check if region is too small
    if width <= min_size or height <= min_size:
        return node
    
    # Check if region is homogeneous
    variance = calculate_variance(image, x, y, width, height)
    
    if variance <= threshold:
        return node
    
    # Split into four quadrants
    node.split()
    half_width = width // 2
    half_height = height // 2
    
    # Top-left
    node.children.append(split_region(image, x, y, half_width, half_height, min_size, threshold))
    # Top-right
    node.children.append(split_region(image, x + half_width, y, width - half_width, half_height, min_size, threshold))
    # Bottom-left
    node.children.append(split_region(image, x, y + half_height, half_width, height - half_height, min_size, threshold))
    # Bottom-right
    node.children.append(split_region(image, x + half_width, y + half_height, 
                                     width - half_width, height - half_height, min_size, threshold))
    
    return node

def merge_similar_regions(node, threshold=20):
    """
    Merge similar adjacent regions (simplified version)
    """
    if node.is_leaf:
        return node
    
    # Recursively process children
    for i in range(len(node.children)):
        node.children[i] = merge_similar_regions(node.children[i], threshold)
    
    # Check if all children are leaves and similar
    if all(child.is_leaf for child in node.children):
        values = [child.mean_value for child in node.children]
        if max(values) - min(values) <= threshold:
            # Merge children
            node.is_leaf = True
            node.children = []
            node.mean_value = np.mean(values)
    
    return node

def draw_quadtree(image, node, color=(0, 255, 0), thickness=1):
    """
    Draw quad-tree boundaries on image
    """
    if node.is_leaf:
        cv2.rectangle(image, (node.x, node.y), 
                     (node.x + node.width, node.y + node.height), 
                     color, thickness)
    else:
        for child in node.children:
            draw_quadtree(image, child, color, thickness)

def create_segmented_image(image_shape, node):
    """
    Create segmented image from quad-tree
    """
    segmented = np.zeros(image_shape, dtype=np.uint8)
    
    def fill_region(node):
        if node.is_leaf:
            segmented[node.y:node.y+node.height, node.x:node.x+node.width] = int(node.mean_value)
        else:
            for child in node.children:
                fill_region(child)
    
    fill_region(node)
    return segmented

def count_regions(node):
    """
    Count number of leaf regions in quad-tree
    """
    if node.is_leaf:
        return 1
    return sum(count_regions(child) for child in node.children)

def get_leaf_sizes(node):
    """
    Get sizes of all leaf regions
    """
    if node.is_leaf:
        return [(node.width, node.height)]
    
    sizes = []
    for child in node.children:
        sizes.extend(get_leaf_sizes(child))
    return sizes

def main():
    # Create test image
    print("Creating test image...")
    image = create_test_image()
    
    print("Image created!")
    print(f"Image shape: {image.shape}")
    
    # Test with different minimum sizes
    min_sizes = [4, 8, 16]
    threshold = 100  # Variance threshold
    
    results = []
    
    print("\nPerforming split and merge segmentation...")
    for min_size in min_sizes:
        print(f"\n--- Min size: {min_size} ---")
        
        # Split phase
        root = split_region(image, 0, 0, image.shape[1], image.shape[0], min_size, threshold)
        num_regions_split = count_regions(root)
        print(f"  After split: {num_regions_split} regions")
        
        # Create segmented image after split
        segmented_split = create_segmented_image(image.shape, root)
        
        # Merge phase
        root_merged = merge_similar_regions(root, threshold=30)
        num_regions_merged = count_regions(root_merged)
        print(f"  After merge: {num_regions_merged} regions")
        
        # Create final segmented image
        segmented_final = create_segmented_image(image.shape, root_merged)
        
        # Get leaf sizes
        leaf_sizes = get_leaf_sizes(root_merged)
        
        results.append({
            'min_size': min_size,
            'root_split': root,
            'root_merged': root_merged,
            'segmented_split': segmented_split,
            'segmented_final': segmented_final,
            'num_regions_split': num_regions_split,
            'num_regions_merged': num_regions_merged,
            'leaf_sizes': leaf_sizes
        })
    
    # Display results
    plt.figure(figsize=(18, 14))
    
    # Original image
    plt.subplot(4, 4, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    # Histogram
    plt.subplot(4, 4, 2)
    plt.hist(image.flatten(), bins=50, alpha=0.7, color='blue')
    plt.title('Image Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Results for each minimum size
    for i, result in enumerate(results):
        min_size = result['min_size']
        
        # After split
        plt.subplot(4, 4, 3 + i)
        img_with_tree = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)
        draw_quadtree(img_with_tree, result['root_split'], color=(0, 255, 0), thickness=1)
        plt.imshow(cv2.cvtColor(img_with_tree, cv2.COLOR_BGR2RGB))
        plt.title(f'Split (min={min_size})\n{result["num_regions_split"]} regions')
        plt.axis('off')
        
        # After merge
        plt.subplot(4, 4, 6 + i)
        img_with_tree = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)
        draw_quadtree(img_with_tree, result['root_merged'], color=(255, 0, 0), thickness=2)
        plt.imshow(cv2.cvtColor(img_with_tree, cv2.COLOR_BGR2RGB))
        plt.title(f'Split+Merge (min={min_size})\n{result["num_regions_merged"]} regions')
        plt.axis('off')
        
        # Segmented after split
        plt.subplot(4, 4, 9 + i)
        plt.imshow(result['segmented_split'], cmap='gray')
        plt.title(f'Segmented (Split)')
        plt.axis('off')
        
        # Final segmented
        plt.subplot(4, 4, 12 + i)
        plt.imshow(result['segmented_final'], cmap='gray')
        plt.title(f'Segmented (Final)')
        plt.axis('off')
    
    # Comparison chart
    plt.subplot(4, 4, 15)
    min_size_labels = [f'Min={r["min_size"]}' for r in results]
    split_counts = [r['num_regions_split'] for r in results]
    merged_counts = [r['num_regions_merged'] for r in results]
    
    x = np.arange(len(min_size_labels))
    width = 0.35
    
    plt.bar(x - width/2, split_counts, width, label='After Split', color='lightgreen')
    plt.bar(x + width/2, merged_counts, width, label='After Merge', color='lightcoral')
    plt.xlabel('Minimum Size')
    plt.ylabel('Number of Regions')
    plt.title('Region Count Comparison')
    plt.xticks(x, min_size_labels)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Region size distribution for middle result
    plt.subplot(4, 4, 16)
    middle_result = results[1]
    sizes = [w * h for w, h in middle_result['leaf_sizes']]
    plt.hist(sizes, bins=20, alpha=0.7, color='purple')
    plt.title(f'Region Size Distribution\n(min={middle_result["min_size"]})')
    plt.xlabel('Region Area (pixels)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print("\n" + "="*60)
    print("SEGMENTATION SUMMARY")
    print("="*60)
    for result in results:
        print(f"\nMinimum size: {result['min_size']}x{result['min_size']}")
        print(f"  Regions after split: {result['num_regions_split']}")
        print(f"  Regions after merge: {result['num_regions_merged']}")
        print(f"  Reduction: {result['num_regions_split'] - result['num_regions_merged']} regions")
        
        sizes = [w * h for w, h in result['leaf_sizes']]
        if sizes:
            print(f"  Average region size: {np.mean(sizes):.1f} pixels")
            print(f"  Region size range: [{np.min(sizes)}, {np.max(sizes)}] pixels")

if __name__ == "__main__":
    main()