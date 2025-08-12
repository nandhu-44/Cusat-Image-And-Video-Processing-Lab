import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

def sobel_edge_detection(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    
    edge_x = cv2.filter2D(image.astype(np.float32), -1, sobel_x)
    edge_y = cv2.filter2D(image.astype(np.float32), -1, sobel_y)
    
    edge_magnitude = np.sqrt(edge_x**2 + edge_y**2)
    return np.clip(edge_magnitude, 0, 255).astype(np.uint8)

def prewitt_edge_detection(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    
    edge_x = cv2.filter2D(image.astype(np.float32), -1, prewitt_x)
    edge_y = cv2.filter2D(image.astype(np.float32), -1, prewitt_y)
    
    edge_magnitude = np.sqrt(edge_x**2 + edge_y**2)
    return np.clip(edge_magnitude, 0, 255).astype(np.uint8)

def roberts_edge_detection(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    roberts_x = np.array([[1, 0], [0, -1]])
    roberts_y = np.array([[0, 1], [-1, 0]])
    
    edge_x = cv2.filter2D(image.astype(np.float32), -1, roberts_x)
    edge_y = cv2.filter2D(image.astype(np.float32), -1, roberts_y)
    
    edge_magnitude = np.sqrt(edge_x**2 + edge_y**2)
    return np.clip(edge_magnitude, 0, 255).astype(np.uint8)

def log_edge_detection(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Gaussian blur followed by Laplacian
    blurred = cv2.GaussianBlur(image, (5, 5), 1.0)
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    
    # Zero crossing detection
    edges = np.zeros_like(laplacian)
    for i in range(1, laplacian.shape[0]-1):
        for j in range(1, laplacian.shape[1]-1):
            # Check for zero crossings
            neighbors = laplacian[i-1:i+2, j-1:j+2]
            if (neighbors.max() > 0 and neighbors.min() < 0):
                edges[i, j] = 255
    
    return edges.astype(np.uint8)

def canny_edge_detection(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    return cv2.Canny(image, 50, 150)

def calculate_performance_metrics(edge_image):
    # Edge density (percentage of edge pixels)
    edge_density = (np.sum(edge_image > 0) / edge_image.size) * 100
    
    # Edge strength (average intensity of edge pixels)
    edge_pixels = edge_image[edge_image > 0]
    edge_strength = np.mean(edge_pixels) if len(edge_pixels) > 0 else 0
    
    # Edge continuity (measure of connected components)
    binary_edges = (edge_image > 0).astype(np.uint8)
    num_components, _ = cv2.connectedComponents(binary_edges)
    edge_continuity = edge_density / max(num_components - 1, 1)  # -1 to exclude background
    
    return {
        'edge_density': edge_density,
        'edge_strength': edge_strength,
        'edge_continuity': edge_continuity,
        'num_components': num_components - 1
    }

def main():
    image = cv2.imread("../images/lena.jpg")
    
    # Apply all edge detection methods
    sobel_edges = sobel_edge_detection(image)
    prewitt_edges = prewitt_edge_detection(image)
    roberts_edges = roberts_edge_detection(image)
    log_edges = log_edge_detection(image)
    canny_edges = canny_edge_detection(image)
    
    # Display results
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    if len(image.shape) == 3:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(sobel_edges, cmap='gray')
    plt.title('Sobel Edge Detection')
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.imshow(prewitt_edges, cmap='gray')
    plt.title('Prewitt Edge Detection')
    plt.axis('off')
    
    plt.subplot(2, 3, 4)
    plt.imshow(roberts_edges, cmap='gray')
    plt.title('Roberts Edge Detection')
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    plt.imshow(log_edges, cmap='gray')
    plt.title('LoG Edge Detection')
    plt.axis('off')
    
    plt.subplot(2, 3, 6)
    plt.imshow(canny_edges, cmap='gray')
    plt.title('Canny Edge Detection')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Calculate and display performance metrics
    methods = {
        'Sobel': sobel_edges,
        'Prewitt': prewitt_edges,
        'Roberts': roberts_edges,
        'LoG': log_edges,
        'Canny': canny_edges
    }
    
    print("\nEdge Detection Performance Comparison:")
    print("=" * 60)
    print(f"{'Method':<10} {'Density(%)':<12} {'Strength':<10} {'Components':<12} {'Continuity':<10}")
    print("-" * 60)
    
    for method_name, edge_result in methods.items():
        metrics = calculate_performance_metrics(edge_result)
        print(f"{method_name:<10} {metrics['edge_density']:<12.2f} {metrics['edge_strength']:<10.1f} "
              f"{metrics['num_components']:<12} {metrics['edge_continuity']:<10.2f}")
    
    print("\nMetrics Explanation:")
    print("- Density: Percentage of pixels detected as edges")
    print("- Strength: Average intensity of edge pixels")
    print("- Components: Number of separate edge regions")
    print("- Continuity: Edge density per component (higher = better connected edges)")

if __name__ == "__main__":
    main()
