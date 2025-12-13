import cv2
import os
import matplotlib.pyplot as plt

path = "../images/lenna.jpg"
img = cv2.imread(path)
negative = 255 - img

print("Image Information:")
print(f"Height: {img.shape[0]}")
print(f"Width: {img.shape[1]}")
print(f"Channels: {img.shape[2] if img.ndim > 2 else 1}")
print(f"Data Type: {img.dtype}")
print(f"Size in Memory: {img.nbytes} bytes")
print(f"Shape: {img.shape}")

file_size = os.path.getsize(path)
memory_size = img.nbytes
print("\nCompression Analysis:")
print(f"File Size on Disk: {file_size} bytes")
print(f"Size in Memory: {memory_size} bytes")
print(f"Compression Ratio: {memory_size/file_size:.2f}")
print(f"Space Saved: {((memory_size - file_size) / memory_size) * 100:.2f}%")

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1); plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); plt.title('Original'); plt.axis('off')
plt.subplot(1, 2, 2); plt.imshow(cv2.cvtColor(negative, cv2.COLOR_BGR2RGB)); plt.title('Negative'); plt.axis('off')
plt.savefig('output.png', bbox_inches='tight')
plt.show()
