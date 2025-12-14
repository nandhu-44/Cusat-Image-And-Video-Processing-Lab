import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
img = cv2.imread('../images/elephant.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (256, 256))

# RGB to HSI
def rgb2hsi(rgb_img):
    rgb = rgb_img.astype(np.float32) / 255.0
    R = rgb[:,:,0]
    G = rgb[:,:,1]
    B = rgb[:,:,2]
    
    I = (R + G + B) / 3.0
    
    min_rgb = np.minimum(np.minimum(R, G), B)
    S = 1 - (min_rgb / (I + 1e-6))
    S[I == 0] = 0
    
    num = 0.5 * ((R - G) + (R - B))
    den = np.sqrt((R - G)**2 + (R - B) * (G - B))
    theta = np.arccos(num / (den + 1e-6))
    
    H = theta.copy()
    H[B > G] = 2*np.pi - H[B > G]
    H = H / (2*np.pi)
    
    return H, S, I

# HSI to RGB
def hsi2rgb(H, S, I):
    H = H * 2 * np.pi
    R = np.zeros_like(H)
    G = np.zeros_like(H)
    B = np.zeros_like(H)
    
    # Sector 1: 0 <= H < 120 (2pi/3)
    mask = (H < 2*np.pi/3)
    B[mask] = I[mask] * (1 - S[mask])
    R[mask] = I[mask] * (1 + S[mask]*np.cos(H[mask]) / np.cos(np.pi/3 - H[mask]))
    G[mask] = 3*I[mask] - (R[mask] + B[mask])
    
    # Sector 2: 120 <= H < 240
    mask = (H >= 2*np.pi/3) & (H < 4*np.pi/3)
    H_ = H - 2*np.pi/3
    R[mask] = I[mask] * (1 - S[mask])
    G[mask] = I[mask] * (1 + S[mask]*np.cos(H_[mask]) / np.cos(np.pi/3 - H_[mask]))
    B[mask] = 3*I[mask] - (R[mask] + G[mask])
    
    # Sector 3: 240 <= H < 360
    mask = (H >= 4*np.pi/3)
    H_ = H - 4*np.pi/3
    G[mask] = I[mask] * (1 - S[mask])
    B[mask] = I[mask] * (1 + S[mask]*np.cos(H_[mask]) / np.cos(np.pi/3 - H_[mask]))
    R[mask] = 3*I[mask] - (G[mask] + B[mask])
    
    rgb = np.dstack((R, G, B))
    rgb = np.clip(rgb, 0, 1)
    return (rgb * 255).astype(np.uint8)

H, S, I = rgb2hsi(img)

# Histogram Equalization on Intensity
I_uint8 = (I * 255).astype(np.uint8)
I_eq = cv2.equalizeHist(I_uint8)
I_eq_float = I_eq.astype(np.float32) / 255.0

# Reconstruct
img_eq = hsi2rgb(H, S, I_eq_float)

results = [
    ('Original RGB', img),
    ('Original Intensity', I_uint8),
    ('Equalized Intensity', I_eq),
    ('Enhanced RGB', img_eq)
]

plt.figure(figsize=(10, 10))
for i, (title, result) in enumerate(results):
    plt.subplot(2, 2, i+1)
    if 'RGB' in title:
        plt.imshow(result)
    else:
        plt.imshow(result, cmap='gray')
    plt.title(title)
    plt.axis('off')

# Minimal whitespace layout
plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.02, wspace=0.05, hspace=0.1)
plt.savefig('output.png', dpi=150, bbox_inches='tight')
plt.show()