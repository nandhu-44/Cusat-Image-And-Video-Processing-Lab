import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
img = cv2.imread('../images/elephant.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (256, 256))

# RGB to HSI Conversion
def rgb2hsi(rgb_img):
    # Normalize to [0, 1]
    rgb = rgb_img.astype(np.float32) / 255.0
    R = rgb[:,:,0]
    G = rgb[:,:,1]
    B = rgb[:,:,2]
    
    # Intensity
    I = (R + G + B) / 3.0
    
    # Saturation
    # S = 1 - 3/(R+G+B) * min(R,G,B)
    # S = 1 - min(R,G,B)/I
    min_rgb = np.minimum(np.minimum(R, G), B)
    S = 1 - (min_rgb / (I + 1e-6))
    S[I == 0] = 0
    
    # Hue
    # theta = arccos( 0.5*((R-G)+(R-B)) / sqrt((R-G)^2 + (R-B)(G-B)) )
    num = 0.5 * ((R - G) + (R - B))
    den = np.sqrt((R - G)**2 + (R - B) * (G - B))
    theta = np.arccos(num / (den + 1e-6))
    
    H = theta.copy()
    H[B > G] = 2*np.pi - H[B > G]
    
    # Normalize H to [0, 1] for display or [0, 255]
    H = H / (2*np.pi)
    
    return H, S, I

H, S, I = rgb2hsi(img)

# Convert to uint8 for display
H_disp = (H * 255).astype(np.uint8)
S_disp = (S * 255).astype(np.uint8)
I_disp = (I * 255).astype(np.uint8)

results = [
    ('Original RGB', img),
    ('Hue', H_disp),
    ('Saturation', S_disp),
    ('Intensity', I_disp)
]

plt.figure(figsize=(10, 10))
for i, (title, result) in enumerate(results):
    plt.subplot(2, 2, i+1)
    if title == 'Original RGB':
        plt.imshow(result)
    else:
        plt.imshow(result, cmap='gray')
    plt.title(title)
    plt.axis('off')

# Minimal whitespace layout
plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.02, wspace=0.05, hspace=0.1)
plt.savefig('output.png', dpi=150, bbox_inches='tight')
plt.show()