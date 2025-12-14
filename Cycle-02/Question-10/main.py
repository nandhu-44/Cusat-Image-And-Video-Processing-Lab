import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
img = cv2.imread('../images/elephant.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (256, 256))

# DFT
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

rows, cols = img.shape
crow, ccol = rows // 2, cols // 2

def apply_filter(fshift, mask):
    fshift_filtered = fshift * mask
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])
    cv2.normalize(img_back, img_back, 0, 255, cv2.NORM_MINMAX)
    return img_back.astype(np.uint8)

# 1. Ideal Low Pass Filter
def ideal_lpf(d0):
    mask = np.zeros((rows, cols, 2), np.float32)
    y, x = np.ogrid[:rows, :cols]
    mask_area = (x - ccol)**2 + (y - crow)**2 <= d0**2
    mask[mask_area] = 1
    return mask

# 2. Butterworth Low Pass Filter
def butterworth_lpf(d0, n=2):
    y, x = np.ogrid[:rows, :cols]
    dist = np.sqrt((x - ccol)**2 + (y - crow)**2)
    mask = 1 / (1 + (dist / d0)**(2*n))
    return np.dstack((mask, mask))

# 3. Gaussian Low Pass Filter
def gaussian_lpf(d0):
    y, x = np.ogrid[:rows, :cols]
    dist = np.sqrt((x - ccol)**2 + (y - crow)**2)
    mask = np.exp(-(dist**2) / (2 * (d0**2)))
    return np.dstack((mask, mask))

D0 = 50 # Cutoff frequency

mask_ideal = ideal_lpf(D0)
mask_butter = butterworth_lpf(D0, n=2)
mask_gauss = gaussian_lpf(D0)

res_ideal = apply_filter(dft_shift, mask_ideal)
res_butter = apply_filter(dft_shift, mask_butter)
res_gauss = apply_filter(dft_shift, mask_gauss)

results = [
    ('Original', img),
    ('Ideal LPF', res_ideal),
    ('Butterworth LPF', res_butter),
    ('Gaussian LPF', res_gauss)
]

plt.figure(figsize=(10, 10))
for i, (title, result) in enumerate(results):
    plt.subplot(2, 2, i+1)
    plt.imshow(result, cmap='gray')
    plt.title(title)
    plt.axis('off')

# Minimal whitespace layout
plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.02, wspace=0.05, hspace=0.1)
plt.savefig('output.png', dpi=150, bbox_inches='tight')
plt.show()
