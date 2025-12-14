import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
img = cv2.imread('../images/elephant.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (256, 256))

def apply_dft_filter(image, use_padding=False):
    rows, cols = image.shape
    
    if use_padding:
        # Pad to optimal size (usually double for convolution to avoid wraparound)
        # Here we pad to 512x512
        m = cv2.getOptimalDFTSize(rows * 2)
        n = cv2.getOptimalDFTSize(cols * 2)
        padded = cv2.copyMakeBorder(image, 0, m - rows, 0, n - cols, cv2.BORDER_CONSTANT, value=0)
    else:
        padded = image
        
    # DFT
    dft = cv2.dft(np.float32(padded), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    
    # Create Low Pass Filter Mask
    p_rows, p_cols = padded.shape
    crow, ccol = p_rows // 2, p_cols // 2
    
    # Create a mask, center square is 1, remaining all zeros
    mask = np.zeros((p_rows, p_cols, 2), np.uint8)
    r = 30 # Radius for LPF
    
    # Circular mask
    y, x = np.ogrid[:p_rows, :p_cols]
    mask_area = (x - ccol)**2 + (y - crow)**2 <= r*r
    mask[mask_area] = 1
    
    # Apply mask
    fshift = dft_shift * mask
    
    # Inverse DFT
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])
    
    # Crop back if padded
    if use_padding:
        img_back = img_back[0:rows, 0:cols]
        
    # Normalize to 0-255
    cv2.normalize(img_back, img_back, 0, 255, cv2.NORM_MINMAX)
    return img_back.astype(np.uint8)

# Apply filters
res_no_pad = apply_dft_filter(img, use_padding=False)
res_pad = apply_dft_filter(img, use_padding=True)

# Visualization
results = [
    ('Original', img),
    ('DFT No Padding', res_no_pad),
    ('DFT With Padding', res_pad)
]

plt.figure(figsize=(12, 4))
for i, (title, result) in enumerate(results):
    plt.subplot(1, 3, i+1)
    plt.imshow(result, cmap='gray')
    plt.title(title)
    plt.axis('off')

# Minimal whitespace layout
plt.subplots_adjust(left=0.02, right=0.98, top=0.85, bottom=0.02, wspace=0.05)
plt.savefig('output.png', dpi=150, bbox_inches='tight')
plt.show()
