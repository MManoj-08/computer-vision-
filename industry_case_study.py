"""
Industry Case Study: Noise Reduction in Camera Images
Company Example: Samsung, Apple (Smartphone Cameras)

This demo covers:
1. Noisy images from low-light smartphone photos (using cv.png)
2. Linear filters (Average, Gaussian) application
3. Before/After comparison
4. Discussion points on convolution, trade-offs, and real-world relevance
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Create output directory
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

print("="*70)
print("INDUSTRY CASE STUDY: NOISE REDUCTION IN CAMERA IMAGES")
print("Company Examples: Samsung Galaxy & Apple iPhone")
print("="*70)

# ============================================================================
# PART 1: SHOW NOISY IMAGES (Low-Light Smartphone Photo Simulation)
# ============================================================================
print("\n[PART 1] Simulating Low-Light Smartphone Photo")
print("-"*70)

# Load the real camera image
original = cv2.imread('camera.jpg')
if original is None:
    print("ERROR: camera.jpg not found! Please ensure the image exists.")
    exit()

print(f"✓ Loaded camera.jpg ({original.shape[1]}x{original.shape[0]} pixels)")

# Simulate low-light noise (what happens in Samsung/Apple cameras at high ISO)
print("✓ Adding Gaussian noise to simulate low-light conditions...")
noise_level = 50  # High ISO noise
noise = np.random.normal(0, noise_level, original.shape)
noisy_image = np.clip(original.astype(np.float32) + noise, 0, 255).astype(np.uint8)

print(f"  - Noise level: {noise_level} (simulating ISO 3200-6400)")
print(f"  - This mimics: Indoor/night photography without flash")

# ============================================================================
# PART 2: APPLY LINEAR FILTERS
# ============================================================================
print("\n[PART 2] Applying Linear Filters (Average & Gaussian)")
print("-"*70)

# Average Filter (Box Filter)
kernel_size = 7
print(f"✓ Applying Average Filter ({kernel_size}x{kernel_size})...")
average_filtered = cv2.blur(noisy_image, (kernel_size, kernel_size))
print(f"  - Simple arithmetic mean of {kernel_size}x{kernel_size} neighbors")

# Gaussian Filter
sigma = 2.0
print(f"✓ Applying Gaussian Filter ({kernel_size}x{kernel_size}, σ={sigma})...")
gaussian_filtered = cv2.GaussianBlur(noisy_image, (kernel_size, kernel_size), sigma)
print(f"  - Weighted average with Gaussian distribution")
print(f"  - Higher weights for closer pixels (bell curve)")

# ============================================================================
# PART 3: COMPARE BEFORE AND AFTER IMAGES
# ============================================================================
print("\n[PART 3] Comparison & Quality Metrics")
print("-"*70)

# Calculate PSNR (Peak Signal-to-Noise Ratio)
psnr_noisy = cv2.PSNR(original, noisy_image)
psnr_average = cv2.PSNR(original, average_filtered)
psnr_gaussian = cv2.PSNR(original, gaussian_filtered)

print(f"PSNR Metrics (Higher = Better Quality):")
print(f"  Noisy Image:      {psnr_noisy:.2f} dB")
print(f"  Average Filter:   {psnr_average:.2f} dB  (improvement: +{psnr_average-psnr_noisy:.2f} dB)")
print(f"  Gaussian Filter:  {psnr_gaussian:.2f} dB  (improvement: +{psnr_gaussian-psnr_noisy:.2f} dB)")

# Save images
cv2.imwrite(str(output_dir / "1_original_clean.png"), original)
cv2.imwrite(str(output_dir / "2_noisy_lowlight.png"), noisy_image)
cv2.imwrite(str(output_dir / "3_average_filtered.png"), average_filtered)
cv2.imwrite(str(output_dir / "4_gaussian_filtered.png"), gaussian_filtered)

# Create visual comparison
fig = plt.figure(figsize=(16, 10))

# Original
plt.subplot(2, 3, 1)
plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
plt.title('Original Image\n(Clean Reference)', fontsize=12, fontweight='bold')
plt.axis('off')

# Noisy
plt.subplot(2, 3, 2)
plt.imshow(cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB))
plt.title(f'Low-Light Smartphone Photo\n(Noisy: {psnr_noisy:.2f} dB)', 
          fontsize=12, fontweight='bold', color='red')
plt.axis('off')

# Comparison
plt.subplot(2, 3, 3)
plt.text(0.5, 0.7, 'Noise Reduction\nNeeded!', 
         ha='center', va='center', fontsize=20, fontweight='bold',
         transform=plt.gca().transAxes)
plt.text(0.5, 0.3, f'Quality Drop:\n{psnr_noisy:.1f} dB', 
         ha='center', va='center', fontsize=14,
         transform=plt.gca().transAxes, color='red')
plt.axis('off')

# Average filtered
plt.subplot(2, 3, 4)
plt.imshow(cv2.cvtColor(average_filtered, cv2.COLOR_BGR2RGB))
plt.title(f'After Average Filter\n({psnr_average:.2f} dB, ↑{psnr_average-psnr_noisy:.2f} dB)', 
          fontsize=12, fontweight='bold', color='blue')
plt.axis('off')

# Gaussian filtered
plt.subplot(2, 3, 5)
plt.imshow(cv2.cvtColor(gaussian_filtered, cv2.COLOR_BGR2RGB))
plt.title(f'After Gaussian Filter\n({psnr_gaussian:.2f} dB, ↑{psnr_gaussian-psnr_noisy:.2f} dB)', 
          fontsize=12, fontweight='bold', color='green')
plt.axis('off')

# Best result indicator
plt.subplot(2, 3, 6)
best_filter = "Gaussian" if psnr_gaussian > psnr_average else "Average"
improvement = max(psnr_gaussian, psnr_average) - psnr_noisy
plt.text(0.5, 0.7, f'✓ Best Result:\n{best_filter} Filter', 
         ha='center', va='center', fontsize=18, fontweight='bold',
         transform=plt.gca().transAxes, color='green')
plt.text(0.5, 0.3, f'Improvement:\n+{improvement:.2f} dB', 
         ha='center', va='center', fontsize=14,
         transform=plt.gca().transAxes)
plt.axis('off')

plt.suptitle('Industry Case Study: Noise Reduction in Camera Images\nSamsung Galaxy & Apple iPhone Technology', 
             fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig(output_dir / "comparison_results.png", dpi=300, bbox_inches='tight')
print(f"\n✓ Comparison saved: {output_dir / 'comparison_results.png'}")
plt.show()

# ============================================================================
# DISCUSSION POINTS
# ============================================================================
print("\n" + "="*70)
print("DISCUSSION POINTS")
print("="*70)

print("\n1️⃣  HOW CONVOLUTION IS USED IN DENOISING")
print("-"*70)
print("""
Convolution Process:
• A filter kernel (matrix) slides over the image
• At each position, it multiplies pixel values with kernel weights
• Sums all products to create the output pixel
• This replaces each noisy pixel with a weighted average of neighbors

Mathematical Formula:
    Output(x,y) = Σ Σ Kernel(i,j) × Input(x+i, y+j)

Why It Works:
• Noise is random → averaging reduces random variations
• Signal is consistent → averaging preserves it
• Larger kernel = more averaging = more denoising (but more blur)

Example:
• Average Filter: All weights equal (1/49 for 7×7 kernel)
• Gaussian Filter: Center weight highest, decreases with distance
""")

print("\n2️⃣  TRADE-OFFS: SMOOTHING vs. DETAIL PRESERVATION")
print("-"*70)
print(f"""
In This Demo:
• Noisy Image:      {psnr_noisy:.2f} dB  ← Lots of detail BUT noisy
• Average Filter:   {psnr_average:.2f} dB  ← Cleaner BUT some blur
• Gaussian Filter:  {psnr_gaussian:.2f} dB  ← Best balance

Key Trade-offs:
┌─────────────────────┬──────────────┬─────────────────────┐
│ Filter Type         │ Noise Reduce │ Detail Preservation │
├─────────────────────┼──────────────┼─────────────────────┤
│ No Filter           │ None         │ Perfect             │
│ Small Kernel (3×3)  │ Low          │ Good                │
│ Medium Kernel (5×5) │ Moderate     │ Moderate            │
│ Large Kernel (7×7)  │ High         │ Poor (blurry)       │
│ Very Large (11×11)  │ Very High    │ Very Poor           │
└─────────────────────┴──────────────┴─────────────────────┘

Gaussian vs Average:
• Gaussian: Better edge preservation (weighted by distance)
• Average: More uniform smoothing (equal weights)
• Gaussian is preferred in smartphone cameras

Real-World Challenge:
• Too little smoothing → Noisy, grainy photos
• Too much smoothing → Blurry, loss of fine details
• Goal: Find optimal balance for each scene
""")

print("\n3️⃣  REAL-WORLD RELEVANCE")
print("-"*70)
print("""
📱 SMARTPHONE CAMERAS (Samsung Galaxy, Apple iPhone)

Night Mode / Low-Light Photography:
• Problem: High ISO causes severe noise in dark conditions
• Solution: Multi-frame + advanced filtering
  1. Capture 6-9 frames over 1-3 seconds
  2. Align frames (compensate hand shake with OIS)
  3. Temporal filtering (average across frames)
  4. Spatial filtering (Gaussian/bilateral filters)
  5. AI enhancement (deep learning denoising)

Samsung's Approach:
• Uses adaptive filtering based on scene brightness
• Combines multiple algorithms in ISP (Image Signal Processor)
• Real-time processing during capture

Apple's Approach:
• Deep Fusion: Pixel-by-pixel analysis
• Smart HDR: Combines multiple exposures
• Computational photography pipeline

🎥 CCTV FOOTAGE ENHANCEMENT

Challenges:
• 24/7 operation in varying light conditions
• Limited camera sensors (cost constraints)
• Real-time processing requirements
• Storage and bandwidth limitations

Applications:
• Low-light surveillance (parking lots, streets)
• Forensic analysis (license plate reading)
• Facial recognition in poor lighting
• Event monitoring and security

Filtering Strategy:
• Fast filters (Average/Gaussian) for real-time
• Adaptive strength based on scene analysis
• Hardware acceleration (GPU/FPGA)
• Balance: Quality vs Processing speed vs Storage

Other Applications:
• Medical Imaging: X-ray, MRI denoising
• Astronomy: Deep space photography
• Security: Fingerprint enhancement
• Industrial: Quality control inspection
• Autonomous Vehicles: Sensor data cleaning
""")

print("\n" + "="*70)
print("KEY TAKEAWAYS")
print("="*70)
print(f"""
✓ Linear filters are fundamental in image processing
✓ Convolution averages neighbors to reduce random noise
✓ Trade-off exists: More smoothing = Less noise BUT more blur
✓ Gaussian filter generally better than average filter
✓ Real smartphone cameras use multi-frame + advanced techniques
✓ Different applications need different filtering strategies

Performance in this demo:
• Gaussian Filter improved quality by {psnr_gaussian-psnr_noisy:.2f} dB
• This represents significant visible improvement
• Modern phones combine this with AI for even better results
""")

print("\n✅ DEMO COMPLETE!")
print(f"📁 All results saved in: {output_dir.absolute()}")
print("="*70)
