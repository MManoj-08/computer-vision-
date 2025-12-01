"""
Linear Filters and Convolution - Noise Reduction in Camera Images
Industry Case Study: Samsung, Apple (Smartphone Cameras)

This script demonstrates:
1. Simulating noisy images (low-light camera conditions)
2. Applying linear filters (Average, Gaussian)
3. Comparing before and after results
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class NoiseReductionDemo:
    """Demonstrates noise reduction using linear filters"""
    
    def __init__(self):
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)
    
    def create_sample_image(self, width=640, height=480):
        """Load the cv.png image or create a sample if not found"""
        # Try to load cv.png
        image = cv2.imread('cv.png')
        
        if image is None:
            print("⚠ Warning: 'cv.png' not found. Creating sample image...")
            # Create a gradient background with some patterns
            image = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Add gradient background
            for i in range(height):
                image[i, :, 0] = int(255 * i / height)  # Blue channel
                image[i, :, 1] = int(128 * (1 - i / height))  # Green channel
                image[i, :, 2] = int(200 * i / height)  # Red channel
            
            # Add some geometric shapes (simulating scene content)
            cv2.circle(image, (width//4, height//4), 60, (255, 200, 100), -1)
            cv2.rectangle(image, (width//2, height//3), (width//2 + 120, height//3 + 100), (100, 255, 200), -1)
            cv2.circle(image, (3*width//4, 3*height//4), 80, (200, 100, 255), -1)
            
            # Add text (simulating signage or details)
            cv2.putText(image, 'Camera Image', (50, height-30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            print("✓ Loaded 'cv.png' successfully")
        
        return image
    
    def add_noise(self, image, noise_type='gaussian', intensity=50):
        """
        Add noise to simulate low-light camera conditions
        
        Args:
            image: Input image
            noise_type: 'gaussian' or 'salt_pepper'
            intensity: Noise intensity level
        """
        noisy_image = image.copy()
        
        if noise_type == 'gaussian':
            # Gaussian noise - common in low-light conditions
            gaussian_noise = np.random.normal(0, intensity, image.shape)
            noisy_image = np.clip(image.astype(np.float32) + gaussian_noise, 0, 255).astype(np.uint8)
            
        elif noise_type == 'salt_pepper':
            # Salt and pepper noise - dead/hot pixels
            noise_prob = intensity / 1000
            salt = np.random.random(image.shape[:2]) < noise_prob
            pepper = np.random.random(image.shape[:2]) < noise_prob
            
            noisy_image[salt] = 255
            noisy_image[pepper] = 0
        
        return noisy_image
    
    def apply_average_filter(self, image, kernel_size=5):
        """Apply average (box) filter for noise reduction"""
        return cv2.blur(image, (kernel_size, kernel_size))
    
    def apply_gaussian_filter(self, image, kernel_size=5, sigma=0):
        """Apply Gaussian filter for noise reduction"""
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    
    def calculate_metrics(self, original, noisy, filtered):
        """Calculate image quality metrics"""
        # Peak Signal-to-Noise Ratio (PSNR)
        psnr_noisy = cv2.PSNR(original, noisy)
        psnr_filtered = cv2.PSNR(original, filtered)
        
        # Mean Squared Error (MSE)
        mse_noisy = np.mean((original.astype(float) - noisy.astype(float)) ** 2)
        mse_filtered = np.mean((original.astype(float) - filtered.astype(float)) ** 2)
        
        return {
            'psnr_noisy': psnr_noisy,
            'psnr_filtered': psnr_filtered,
            'mse_noisy': mse_noisy,
            'mse_filtered': mse_filtered
        }
    
    def visualize_comparison(self, original, noisy, filtered_avg, filtered_gauss, 
                            filter_name, save_path=None):
        """Create a comparison visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Noise Reduction using Linear Filters - Industry Case Study', 
                     fontsize=16, fontweight='bold')
        
        # Convert BGR to RGB for matplotlib
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        noisy_rgb = cv2.cvtColor(noisy, cv2.COLOR_BGR2RGB)
        filtered_avg_rgb = cv2.cvtColor(filtered_avg, cv2.COLOR_BGR2RGB)
        filtered_gauss_rgb = cv2.cvtColor(filtered_gauss, cv2.COLOR_BGR2RGB)
        
        # Original image
        axes[0, 0].imshow(original_rgb)
        axes[0, 0].set_title('Original Image\n(Clean Camera Capture)', fontsize=12)
        axes[0, 0].axis('off')
        
        # Noisy image
        axes[0, 1].imshow(noisy_rgb)
        axes[0, 1].set_title('Noisy Image\n(Low-Light Conditions)', fontsize=12)
        axes[0, 1].axis('off')
        
        # Average filter result
        axes[1, 0].imshow(filtered_avg_rgb)
        axes[1, 0].set_title(f'Average Filter (7x7)\n{filter_name}', fontsize=12)
        axes[1, 0].axis('off')
        
        # Gaussian filter result
        axes[1, 1].imshow(filtered_gauss_rgb)
        axes[1, 1].set_title(f'Gaussian Filter (7x7, σ=2.0)\n{filter_name}', fontsize=12)
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Comparison saved to: {save_path}")
        
        plt.show()
    
    def demonstrate_convolution(self):
        """Demonstrate how convolution works in filtering"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle('How Convolution Works in Denoising', fontsize=14, fontweight='bold')
        
        # Show different kernels
        kernels = {
            'Average Filter\n(Box Kernel)': np.ones((5, 5)) / 25,
            'Gaussian Filter\n(Gaussian Kernel)': cv2.getGaussianKernel(5, 1) @ cv2.getGaussianKernel(5, 1).T,
            'Identity\n(No Filter)': np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        }
        
        for ax, (title, kernel) in zip(axes, kernels.items()):
            im = ax.imshow(kernel, cmap='hot', interpolation='nearest')
            ax.set_title(title, fontsize=11)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "convolution_kernels.png", dpi=300, bbox_inches='tight')
        print(f"✓ Convolution kernels visualization saved")
        plt.show()
    
    def run_full_demo(self):
        """Run the complete demonstration"""
        print("=" * 70)
        print("LINEAR FILTERS AND CONVOLUTION - NOISE REDUCTION DEMO")
        print("Industry Case Study: Samsung, Apple (Smartphone Cameras)")
        print("=" * 70)
        print()
        
        # Step 1: Create or load sample image
        print("📷 Step 1: Loading camera image...")
        original = self.create_sample_image()
        cv2.imwrite(str(self.output_dir / "01_original.png"), original)
        print("✓ Image loaded and saved")
        print()
        
        # Step 2: Add noise (simulating low-light conditions)
        print("🌙 Step 2: Adding HIGH noise level (simulating very low-light conditions)...")
        noisy = self.add_noise(original, noise_type='gaussian', intensity=50)
        cv2.imwrite(str(self.output_dir / "02_noisy.png"), noisy)
        print("✓ Noisy image generated")
        print()
        
        # Step 3: Apply filters
        print("🔧 Step 3: Applying linear filters...")
        filtered_avg = self.apply_average_filter(noisy, kernel_size=7)
        filtered_gauss = self.apply_gaussian_filter(noisy, kernel_size=7, sigma=2.0)
        
        cv2.imwrite(str(self.output_dir / "03_average_filtered.png"), filtered_avg)
        cv2.imwrite(str(self.output_dir / "04_gaussian_filtered.png"), filtered_gauss)
        print("✓ Average filter applied (7x7 kernel)")
        print("✓ Gaussian filter applied (7x7 kernel, σ=2.0)")
        print()
        
        # Step 4: Calculate metrics
        print("📊 Step 4: Calculating image quality metrics...")
        metrics_avg = self.calculate_metrics(original, noisy, filtered_avg)
        metrics_gauss = self.calculate_metrics(original, noisy, filtered_gauss)
        
        print("\nNoise Reduction Metrics:")
        print("-" * 50)
        print(f"Noisy Image PSNR: {metrics_avg['psnr_noisy']:.2f} dB")
        print(f"Average Filter PSNR: {metrics_avg['psnr_filtered']:.2f} dB")
        print(f"Gaussian Filter PSNR: {metrics_gauss['psnr_filtered']:.2f} dB")
        print(f"\nImprovement (Average): {metrics_avg['psnr_filtered'] - metrics_avg['psnr_noisy']:.2f} dB")
        print(f"Improvement (Gaussian): {metrics_gauss['psnr_filtered'] - metrics_gauss['psnr_noisy']:.2f} dB")
        print()
        
        # Step 5: Visualize results
        print("📈 Step 5: Creating visualizations...")
        self.visualize_comparison(original, noisy, filtered_avg, filtered_gauss,
                                 "Noise Reduction Applied",
                                 self.output_dir / "comparison.png")
        
        # Step 6: Demonstrate convolution
        print("\n🔍 Step 6: Demonstrating convolution kernels...")
        self.demonstrate_convolution()
        
        print("\n" + "=" * 70)
        print("DISCUSSION POINTS:")
        print("=" * 70)
        print("""
1. HOW CONVOLUTION IS USED IN DENOISING:
   • Convolution slides a filter kernel over the image
   • Each pixel is replaced by a weighted average of its neighbors
   • Average filter: Equal weights for all neighbors
   • Gaussian filter: Higher weight to closer pixels (bell curve)

2. TRADE-OFFS: SMOOTHING VS. DETAIL PRESERVATION:
   • Average Filter: More aggressive smoothing, can blur edges
   • Gaussian Filter: Better preserves edges and fine details
   • Larger kernel size: More noise reduction, more blur
   • Higher PSNR indicates better quality

3. REAL-WORLD RELEVANCE:
   • Smartphone Cameras (Samsung, Apple):
     - Night mode uses sophisticated filtering
     - Multi-frame noise reduction
     - Computational photography pipeline
   
   • CCTV Footage Enhancement:
     - Low-light surveillance
     - Real-time processing requirements
     - Balance between quality and speed
   
   • Professional Applications:
     - Medical imaging (MRI, X-ray denoising)
     - Astronomical photography
     - Security and forensics
        """)
        
        print("\n✅ All results saved to:", self.output_dir.absolute())
        print("=" * 70)


if __name__ == "__main__":
    demo = NoiseReductionDemo()
    demo.run_full_demo()
