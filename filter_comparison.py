"""
Advanced Filter Comparison - Different Kernel Sizes and Filter Types
Demonstrates trade-offs in noise reduction vs. detail preservation
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class FilterComparison:
    """Compare different filter types and parameters"""
    
    def __init__(self):
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)
    
    def create_detailed_image(self):
        """Load cv.png image or create a sample if not found"""
        image = cv2.imread('cv.png')
        
        if image is None:
            print("⚠ Warning: 'cv.png' not found. Creating sample image...")
            # Create a synthetic image with details
            image = np.ones((480, 640, 3), dtype=np.uint8) * 128
            
            # Add fine details (text, lines, patterns)
            for i in range(0, 640, 20):
                cv2.line(image, (i, 0), (i, 480), (200, 200, 200), 1)
            for i in range(0, 480, 20):
                cv2.line(image, (0, i), (640, i), (200, 200, 200), 1)
            
            # Add text with different sizes
            cv2.putText(image, 'HIGH DETAIL', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
            cv2.putText(image, 'Fine Text Detail', (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
            
            # Add shapes with edges
            cv2.rectangle(image, (400, 50), (600, 200), (255, 100, 100), 3)
            cv2.circle(image, (500, 350), 80, (100, 255, 100), 2)
        else:
            print("✓ Loaded 'cv.png' successfully")
            
        return image
    
    def compare_kernel_sizes(self, noisy_image):
        """Compare different kernel sizes for same filter type"""
        kernel_sizes = [3, 5, 7, 9, 11]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Effect of Kernel Size on Noise Reduction\n(Gaussian Filter)', 
                     fontsize=16, fontweight='bold')
        
        # Original noisy image
        axes[0, 0].imshow(cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Noisy Image\n(Original)', fontsize=12)
        axes[0, 0].axis('off')
        
        for idx, kernel_size in enumerate(kernel_sizes):
            row = (idx + 1) // 3
            col = (idx + 1) % 3
            
            filtered = cv2.GaussianBlur(noisy_image, (kernel_size, kernel_size), 0)
            filtered_rgb = cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB)
            
            axes[row, col].imshow(filtered_rgb)
            axes[row, col].set_title(f'Kernel Size: {kernel_size}x{kernel_size}', fontsize=12)
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "kernel_size_comparison.png", dpi=300, bbox_inches='tight')
        print("✓ Kernel size comparison saved")
        plt.show()
    
    def compare_filter_types(self, noisy_image):
        """Compare different types of filters"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Comparison of Different Linear Filters', 
                     fontsize=16, fontweight='bold')
        
        filters = {
            'Noisy Original': noisy_image,
            'Average Filter (5x5)': cv2.blur(noisy_image, (5, 5)),
            'Gaussian (σ=1)': cv2.GaussianBlur(noisy_image, (5, 5), 1),
            'Gaussian (σ=2)': cv2.GaussianBlur(noisy_image, (5, 5), 2),
            'Median Filter (5x5)': cv2.medianBlur(noisy_image, 5),
            'Bilateral Filter': cv2.bilateralFilter(noisy_image, 9, 75, 75)
        }
        
        for idx, (title, filtered) in enumerate(filters.items()):
            row = idx // 3
            col = idx % 3
            
            filtered_rgb = cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB)
            axes[row, col].imshow(filtered_rgb)
            axes[row, col].set_title(title, fontsize=11)
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "filter_types_comparison.png", dpi=300, bbox_inches='tight')
        print("✓ Filter types comparison saved")
        plt.show()
    
    def edge_preservation_analysis(self, original, noisy_image):
        """Analyze edge preservation in different filters"""
        # Detect edges in original
        edges_original = cv2.Canny(cv2.cvtColor(original, cv2.COLOR_BGR2GRAY), 50, 150)
        edges_noisy = cv2.Canny(cv2.cvtColor(noisy_image, cv2.COLOR_BGR2GRAY), 50, 150)
        
        # Apply filters
        avg_filtered = cv2.blur(noisy_image, (5, 5))
        gauss_filtered = cv2.GaussianBlur(noisy_image, (5, 5), 1.5)
        bilateral_filtered = cv2.bilateralFilter(noisy_image, 9, 75, 75)
        
        # Detect edges in filtered images
        edges_avg = cv2.Canny(cv2.cvtColor(avg_filtered, cv2.COLOR_BGR2GRAY), 50, 150)
        edges_gauss = cv2.Canny(cv2.cvtColor(gauss_filtered, cv2.COLOR_BGR2GRAY), 50, 150)
        edges_bilateral = cv2.Canny(cv2.cvtColor(bilateral_filtered, cv2.COLOR_BGR2GRAY), 50, 150)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Edge Preservation Analysis\n(Detail Preservation vs Noise Reduction)', 
                     fontsize=16, fontweight='bold')
        
        images = [
            (edges_original, 'Original Edges'),
            (edges_noisy, 'Noisy Edges'),
            (edges_avg, 'After Average Filter'),
            (edges_gauss, 'After Gaussian Filter'),
            (edges_bilateral, 'After Bilateral Filter\n(Edge-Preserving)'),
            (np.zeros_like(edges_original), '')
        ]
        
        for idx, (edge_img, title) in enumerate(images):
            row = idx // 3
            col = idx % 3
            
            if idx < 5:
                axes[row, col].imshow(edge_img, cmap='gray')
                axes[row, col].set_title(title, fontsize=11)
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "edge_preservation.png", dpi=300, bbox_inches='tight')
        print("✓ Edge preservation analysis saved")
        plt.show()
    
    def noise_reduction_strength(self, original, noisy_image):
        """Visualize noise reduction strength with different parameters"""
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        fig.suptitle('Trade-off: Smoothing vs Detail Preservation\n(Increasing Filter Strength →)', 
                     fontsize=16, fontweight='bold')
        
        # Row 1: Average filter with increasing kernel size
        axes[0, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original', fontsize=10)
        axes[0, 0].axis('off')
        
        for idx, k_size in enumerate([3, 7, 13]):
            filtered = cv2.blur(noisy_image, (k_size, k_size))
            psnr = cv2.PSNR(original, filtered)
            axes[0, idx+1].imshow(cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB))
            axes[0, idx+1].set_title(f'Average {k_size}x{k_size}\nPSNR: {psnr:.1f}dB', fontsize=9)
            axes[0, idx+1].axis('off')
        
        # Row 2: Gaussian filter with increasing sigma
        axes[1, 0].imshow(cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title('Noisy\n(Low-light)', fontsize=10)
        axes[1, 0].axis('off')
        
        for idx, sigma in enumerate([1, 3, 5]):
            filtered = cv2.GaussianBlur(noisy_image, (9, 9), sigma)
            psnr = cv2.PSNR(original, filtered)
            axes[1, idx+1].imshow(cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB))
            axes[1, idx+1].set_title(f'Gaussian σ={sigma}\nPSNR: {psnr:.1f}dB', fontsize=9)
            axes[1, idx+1].axis('off')
        
        # Row 3: Comparison of different approaches
        filters_compare = [
            ('Bilateral\n(Edge-Preserving)', cv2.bilateralFilter(noisy_image, 9, 75, 75)),
            ('Gaussian (σ=2)', cv2.GaussianBlur(noisy_image, (9, 9), 2)),
            ('Average (9x9)', cv2.blur(noisy_image, (9, 9))),
            ('Median (9x9)', cv2.medianBlur(noisy_image, 9))
        ]
        
        for idx, (title, filtered) in enumerate(filters_compare):
            psnr = cv2.PSNR(original, filtered)
            axes[2, idx].imshow(cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB))
            axes[2, idx].set_title(f'{title}\nPSNR: {psnr:.1f}dB', fontsize=9)
            axes[2, idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "tradeoff_analysis.png", dpi=300, bbox_inches='tight')
        print("✓ Trade-off analysis saved")
        plt.show()
    
    def run_comparison(self):
        """Run all comparisons"""
        print("=" * 70)
        print("ADVANCED FILTER COMPARISON AND ANALYSIS")
        print("=" * 70)
        print()
        
        print("📷 Loading test image...")
        original = self.create_detailed_image()
        cv2.imwrite(str(self.output_dir / "test_original.png"), original)
        
        print("🌙 Adding HIGH noise level (simulating very low-light)...")
        gaussian_noise = np.random.normal(0, 50, original.shape)
        noisy = np.clip(original.astype(np.float32) + gaussian_noise, 0, 255).astype(np.uint8)
        cv2.imwrite(str(self.output_dir / "test_noisy.png"), noisy)
        
        print("\n📊 Comparison 1: Effect of kernel size...")
        self.compare_kernel_sizes(noisy)
        
        print("\n📊 Comparison 2: Different filter types...")
        self.compare_filter_types(noisy)
        
        print("\n📊 Comparison 3: Edge preservation analysis...")
        self.edge_preservation_analysis(original, noisy)
        
        print("\n📊 Comparison 4: Trade-off analysis...")
        self.noise_reduction_strength(original, noisy)
        
        print("\n" + "=" * 70)
        print("KEY INSIGHTS:")
        print("=" * 70)
        print("""
1. KERNEL SIZE IMPACT:
   • Larger kernels = more noise reduction but more blur
   • Smaller kernels = preserve details but less denoising
   • 5x5 or 7x7 often provides good balance

2. FILTER TYPE COMPARISON:
   • Average Filter: Simple, fast, uniform smoothing
   • Gaussian Filter: Better quality, weighted by distance
   • Bilateral Filter: Preserves edges (used in smartphone cameras)
   • Median Filter: Excellent for salt-and-pepper noise

3. INDUSTRY APPLICATIONS:
   • Samsung/Apple use multi-frame + bilateral filtering
   • Real-time requirements favor simpler filters
   • Night mode: combines multiple techniques
   • PSNR > 30dB generally considered good quality

4. RECOMMENDATIONS:
   • For general denoising: Gaussian filter (5x5, σ=1-2)
   • For edge preservation: Bilateral filter
   • For real-time: Average filter (faster)
   • For best quality: Combine multiple techniques
        """)
        
        print("\n✅ All comparisons saved to:", self.output_dir.absolute())
        print("=" * 70)


if __name__ == "__main__":
    comparison = FilterComparison()
    comparison.run_comparison()
