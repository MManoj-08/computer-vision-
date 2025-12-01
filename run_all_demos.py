"""
Quick Start Guide - Linear Filters and Convolution Project
Run this script to execute all demonstrations in sequence
"""

import subprocess
import sys
from pathlib import Path

def print_header(title):
    """Print a formatted header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")

def main():
    print_header("LINEAR FILTERS & CONVOLUTION - COMPLETE DEMONSTRATION")
    
    print("This will run all demonstrations in sequence:\n")
    print("1. Main Noise Reduction Demo")
    print("2. Advanced Filter Comparison")
    print("\nPress Enter to continue or Ctrl+C to cancel...")
    
    try:
        input()
    except KeyboardInterrupt:
        print("\n\nCancelled by user.")
        return
    
    # Demo 1: Main demonstration
    print_header("DEMO 1: Main Noise Reduction Demonstration")
    try:
        subprocess.run([sys.executable, "noise_reduction_demo.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running main demo: {e}")
        return
    
    print("\n✓ Main demonstration completed!")
    print("\nPress Enter to continue to advanced comparison...")
    
    try:
        input()
    except KeyboardInterrupt:
        print("\n\nStopped by user.")
        return
    
    # Demo 2: Filter comparison
    print_header("DEMO 2: Advanced Filter Comparison")
    try:
        subprocess.run([sys.executable, "filter_comparison.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running comparison: {e}")
        return
    
    print("\n✓ Advanced comparison completed!")
    
    # Summary
    print_header("ALL DEMONSTRATIONS COMPLETED! 🎉")
    
    output_dir = Path("output")
    if output_dir.exists():
        files = list(output_dir.glob("*.png"))
        print(f"✓ Generated {len(files)} visualization images")
        print(f"✓ All results saved to: {output_dir.absolute()}")
    
    print("\n" + "=" * 70)
    print("NEXT STEPS:")
    print("=" * 70)
    print("""
1. VIEW RESULTS:
   • Check the 'output/' folder for all generated images
   • Open comparison.png for side-by-side results
   
2. INTERACTIVE LEARNING:
   • Run: jupyter notebook analysis_notebook.ipynb
   • Step through explanations and experiments
   
3. EXPERIMENT:
   • Modify parameters in the Python scripts
   • Try different noise levels and filter sizes
   • Compare results with different settings
   
4. READ DOCUMENTATION:
   • Open README.md for comprehensive guide
   • Learn about industry applications
   • Understand trade-offs and best practices
    """)
    
    print("=" * 70)
    print("\n🎓 Happy Learning! Explore computer vision and image processing!")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
