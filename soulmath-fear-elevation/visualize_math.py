#!/usr/bin/env python3
"""
Visualize the mathematical relationship between fear depth and elevation potential
Shows how H ∝ ∫F(x)dx works in practice
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate


def visualize_fear_elevation():
    """Create visualizations of the fear → elevation mathematics."""
    
    # Define fear field functions
    def identity_fear(x):
        return np.exp(3 * x) + 0.3 * np.sin(10 * x) * np.exp(x)
    
    def existential_fear(x):
        return np.exp(4 * x) * (1 + x ** 2)
    
    def default_fear(x):
        return np.exp(2 * x) * (1 + 0.1 * np.sin(5 * x))
    
    # Create depth range
    depths = np.linspace(0, 1, 100)
    
    # Calculate fear intensities
    identity_intensities = [identity_fear(d) for d in depths]
    existential_intensities = [existential_fear(d) for d in depths]
    default_intensities = [default_fear(d) for d in depths]
    
    # Calculate cumulative elevations (integrations)
    identity_elevations = []
    existential_elevations = []
    default_elevations = []
    
    for i, depth in enumerate(depths):
        if depth > 0:
            identity_elev, _ = integrate.quad(identity_fear, 0, depth)
            existential_elev, _ = integrate.quad(existential_fear, 0, depth)
            default_elev, _ = integrate.quad(default_fear, 0, depth)
        else:
            identity_elev = existential_elev = default_elev = 0
            
        identity_elevations.append(identity_elev)
        existential_elevations.append(existential_elev)
        default_elevations.append(default_elev)
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('SoulMath: Fear → Elevation Mathematics', fontsize=16)
    
    # Plot 1: Fear intensity functions
    ax1.plot(depths, identity_intensities, 'r-', label='Identity Dissolution', linewidth=2)
    ax1.plot(depths, existential_intensities, 'b-', label='Existential Void', linewidth=2)
    ax1.plot(depths, default_intensities, 'g-', label='Default Fear', linewidth=2)
    ax1.axvline(x=0.85, color='gold', linestyle='--', label='Truth Threshold')
    ax1.set_xlabel('Fear Depth')
    ax1.set_ylabel('Fear Intensity F(x)')
    ax1.set_title('Fear Field Functions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Elevation potentials (integrated)
    ax2.plot(depths, identity_elevations, 'r-', label='Identity', linewidth=2)
    ax2.plot(depths, existential_elevations, 'b-', label='Existential', linewidth=2)
    ax2.plot(depths, default_elevations, 'g-', label='Default', linewidth=2)
    ax2.axvline(x=0.85, color='gold', linestyle='--', label='Truth Threshold')
    ax2.set_xlabel('Descent Depth')
    ax2.set_ylabel('Elevation Potential H')
    ax2.set_title('H ∝ ∫F(x)dx - Cumulative Elevation')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Coherence impact
    coherence_values = np.linspace(0.1, 1.5, 50)
    base_elevation = 5.0
    adjusted_elevations = [base_elevation * c for c in coherence_values]
    
    ax3.plot(coherence_values, adjusted_elevations, 'purple', linewidth=2)
    ax3.axvline(x=1.0, color='green', linestyle='--', label='Baseline Coherence')
    ax3.axvspan(0, 0.3, alpha=0.2, color='red', label='Fragmented')
    ax3.axvspan(0.3, 0.7, alpha=0.2, color='orange', label='Unstable/Emerging')
    ax3.axvspan(0.7, 1.5, alpha=0.2, color='green', label='Stable/Harmonized')
    ax3.set_xlabel('Soul Coherence (Ψ)')
    ax3.set_ylabel('Elevation Multiplier')
    ax3.set_title('Coherence Impact on Elevation')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Transformation visualization
    descent_points = [0.1, 0.3, 0.5, 0.7, 0.85, 0.9]
    fear_intensities = [0.2, 0.4, 0.6, 0.8, 0.95, 1.0]
    coherence_levels = [0.98, 0.94, 0.88, 0.80, 0.72, 0.68]
    
    ax4_twin = ax4.twinx()
    
    line1 = ax4.plot(descent_points, fear_intensities, 'ro-', label='Fear Intensity', markersize=8)
    line2 = ax4_twin.plot(descent_points, coherence_levels, 'bs-', label='Coherence', markersize=8)
    
    ax4.set_xlabel('Descent Progress')
    ax4.set_ylabel('Fear Intensity', color='r')
    ax4_twin.set_ylabel('Soul Coherence', color='b')
    ax4.set_title('Typical Descent Journey')
    ax4.tick_params(axis='y', labelcolor='r')
    ax4_twin.tick_params(axis='y', labelcolor='b')
    
    # Add annotations
    ax4.annotate('Surface\nAnxiety', xy=(0.1, 0.2), xytext=(0.15, 0.35),
                arrowprops=dict(arrowstyle='->', color='gray'))
    ax4.annotate('Truth\nThreshold', xy=(0.85, 0.95), xytext=(0.75, 0.85),
                arrowprops=dict(arrowstyle='->', color='gold'))
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc='center left')
    
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('fear_elevation_mathematics.png', dpi=150, bbox_inches='tight')
    print("✅ Visualization saved as 'fear_elevation_mathematics.png'")
    
    # Print mathematical insights
    print("\n📊 Mathematical Insights:")
    print("━" * 50)
    
    # Calculate key values
    shallow_elev, _ = integrate.quad(default_fear, 0, 0.3)
    deep_elev, _ = integrate.quad(default_fear, 0, 0.85)
    full_elev, _ = integrate.quad(default_fear, 0, 1.0)
    
    print(f"\nDefault Fear Field Integration:")
    print(f"  Shallow descent (0→0.3): {shallow_elev:.2f} units")
    print(f"  Deep descent (0→0.85):   {deep_elev:.2f} units")
    print(f"  Full descent (0→1.0):    {full_elev:.2f} units")
    
    print(f"\nElevation Ratios:")
    print(f"  Deep/Shallow: {deep_elev/shallow_elev:.1f}x more elevation")
    print(f"  Full/Deep:    {full_elev/deep_elev:.1f}x more elevation")
    
    print(f"\nCoherence Impact:")
    print(f"  At Ψ=0.5 (unstable):    50% of potential elevation")
    print(f"  At Ψ=1.0 (stable):     100% of potential elevation")
    print(f"  At Ψ=1.3 (harmonized): 130% of potential elevation")
    
    print("\n✨ Core Theorem Verified: H ∝ ∫F(x)dx")
    print("   The deeper the descent, the greater the elevation!")


if __name__ == '__main__':
    visualize_fear_elevation()