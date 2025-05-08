"""
Demonstration of Vectorization Improvements in Kohonen SOM implementation.

This script compares:
1. Naive triple-loop implementation (traditional approach)
2. Vectorized implementation (optimized approach)

It measures training time and memory usage for both approaches and generates
comparative visualizations.
"""
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import psutil
from memory_profiler import memory_usage
import sys
from pathlib import Path

# Add the project root to the path so we can import the package
sys.path.append(str(Path(__file__).resolve().parents[2]))

from kohonen import SelfOrganizingMap

# Set random seed for reproducibility
np.random.seed(42)

# Directory for saving comparison results
RESULTS_DIR = Path(__file__).parent
os.makedirs(RESULTS_DIR, exist_ok=True)


class NaiveSOM:
    """
    Naive implementation of Self-Organizing Map with triple nested loops.
    This represents a traditional, non-optimized approach for comparison.
    """
    def __init__(self, width, height, input_dim, learning_rate=0.1, sigma=None):
        self.width = width
        self.height = height
        self.input_dim = input_dim
        self.learning_rate_0 = learning_rate
        self.sigma_0 = sigma if sigma is not None else max(width, height) / 2
        
        # Initialize weights
        self.weights = np.random.random((width, height, input_dim))
    
    def _find_bmu(self, input_vector):
        """Find Best Matching Unit using loops"""
        min_dist = float('inf')
        bmu_x, bmu_y = 0, 0
        
        # Loop through all neurons to find the BMU
        for x in range(self.width):
            for y in range(self.height):
                # Calculate Euclidean distance
                dist = np.sqrt(np.sum((self.weights[x, y] - input_vector) ** 2))
                if dist < min_dist:
                    min_dist = dist
                    bmu_x, bmu_y = x, y
                    
        return bmu_x, bmu_y
    
    def train(self, input_data, n_iterations, verbose=False):
        """Train the SOM using triple nested loops (traditional approach)"""
        n_samples = input_data.shape[0]
        
        # Decay parameter
        λ = n_iterations / np.log(self.sigma_0)
        
        # Training loop - 1st loop: iterations
        for t in range(n_iterations):
            σt = self.sigma_0 * np.exp(-t/λ)
            αt = self.learning_rate_0 * np.exp(-t/λ)
            
            if verbose and t % max(1, n_iterations // 10) == 0:
                print(f"Iteration {t}/{n_iterations}, σ={σt:.4f}, α={αt:.4f}")
            
            # 2nd loop: samples
            for i in range(n_samples):
                vt = input_data[i]
                
                # Find BMU
                bmu_x, bmu_y = self._find_bmu(vt)
                
                # 3rd & 4th loops: update all neurons based on their distance to BMU
                for x in range(self.width):
                    for y in range(self.height):
                        # Calculate squared distance to BMU
                        dist_sq = (x - bmu_x) ** 2 + (y - bmu_y) ** 2
                        
                        # Calculate neighborhood influence
                        influence = np.exp(-dist_sq / (2 * σt ** 2))
                        
                        # Update weights
                        self.weights[x, y] += αt * influence * (vt - self.weights[x, y])
        
        return {"status": "completed"}


def measure_execution_time(func, *args, **kwargs):
    """Measure the execution time of a function"""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return result, end_time - start_time


def measure_peak_memory(func, *args, **kwargs):
    """Measure the peak memory usage of a function"""
    baseline = memory_usage()[0]
    mem_usage = memory_usage((func, args, kwargs), interval=0.1)
    return max(mem_usage) - baseline


def generate_training_data(n_samples=1000, input_dim=3):
    """Generate random training data"""
    return np.random.rand(n_samples, input_dim)


def run_comparison(grid_sizes=[(10, 10), (20, 20), (30, 30)], 
                  n_samples=1000, 
                  input_dim=3,
                  n_iterations=100):
    """
    Run comparison between naive and vectorized SOM implementations
    """
    results = {
        'grid_size': [],
        'naive_time': [],
        'vectorized_time': [],
        'naive_memory': [],
        'vectorized_memory': [],
        'speedup': []
    }
    
    for width, height in grid_sizes:
        print(f"\nComparing grid size: {width}x{height}")
        grid_name = f"{width}x{height}"
        results['grid_size'].append(grid_name)
        
        # Generate data
        data = generate_training_data(n_samples, input_dim)
        
        # Test naive implementation
        naive_som = NaiveSOM(width, height, input_dim)
        
        print("Training naive SOM implementation...")
        _, naive_time = measure_execution_time(naive_som.train, data, n_iterations)
        naive_memory = measure_peak_memory(naive_som.train, data, n_iterations)
        
        results['naive_time'].append(naive_time)
        results['naive_memory'].append(naive_memory)
        
        print(f"Naive SOM: Time={naive_time:.2f}s, Memory={naive_memory:.2f}MB")
        
        # Test vectorized implementation
        vectorized_som = SelfOrganizingMap(width, height, input_dim)
        
        print("Training vectorized SOM implementation...")
        _, vectorized_time = measure_execution_time(vectorized_som.train, data, n_iterations)
        vectorized_memory = measure_peak_memory(vectorized_som.train, data, n_iterations)
        
        results['vectorized_time'].append(vectorized_time)
        results['vectorized_memory'].append(vectorized_memory)
        
        speedup = naive_time / vectorized_time
        results['speedup'].append(speedup)
        
        print(f"Vectorized SOM: Time={vectorized_time:.2f}s, Memory={vectorized_memory:.2f}MB")
        print(f"Speedup: {speedup:.2f}x")
    
    return results


def plot_comparison_results(results):
    """Generate comparison plots"""
    grid_sizes = results['grid_size']
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: Execution Time Comparison
    axes[0].bar(grid_sizes, results['naive_time'], label='Naive Implementation', alpha=0.7, color='#FF9999')
    axes[0].bar(grid_sizes, results['vectorized_time'], label='Vectorized Implementation', alpha=0.7, color='#66B2FF')
    axes[0].set_ylabel('Time (seconds)')
    axes[0].set_xlabel('Grid Size')
    axes[0].set_title('Execution Time Comparison')
    axes[0].legend()
    
    # Plot 2: Memory Usage Comparison
    axes[1].bar(grid_sizes, results['naive_memory'], label='Naive Implementation', alpha=0.7, color='#FF9999')
    axes[1].bar(grid_sizes, results['vectorized_memory'], label='Vectorized Implementation', alpha=0.7, color='#66B2FF')
    axes[1].set_ylabel('Memory (MB)')
    axes[1].set_xlabel('Grid Size')
    axes[1].set_title('Memory Usage Comparison')
    axes[1].legend()
    
    # Plot 3: Speedup
    axes[2].bar(grid_sizes, results['speedup'], color='#99CC99')
    axes[2].axhline(y=1, color='r', linestyle='--')
    axes[2].set_ylabel('Speedup Factor (x times)')
    axes[2].set_xlabel('Grid Size')
    axes[2].set_title('Speedup: Naive vs. Vectorized')
    
    # Add exact speedup values on top of the bars
    for i, v in enumerate(results['speedup']):
        axes[2].text(i, v + 0.1, f"{v:.1f}x", ha='center')
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'vectorization_comparison.png', dpi=300)
    print(f"Saved comparison plot to {RESULTS_DIR / 'vectorization_comparison.png'}")
    plt.close()
    
    # Create a diagram to illustrate the difference between approaches
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    diagram_text = """
    Naive Implementation (Triple Loops)                     Vectorized Implementation
    --------------------------                     --------------------------
    
    for iteration in range(n_iterations):          for iteration in range(n_iterations):
        for sample in data:                           # Calculate BMUs for all samples at once
            # Find BMU                                bmu_indices = find_bmus_batch(data)
            bmu = find_closest_node(sample)           
                                                      # Update all weights using numpy broadcasting
            # Update all nodes                        influence = calculate_influence_matrix()
            for x in range(width):                    weights += learning_rate * influence * (data - weights)
                for y in range(height):
                    # Calculate influence
                    influence = calc_influence()
                    
                    # Update weight
                    weights[x,y] += update
    
    Time Complexity: O(iterations * samples * width * height)    Time Complexity: O(iterations * (samples + width*height))
    """
    
    ax.text(0.1, 0.5, diagram_text, fontsize=12, family='monospace', va='center')
    
    plt.savefig(RESULTS_DIR / 'vectorization_diagram.png', dpi=300, bbox_inches='tight')
    print(f"Saved vectorization diagram to {RESULTS_DIR / 'vectorization_diagram.png'}")
    plt.close()


if __name__ == "__main__":
    print("Running Kohonen SOM Vectorization Comparison")
    
    # Small-scale comparison for quick demonstration
    small_results = run_comparison(
        grid_sizes=[(10, 10), (20, 20), (30, 30)],
        n_samples=1000,
        n_iterations=50
    )
    
    # Plot results
    plot_comparison_results(small_results)
    
    print("\nComparison complete! Check the plots for visualization of performance differences.") 