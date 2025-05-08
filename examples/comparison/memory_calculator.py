"""
Memory Usage Calculator for Kohonen SOM

This script calculates the theoretical memory usage for different SOM configurations
based on the actual implementation code.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add the project root to the path so we can import the package
sys.path.append(str(Path(__file__).resolve().parents[2]))

# Directory for saving results
RESULTS_DIR = Path(__file__).parent
os.makedirs(RESULTS_DIR, exist_ok=True)

def calculate_memory_usage(width, height, input_dim, n_samples, batch_size=None):
    """
    Calculate the theoretical memory usage for a SOM with the given parameters.
    
    Parameters:
    -----------
    width : int
        Width of the SOM grid
    height : int
        Height of the SOM grid
    input_dim : int
        Dimensionality of the input vectors
    n_samples : int
        Number of input samples
    batch_size : int, optional
        Batch size for processing. If None, uses fully vectorized approach.
    
    Returns:
    --------
    dict
        Dictionary containing memory usage information in MB
    """
    # Size of a float64 in bytes
    float_size = 8
    
    # Calculate memory usage for core structures
    memory_usage = {}
    
    # Weights matrix: width × height × input_dim
    weights_memory = width * height * input_dim * float_size
    memory_usage['weights'] = weights_memory / (1024 * 1024)  # Convert to MB
    
    # Grid positions: width × height × 2
    grid_memory = width * height * 2 * float_size
    memory_usage['grid'] = grid_memory / (1024 * 1024)  # Convert to MB
    
    # Process actual batch size
    actual_batch = n_samples if batch_size is None else min(batch_size, n_samples)
    
    # Memory for distance calculations in _find_bmus_batch
    # squared_diffs: actual_batch × (width×height) × input_dim
    squared_diffs_memory = actual_batch * width * height * input_dim * float_size
    memory_usage['squared_diffs'] = squared_diffs_memory / (1024 * 1024)  # Convert to MB
    
    # Memory for distances: actual_batch × (width×height)
    distances_memory = actual_batch * width * height * float_size
    memory_usage['distances'] = distances_memory / (1024 * 1024)  # Convert to MB
    
    # Memory for weight updates (influence matrix): width × height
    influence_memory = width * height * float_size
    memory_usage['influence'] = influence_memory / (1024 * 1024)  # Convert to MB
    
    # Total memory estimation (add 20% overhead for Python objects and other arrays)
    total_memory = weights_memory + grid_memory + squared_diffs_memory + distances_memory + influence_memory
    memory_usage['total'] = (total_memory * 1.2) / (1024 * 1024)  # Convert to MB with overhead
    
    return memory_usage

def compare_vectorization_memory(grid_sizes=[(10, 10), (30, 30), (50, 50)], 
                               n_samples=1000, input_dim=3):
    """
    Compare memory usage for naive vs vectorized implementations with different grid sizes.
    """
    results = []
    
    for width, height in grid_sizes:
        grid_name = f"{width}x{height}"
        print(f"\nCalculating memory for grid size: {grid_name}")
        
        # Calculate memory usage
        memory_usage = calculate_memory_usage(width, height, input_dim, n_samples)
        
        # For naive implementation, we don't have the large temporary arrays
        # We estimate it's primarily the weights matrix plus overhead
        naive_memory = (width * height * input_dim * 8 * 1.5) / (1024 * 1024)
        
        # Add to results
        results.append({
            'grid_size': grid_name,
            'naive_memory': naive_memory,
            'vectorized_memory': memory_usage['total'],
            'ratio': memory_usage['total'] / naive_memory
        })
        
        print(f"  Naive implementation: {naive_memory:.2f} MB")
        print(f"  Vectorized implementation: {memory_usage['total']:.2f} MB")
        print(f"  Memory ratio: {memory_usage['total'] / naive_memory:.2f}x")
    
    return results

def compare_batch_memory(batch_sizes=[None, 100, 500, 1000], 
                        width=30, height=30, n_samples=10000, input_dim=3):
    """
    Compare memory usage for different batch sizes.
    """
    results = []
    
    for batch_size in batch_sizes:
        batch_label = 'Full' if batch_size is None else str(batch_size)
        print(f"\nCalculating memory for batch size: {batch_label}")
        
        # Calculate memory usage
        memory_usage = calculate_memory_usage(width, height, input_dim, n_samples, batch_size)
        
        # Add to results
        results.append({
            'batch_size': batch_label,
            'peak_memory': memory_usage['total'],
            'squared_diffs_memory': memory_usage['squared_diffs']
        })
        
        print(f"  Total estimated memory: {memory_usage['total']:.2f} MB")
        print(f"  Main array (squared_diffs): {memory_usage['squared_diffs']:.2f} MB")
    
    return results

def plot_vectorization_memory(results):
    """Plot memory usage comparison for vectorization."""
    grid_sizes = [r['grid_size'] for r in results]
    naive_memory = [r['naive_memory'] for r in results]
    vectorized_memory = [r['vectorized_memory'] for r in results]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(grid_sizes))
    width = 0.35
    
    ax.bar(x - width/2, naive_memory, width, label='Naive Implementation', color='#FF9999')
    ax.bar(x + width/2, vectorized_memory, width, label='Vectorized Implementation', color='#66B2FF')
    
    ax.set_xlabel('Grid Size')
    ax.set_ylabel('Memory Usage (MB)')
    ax.set_title('Memory Usage Comparison: Naive vs Vectorized')
    ax.set_xticks(x)
    ax.set_xticklabels(grid_sizes)
    ax.legend()
    
    for i, v in enumerate(naive_memory):
        ax.text(i - width/2, v + 1, f"{v:.1f} MB", ha='center')
        
    for i, v in enumerate(vectorized_memory):
        ax.text(i + width/2, v + 1, f"{v:.1f} MB", ha='center')
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'theoretical_memory_vectorization.png', dpi=300)
    plt.close()

def plot_batch_memory(results):
    """Plot memory usage comparison for different batch sizes."""
    batch_sizes = [r['batch_size'] for r in results]
    peak_memory = [r['peak_memory'] for r in results]
    squared_diffs_memory = [r['squared_diffs_memory'] for r in results]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(batch_sizes))
    width = 0.35
    
    ax.bar(x - width/2, peak_memory, width, label='Total Memory', color='#FF9999')
    ax.bar(x + width/2, squared_diffs_memory, width, label='Main Array Memory', color='#66B2FF')
    
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Memory Usage (MB)')
    ax.set_title('Memory Usage by Batch Size (Theoretical)')
    ax.set_xticks(x)
    ax.set_xticklabels(batch_sizes)
    ax.legend()
    
    for i, v in enumerate(peak_memory):
        ax.text(i - width/2, v + 5, f"{v:.1f} MB", ha='center')
        
    for i, v in enumerate(squared_diffs_memory):
        ax.text(i + width/2, v + 5, f"{v:.1f} MB", ha='center')
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'theoretical_memory_batch.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    print("Calculating theoretical memory usage for Kohonen SOM")
    
    # Compare memory usage for different grid sizes
    vectorization_results = compare_vectorization_memory()
    plot_vectorization_memory(vectorization_results)
    
    # Compare memory usage for different batch sizes
    batch_results = compare_batch_memory()
    plot_batch_memory(batch_results)
    
    print("\nCalculations complete! Check the plots for theoretical memory usage patterns.") 