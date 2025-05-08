"""
Demonstration of Batch Processing Benefits in Kohonen SOM.

This script compares:
1. Fully vectorized approach (processes all vectors at once)
2. Batch processing approach (processes vectors in batches)

It measures memory usage and demonstrates how batch processing can handle larger
datasets with controlled memory consumption.
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


def measure_execution_time(func, *args, **kwargs):
    """Measure the execution time of a function"""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return result, end_time - start_time


def measure_memory_usage(func, *args, **kwargs):
    """Measure and track memory usage during function execution"""
    # Initialize process for memory tracking
    process = psutil.Process(os.getpid())
    
    # Track memory at specific intervals
    memory_profile = []
    timestamps = []
    
    # Start time
    start_time = time.time()
    start_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Define a callback to record memory usage
    def record_memory():
        current_time = time.time() - start_time
        current_memory = process.memory_info().rss / 1024 / 1024
        memory_profile.append(current_memory - start_memory)
        timestamps.append(current_time)
    
    # Record initial memory
    record_memory()
    
    # Run the function in a separate thread and monitor memory
    from threading import Thread
    
    def run_function():
        nonlocal result
        result = func(*args, **kwargs)
    
    result = None
    thread = Thread(target=run_function)
    thread.start()
    
    # Monitor memory while the function is running
    while thread.is_alive():
        record_memory()
        time.sleep(0.1)  # Check memory every 100ms
    
    thread.join()
    
    # Record final memory
    record_memory()
    
    return result, timestamps, memory_profile


def generate_training_data(n_samples=1000, input_dim=3):
    """Generate random training data"""
    return np.random.rand(n_samples, input_dim)


def run_batch_comparison(n_samples=10000, input_dim=3, n_iterations=50, batch_sizes=[None, 100, 500, 1000]):
    """
    Run comparison between different batch sizes
    """
    width, height = 30, 30  # Fixed grid size
    
    results = {
        'batch_size': [],
        'peak_memory': [],
        'execution_time': [],
        'memory_profiles': {},
        'time_profiles': {}
    }
    
    # Generate data
    print(f"Generating dataset with {n_samples} samples, {input_dim} dimensions...")
    data = generate_training_data(n_samples, input_dim)
    
    for batch_size in batch_sizes:
        batch_label = 'Full' if batch_size is None else str(batch_size)
        print(f"\nTraining with batch size: {batch_label}")
        results['batch_size'].append(batch_label)
        
        # Create a fresh SOM for this test
        som = SelfOrganizingMap(width, height, input_dim)
        
        # Measure memory usage during training
        _, time_profile, memory_profile = measure_memory_usage(
            som.train, data, n_iterations=n_iterations, batch_size=batch_size, verbose=True
        )
        
        # Store detailed profiles
        results['memory_profiles'][batch_label] = memory_profile
        results['time_profiles'][batch_label] = time_profile
        
        # Get peak memory and execution time
        peak_memory = max(memory_profile)
        execution_time = time_profile[-1]
        
        results['peak_memory'].append(peak_memory)
        results['execution_time'].append(execution_time)
        
        print(f"Peak memory: {peak_memory:.2f} MB")
        print(f"Execution time: {execution_time:.2f} seconds")
    
    return results


def plot_batch_comparison_results(results):
    """Generate comparison plots for batch processing"""
    batch_sizes = results['batch_size']
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Peak Memory Usage
    axes[0].bar(batch_sizes, results['peak_memory'], color='#66B2FF')
    axes[0].set_ylabel('Peak Memory Usage (MB)')
    axes[0].set_xlabel('Batch Size')
    axes[0].set_title('Peak Memory Usage by Batch Size')
    
    # Add values on top of bars
    for i, v in enumerate(results['peak_memory']):
        axes[0].text(i, v + 1, f"{v:.1f} MB", ha='center')
    
    # Plot 2: Execution Time
    axes[1].bar(batch_sizes, results['execution_time'], color='#99CC99')
    axes[1].set_ylabel('Execution Time (seconds)')
    axes[1].set_xlabel('Batch Size')
    axes[1].set_title('Execution Time by Batch Size')
    
    # Add values on top of bars
    for i, v in enumerate(results['execution_time']):
        axes[1].text(i, v + 0.5, f"{v:.1f} s", ha='center')
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'batch_comparison.png', dpi=300)
    print(f"Saved batch comparison plot to {RESULTS_DIR / 'batch_comparison.png'}")
    plt.close()
    
    # Plot memory profiles over time
    plt.figure(figsize=(12, 6))
    
    for batch_size, memory_profile in results['memory_profiles'].items():
        time_profile = results['time_profiles'][batch_size]
        plt.plot(time_profile, memory_profile, label=f'Batch Size: {batch_size}', linewidth=2)
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Memory Usage (MB)')
    plt.title('Memory Usage Over Time for Different Batch Sizes')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.savefig(RESULTS_DIR / 'memory_profiles.png', dpi=300)
    print(f"Saved memory profiles plot to {RESULTS_DIR / 'memory_profiles.png'}")
    plt.close()
    
    # Create a diagram to explain batch processing
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    diagram_text = """
    Full Vectorization vs. Batch Processing
    --------------------------------------
    
    Full Vectorization (batch_size=None):
    -------------------------------------
    All input vectors (N samples) are processed simultaneously
    
    Memory Usage = O(N * width * height)
    Pros: Fastest for small to medium datasets
    Cons: Memory usage grows linearly with sample count
    
    
    Batch Processing (batch_size=B):
    -------------------------------
    Input vectors are processed in batches of size B
    
    Memory Usage = O(B * width * height)
    Pros: Controlled memory usage regardless of total sample count
    Cons: Slightly slower due to more Python-level operations
    
    
                    Memory Usage
                        ^
                        |
    Full               /
    Vectorization --> /
                      /
                     /         Batch Processing
                    /          (constant memory)
                   /           ---------------->
                  /
                 /
                +--------------------------------> Dataset Size
    """
    
    ax.text(0.1, 0.5, diagram_text, fontsize=12, family='monospace', va='center')
    
    plt.savefig(RESULTS_DIR / 'batch_processing_diagram.png', dpi=300, bbox_inches='tight')
    print(f"Saved batch processing diagram to {RESULTS_DIR / 'batch_processing_diagram.png'}")
    plt.close()
    
    # Create a large vs small dataset comparison diagram
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    large_vs_small_text = """
    Dataset Size Impact on Memory Usage
    -----------------------------------
    
    Small Dataset (1,000 samples):
    -----------------------------
    Full Vectorization: Efficient and fast
    Memory Usage: Low (both approaches similar)
    Recommendation: Use batch_size=None
    
    
    Medium Dataset (10,000 samples):
    ------------------------------
    Full Vectorization: Higher memory usage but still fast
    Batch Processing: Lower memory, similar speed
    Recommendation: Consider batch_size=500 if memory constrained
    
    
    Large Dataset (100,000+ samples):
    ------------------------------
    Full Vectorization: Very high memory usage, potential OOM errors
    Batch Processing: Controlled memory usage, slightly slower
    Recommendation: Use batch_size=500 or batch_size=1000
    
    
    General Guideline:
    -----------------
    1. For quick prototyping or small datasets: batch_size=None
    2. For production with large datasets: batch_size=500 or 1000
    3. For memory-constrained environments: Use smaller batch sizes
    """
    
    ax.text(0.1, 0.5, large_vs_small_text, fontsize=12, family='monospace', va='center')
    
    plt.savefig(RESULTS_DIR / 'dataset_size_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved dataset size comparison diagram to {RESULTS_DIR / 'dataset_size_comparison.png'}")
    plt.close()


if __name__ == "__main__":
    print("Running Kohonen SOM Batch Processing Comparison")
    
    # Run comparison with different batch sizes on a medium-sized dataset
    batch_results = run_batch_comparison(
        n_samples=10000,  # 10,000 samples for more pronounced effect
        input_dim=3,
        n_iterations=20,  # Fewer iterations for quicker demonstration
        batch_sizes=[None, 100, 500, 1000]
    )
    
    # Plot results
    plot_batch_comparison_results(batch_results)
    
    print("\nComparison complete! Check the plots for visualization of memory usage patterns.") 