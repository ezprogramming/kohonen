"""
Demonstration of MLflow Integration in Kohonen SOM.

This script showcases:
1. How to track SOM training experiments with MLflow
2. Logging parameters, metrics, and artifacts
3. How to load models from MLflow
4. Benefits of experiment tracking for model comparison

The script will train multiple SOMs with different parameters and track results.
"""
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import tempfile
import shutil
import pandas as pd
import json

# Add the project root to the path so we can import the package
sys.path.append(str(Path(__file__).resolve().parents[2]))

from kohonen import SelfOrganizingMap
from kohonen.mlflow_utils import log_som_model, load_som_model

# Set random seed for reproducibility
np.random.seed(42)

# Directory for saving comparison results
RESULTS_DIR = Path(__file__).parent
os.makedirs(RESULTS_DIR, exist_ok=True)

# Create a temporary directory for MLflow tracking (for demonstration)
DEMO_MLFLOW_DIR = Path(tempfile.mkdtemp(prefix="kohonen_mlflow_demo_"))
# Set the tracking URI as an environment variable
os.environ["MLFLOW_TRACKING_URI"] = f"file:{DEMO_MLFLOW_DIR}"


def generate_training_data(n_samples=1000, input_dim=3):
    """Generate random training data"""
    return np.random.rand(n_samples, input_dim)


def train_multiple_models(experiment_name="kohonen_comparison"):
    """
    Train multiple SOMs with different parameters and track in MLflow
    
    Returns:
    --------
    dict
        Dictionary of run IDs and their parameters/metrics
    """
    # Set up grid search parameters
    width_height_pairs = [(20, 20), (30, 30), (50, 50)]
    iterations = [100, 200]
    learning_rates = [0.1, 0.05]
    
    # Generate fixed training data
    n_samples = 5000
    input_dim = 3
    print(f"Generating training data: {n_samples} samples, {input_dim} dimensions")
    data = generate_training_data(n_samples, input_dim)
    
    results = []
    
    # Run grid search
    for width, height in width_height_pairs:
        for n_iterations in iterations:
            for learning_rate in learning_rates:
                print(f"\nTraining SOM: {width}x{height} grid, {n_iterations} iterations, lr={learning_rate}")
                
                # Create and train SOM
                som = SelfOrganizingMap(
                    width=width,
                    height=height,
                    input_dim=input_dim,
                    learning_rate=learning_rate
                )
                
                # Train the model and get metrics
                training_metrics = som.train(data, n_iterations=n_iterations, verbose=True)
                
                # Get final quantization error
                final_qe = training_metrics['quantization_error']
                print(f"Training complete. Final quantization error: {final_qe:.6f}")
                
                # Create parameter dict
                params = {
                    'width': width,
                    'height': height,
                    'input_dim': input_dim,
                    'learning_rate': learning_rate,
                    'iterations': n_iterations,
                    'samples': n_samples
                }
                
                # Log the trained model to MLflow
                run_name = f"som_{width}x{height}_i{n_iterations}_lr{learning_rate}"
                run_id = log_som_model(
                    model=som,
                    input_data=data,
                    params=params,
                    training_metrics=training_metrics,
                    run_name=run_name,
                    save_visualizations=True
                )
                
                # Store results for comparison
                result = {
                    'run_id': run_id,
                    'run_name': run_name,
                    'width': width,
                    'height': height,
                    'learning_rate': learning_rate,
                    'iterations': n_iterations,
                    'quantization_error': final_qe
                }
                results.append(result)
                
                print(f"Logged model to MLflow with run_id: {run_id}")
    
    return results


def load_and_test_model(run_id, grid_size=25):
    """
    Load a model from MLflow and test it
    
    Returns:
    --------
    np.ndarray
        Model weights
    """
    print(f"\nLoading model from run_id: {run_id}")
    
    # Load the model from MLflow
    loaded_som = load_som_model(run_id)
    
    # Generate test point
    test_point = np.array([0.2, 0.5, 0.8])
    
    # Find BMU
    bmu_x, bmu_y = loaded_som.predict_bmu(test_point)
    print(f"BMU for test point {test_point}: ({bmu_x}, {bmu_y})")
    
    # Get weights for visualization
    weights = loaded_som.get_weights()
    
    return weights


def create_model_comparison_table(results):
    """Create a pandas DataFrame for model comparison"""
    df = pd.DataFrame(results)
    df = df.sort_values('quantization_error')
    
    return df


def create_mlflow_diagrams():
    """Create diagrams explaining MLflow integration"""
    # Create general MLflow workflow diagram
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    workflow_text = """
    Traditional ML Workflow vs. MLflow Integration
    ---------------------------------------------
    
    Traditional Approach:
    -------------------
    1. Train model with parameters
    2. Manually record metrics in spreadsheet/notes
    3. Save model to filesystem
    4. Record file path in documentation
    5. When using model later, need to remember all parameters
    
    MLflow Integration:
    -----------------
    1. Train model with parameters
    2. MLflow automatically logs:
       - All hyperparameters
       - Performance metrics at each step
       - Model artifacts (weights, visualizations)
       - Runtime environment
    3. Find best models by querying metrics
    4. Load model directly from MLflow
    
    Benefits:
    --------
    1. Reproducibility - track all parameters, code versions
    2. Discoverability - find all past experiments
    3. Observability - visualize metrics across experiments
    4. Production-readiness - standardized model loading
    """
    
    ax.text(0.1, 0.5, workflow_text, fontsize=12, family='monospace', va='center')
    
    plt.savefig(RESULTS_DIR / 'mlflow_workflow.png', dpi=300, bbox_inches='tight')
    print(f"Saved MLflow workflow diagram to {RESULTS_DIR / 'mlflow_workflow.png'}")
    plt.close()
    
    # Create MLflow SOM artifacts diagram
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    artifacts_text = """
    MLflow Tracking for Kohonen SOM
    -----------------------------
    
    For each SOM experiment, the following is tracked:
    
    Parameters:
    - Grid dimensions (width, height)
    - Input dimensions
    - Learning rate
    - Number of iterations
    - Dataset size
    - Sigma (neighborhood radius)
    - Random seed
    
    Metrics:
    - Quantization error per 100 iterations
    - Final quantization error
    - Training time
    
    Artifacts:
    - Model weights (numpy .npy file)
    - SOM grid visualization
    - Component planes for each input dimension
    - Quantization error plot
    - Sample BMU map
    
    Environment:
    - Python version
    - Library dependencies
    - System information
    """
    
    ax.text(0.1, 0.5, artifacts_text, fontsize=12, family='monospace', va='center')
    
    plt.savefig(RESULTS_DIR / 'mlflow_artifacts.png', dpi=300, bbox_inches='tight')
    print(f"Saved MLflow artifacts diagram to {RESULTS_DIR / 'mlflow_artifacts.png'}")
    plt.close()


def plot_model_comparison(results_df):
    """Create plots comparing models based on parameters"""
    # 1. Effect of grid size on quantization error
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Group by grid size and average the quantization error
    grid_sizes = [f"{row.width}x{row.height}" for _, row in results_df.iterrows()]
    results_df['grid_size'] = grid_sizes
    grid_performance = results_df.groupby('grid_size')['quantization_error'].mean().reset_index()
    
    ax.bar(grid_performance['grid_size'], grid_performance['quantization_error'], color='#66B2FF')
    ax.set_xlabel('Grid Size')
    ax.set_ylabel('Average Quantization Error')
    ax.set_title('Effect of Grid Size on Model Performance')
    
    # Add values on top of bars
    for i, v in enumerate(grid_performance['quantization_error']):
        ax.text(i, v - 0.01, f"{v:.4f}", ha='center')
    
    plt.savefig(RESULTS_DIR / 'grid_size_comparison.png', dpi=300)
    print(f"Saved grid size comparison to {RESULTS_DIR / 'grid_size_comparison.png'}")
    plt.close()
    
    # 2. Effect of iterations on quantization error
    fig, ax = plt.subplots(figsize=(10, 6))
    
    iteration_performance = results_df.groupby('iterations')['quantization_error'].mean().reset_index()
    
    ax.bar(iteration_performance['iterations'].astype(str), 
           iteration_performance['quantization_error'], 
           color='#99CC99')
    ax.set_xlabel('Number of Iterations')
    ax.set_ylabel('Average Quantization Error')
    ax.set_title('Effect of Iteration Count on Model Performance')
    
    # Add values on top of bars
    for i, v in enumerate(iteration_performance['quantization_error']):
        ax.text(i, v - 0.01, f"{v:.4f}", ha='center')
    
    plt.savefig(RESULTS_DIR / 'iterations_comparison.png', dpi=300)
    print(f"Saved iterations comparison to {RESULTS_DIR / 'iterations_comparison.png'}")
    plt.close()
    
    # 3. Effect of learning rate on quantization error
    fig, ax = plt.subplots(figsize=(10, 6))
    
    lr_performance = results_df.groupby('learning_rate')['quantization_error'].mean().reset_index()
    
    ax.bar(lr_performance['learning_rate'].astype(str), 
           lr_performance['quantization_error'], 
           color='#FF9999')
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Average Quantization Error')
    ax.set_title('Effect of Learning Rate on Model Performance')
    
    # Add values on top of bars
    for i, v in enumerate(lr_performance['quantization_error']):
        ax.text(i, v - 0.01, f"{v:.4f}", ha='center')
    
    plt.savefig(RESULTS_DIR / 'learning_rate_comparison.png', dpi=300)
    print(f"Saved learning rate comparison to {RESULTS_DIR / 'learning_rate_comparison.png'}")
    plt.close()


def save_best_model_info(results_df):
    """Save information about the best model"""
    # Get the best model (lowest quantization error)
    best_model = results_df.iloc[0]
    
    # Format model info for display
    model_info = {
        'run_id': best_model['run_id'],
        'run_name': best_model['run_name'],
        'grid_size': f"{best_model['width']}x{best_model['height']}",
        'learning_rate': float(best_model['learning_rate']),
        'iterations': int(best_model['iterations']),
        'quantization_error': float(best_model['quantization_error'])
    }
    
    # Save as JSON
    with open(RESULTS_DIR / 'best_model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print(f"Saved best model info to {RESULTS_DIR / 'best_model_info.json'}")
    
    return model_info


def cleanup():
    """Clean up temporary MLflow directory"""
    shutil.rmtree(DEMO_MLFLOW_DIR)
    print(f"Cleaned up temporary MLflow directory: {DEMO_MLFLOW_DIR}")


if __name__ == "__main__":
    print("Running Kohonen SOM MLflow Integration Demonstration")
    print(f"Using temporary MLflow directory: {DEMO_MLFLOW_DIR}")
    
    # 1. Train multiple models with different parameters
    results = train_multiple_models()
    
    # 2. Create comparison table
    comparison_df = create_model_comparison_table(results)
    
    # 3. Save comparison table
    comparison_df.to_csv(RESULTS_DIR / 'model_comparison.csv', index=False)
    print(f"Saved model comparison table to {RESULTS_DIR / 'model_comparison.csv'}")
    
    # 4. Plot model comparisons
    plot_model_comparison(comparison_df)
    
    # 5. Save best model info
    best_model_info = save_best_model_info(comparison_df)
    
    # 6. Load and test best model
    best_run_id = best_model_info['run_id']
    weights = load_and_test_model(best_run_id)
    
    # 7. Create MLflow diagrams
    create_mlflow_diagrams()
    
    # 8. Clean up (in practice, you would keep the MLflow directory)
    cleanup()
    
    print("\nMLflow integration demonstration complete!") 