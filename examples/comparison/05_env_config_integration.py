"""
Demonstration of Environment Variable Configuration in Kohonen SOM.

This script showcases:
1. How to configure SOM parameters using environment variables
2. Comparison between hardcoded configuration and environment-based configuration
3. Benefits of environment variable configuration for deployment flexibility
4. Integration with Docker and docker-compose

The script demonstrates how to make the SOM implementation more configurable without
code changes.
"""
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import json
import tempfile
import shutil
import subprocess
from dotenv import load_dotenv, dotenv_values
from typing import Dict, Any, Optional

# Add the project root to the path so we can import the package
sys.path.append(str(Path(__file__).resolve().parents[2]))

from kohonen import SelfOrganizingMap

# Set random seed for reproducibility
np.random.seed(42)

# Directory for saving comparison results
RESULTS_DIR = Path(__file__).parent
os.makedirs(RESULTS_DIR, exist_ok=True)

# Create a temporary directory for env files
TEMP_ENV_DIR = Path(tempfile.mkdtemp(prefix="kohonen_env_demo_"))


def generate_training_data(n_samples=1000, input_dim=3):
    """Generate random training data"""
    return np.random.rand(n_samples, input_dim)


def create_env_files():
    """Create different .env files for demonstration"""
    # Basic configuration - small grid
    basic_env = """
# SOM Configuration
SOM_WIDTH=10
SOM_HEIGHT=10
SOM_INPUT_DIM=3
SOM_ITERATIONS=100
SOM_SAMPLES=1000
SOM_LEARNING_RATE=0.1
SOM_SIGMA=
SOM_RANDOM_STATE=42
SOM_BATCH_SIZE=
    """
    
    # Medium configuration - larger grid, more samples
    medium_env = """
# SOM Configuration
SOM_WIDTH=30
SOM_HEIGHT=30
SOM_INPUT_DIM=3
SOM_ITERATIONS=200
SOM_SAMPLES=5000
SOM_LEARNING_RATE=0.05
SOM_SIGMA=
SOM_RANDOM_STATE=42
SOM_BATCH_SIZE=500
    """
    
    # Large configuration - large grid, many samples, batched
    large_env = """
# SOM Configuration
SOM_WIDTH=50
SOM_HEIGHT=50
SOM_INPUT_DIM=3
SOM_ITERATIONS=300
SOM_SAMPLES=10000
SOM_LEARNING_RATE=0.1
SOM_SIGMA=
SOM_RANDOM_STATE=42
SOM_BATCH_SIZE=1000
    """
    
    # Save env files
    (TEMP_ENV_DIR / "basic.env").write_text(basic_env.strip())
    (TEMP_ENV_DIR / "medium.env").write_text(medium_env.strip())
    (TEMP_ENV_DIR / "large.env").write_text(large_env.strip())
    
    # Also save to the comparison directory for reference
    (RESULTS_DIR / "example.env").write_text(medium_env.strip())
    
    print(f"Created environment files in {TEMP_ENV_DIR}")
    return TEMP_ENV_DIR


def load_env_file(env_file):
    """Load an environment file and set environment variables"""
    # First, clear any existing SOM_ environment variables
    for key in list(os.environ.keys()):
        if key.startswith("SOM_"):
            del os.environ[key]
    
    # Load the new environment file
    env_config = dotenv_values(env_file)
    
    # Set the environment variables
    for key, value in env_config.items():
        if value:  # Only set if the value is not empty
            os.environ[key] = value
    
    return env_config


def get_som_config_from_env():
    """Get SOM configuration from environment variables"""
    config = {
        'width': int(os.environ.get('SOM_WIDTH', 20)),
        'height': int(os.environ.get('SOM_HEIGHT', 20)),
        'input_dim': int(os.environ.get('SOM_INPUT_DIM', 3)),
        'learning_rate': float(os.environ.get('SOM_LEARNING_RATE', 0.1)),
        'iterations': int(os.environ.get('SOM_ITERATIONS', 100)),
        'samples': int(os.environ.get('SOM_SAMPLES', 1000)),
        'random_state': int(os.environ.get('SOM_RANDOM_STATE', 42)) if os.environ.get('SOM_RANDOM_STATE') else None,
        'batch_size': int(os.environ.get('SOM_BATCH_SIZE')) if os.environ.get('SOM_BATCH_SIZE') else None,
        'sigma': float(os.environ.get('SOM_SIGMA')) if os.environ.get('SOM_SIGMA') else None
    }
    
    return config


def train_som_with_config(config):
    """Train a SOM with the given configuration"""
    # Extract training parameters
    n_samples = config['samples']
    input_dim = config['input_dim']
    n_iterations = config['iterations']
    batch_size = config['batch_size']
    
    # Create SOM
    som = SelfOrganizingMap(
        width=config['width'],
        height=config['height'],
        input_dim=input_dim,
        learning_rate=config['learning_rate'],
        sigma=config['sigma'],
        random_state=config['random_state']
    )
    
    # Generate training data
    data = generate_training_data(n_samples, input_dim)
    
    # Train and measure execution time
    start_time = time.time()
    metrics = som.train(data, n_iterations=n_iterations, batch_size=batch_size, verbose=True)
    end_time = time.time()
    
    training_time = end_time - start_time
    
    return som, metrics, training_time


def run_env_comparison():
    """Run training with different environment configurations"""
    # Create environment files
    env_dir = create_env_files()
    
    results = {
        'configs': [],
        'metrics': [],
        'training_times': []
    }
    
    env_files = ['basic.env', 'medium.env', 'large.env']
    
    for env_file in env_files:
        env_path = env_dir / env_file
        print(f"\nLoading environment from {env_file}")
        
        # Load environment
        env_config = load_env_file(env_path)
        
        # Get SOM configuration
        config = get_som_config_from_env()
        print(f"Configuration: {json.dumps(config, indent=2)}")
        
        # Train SOM
        print(f"Training SOM with {config['width']}x{config['height']} grid, {config['samples']} samples...")
        som, metrics, training_time = train_som_with_config(config)
        
        # Store results
        config_name = env_file.replace('.env', '')
        results['configs'].append(config_name)
        results['metrics'].append(metrics)
        results['training_times'].append(training_time)
        
        print(f"Training completed in {training_time:.2f} seconds")
        print(f"Final quantization error: {metrics['quantization_error']:.6f}")
    
    return results


def create_docker_artifacts():
    """Create Dockerfile and docker-compose.yml examples for environment configuration"""
    # Example Dockerfile that uses environment variables
    dockerfile = """
FROM python:3.9-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ /app/src/
COPY pyproject.toml .

# Install the package
RUN pip install -e .

# Create directory for MLflow data
RUN mkdir -p /app/mlflow_data

# Environment variables are provided at runtime via docker-compose.yml
# or the docker run command with -e or --env-file

# Default command to run the training script
CMD ["python", "-m", "kohonen.scripts.train_script"]
    """
    
    # Example docker-compose.yml that uses environment variables
    docker_compose = """
version: '3'

services:
  mlflow:
    build: .
    ports:
      - "5050:5000"
    volumes:
      - ./mlflow_data:/app/mlflow_data
    command: ["python", "-m", "mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]
    networks:
      - kohonen-network

  trainer:
    build: .
    depends_on:
      - mlflow
    volumes:
      - ./mlflow_data:/app/mlflow_data
    env_file:
      - .env
    environment:
      - MLFLOW_TRACKING_URI=http://mlflow:5000
      - RUN_ID_FILE=/app/mlflow_data/run_id.txt
    networks:
      - kohonen-network

  api:
    build: .
    ports:
      - "8000:8000"
    depends_on:
      - mlflow
    volumes:
      - ./mlflow_data:/app/mlflow_data
    env_file:
      - .env
    environment:
      - MLFLOW_TRACKING_URI=http://mlflow:5000
      - SOM_RUN_ID_FILE=/app/mlflow_data/run_id.txt
      - PORT=8000
    command: ["python", "-m", "kohonen.scripts.api_script"]
    networks:
      - kohonen-network

networks:
  kohonen-network:
    """
    
    # Save the files for reference
    (RESULTS_DIR / "Dockerfile.example").write_text(dockerfile.strip())
    (RESULTS_DIR / "docker-compose.example.yml").write_text(docker_compose.strip())
    
    print(f"Created Docker artifacts in {RESULTS_DIR}")


def create_comparison_diagrams():
    """Create diagrams explaining the environment variable configuration"""
    # Create env vs hardcoded diagram
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    comparison_text = """
    Hardcoded Configuration vs. Environment Variable Configuration
    -----------------------------------------------------------
    
    Hardcoded Configuration:
    ---------------------
    ```python
    # Parameters are hardcoded in the script
    width = 30
    height = 30
    input_dim = 3
    learning_rate = 0.1
    iterations = 200
    samples = 5000
    
    # To change parameters, you must modify the source code
    som = SelfOrganizingMap(width, height, input_dim, learning_rate)
    ```
    
    Environment Variable Configuration:
    --------------------------------
    ```python
    # Parameters are read from environment variables with defaults
    width = int(os.environ.get('SOM_WIDTH', 30))
    height = int(os.environ.get('SOM_HEIGHT', 30))
    input_dim = int(os.environ.get('SOM_INPUT_DIM', 3))
    learning_rate = float(os.environ.get('SOM_LEARNING_RATE', 0.1))
    iterations = int(os.environ.get('SOM_ITERATIONS', 200))
    samples = int(os.environ.get('SOM_SAMPLES', 5000))
    
    # To change parameters, no code modification needed
    som = SelfOrganizingMap(width, height, input_dim, learning_rate)
    ```
    
    Benefits:
    -------
    1. No code changes needed to adjust parameters
    2. Different environments can use different configurations
    3. Easy integration with Docker and orchestration tools
    4. Configuration can be version controlled separately
    5. Makes testing with different parameters simpler
    """
    
    ax.text(0.1, 0.5, comparison_text, fontsize=12, family='monospace', va='center')
    
    plt.savefig(RESULTS_DIR / 'env_config_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved environment config comparison to {RESULTS_DIR / 'env_config_comparison.png'}")
    plt.close()
    
    # Create Docker workflow diagram
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    docker_text = """
    Docker Integration with Environment Variables
    ------------------------------------------
    
    Development Workflow:
    ------------------
    1. Developer creates a local .env file with development settings
       ```
       SOM_WIDTH=10
       SOM_HEIGHT=10
       SOM_ITERATIONS=50
       ```
    
    2. Run locally for quick testing:
       ```bash
       python -m kohonen.scripts.train_script
       ```
    
    Production Deployment:
    -------------------
    1. Create production .env file with optimized settings
       ```
       SOM_WIDTH=50
       SOM_HEIGHT=50
       SOM_ITERATIONS=300
       SOM_BATCH_SIZE=1000
       ```
    
    2. Deploy with Docker and environment variables:
       ```bash
       docker run --env-file production.env kohonen-som
       ```
    
    3. Or deploy with docker-compose:
       ```bash
       docker-compose --env-file production.env up
       ```
    
    CI/CD Pipeline:
    ------------
    1. Test with different configurations automatically:
       ```bash
       # Test small config
       docker run --env-file test_small.env kohonen-som
       
       # Test large config
       docker run --env-file test_large.env kohonen-som
       ```
    
    2. No code changes or rebuilds needed for different configurations
    """
    
    ax.text(0.1, 0.5, docker_text, fontsize=12, family='monospace', va='center')
    
    plt.savefig(RESULTS_DIR / 'docker_env_workflow.png', dpi=300, bbox_inches='tight')
    print(f"Saved Docker workflow diagram to {RESULTS_DIR / 'docker_env_workflow.png'}")
    plt.close()


def plot_comparison_results(results):
    """Plot comparison between different environment configurations"""
    configs = results['configs']
    training_times = results['training_times']
    quantization_errors = [metrics['quantization_error'] for metrics in results['metrics']]
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Training Times
    axes[0].bar(configs, training_times, color='#66B2FF')
    axes[0].set_ylabel('Training Time (seconds)')
    axes[0].set_xlabel('Configuration')
    axes[0].set_title('Training Time by Configuration')
    
    # Add values on top of bars
    for i, v in enumerate(training_times):
        axes[0].text(i, v + 0.5, f"{v:.1f}s", ha='center')
    
    # Plot 2: Quantization Errors
    axes[1].bar(configs, quantization_errors, color='#FF9999')
    axes[1].set_ylabel('Quantization Error')
    axes[1].set_xlabel('Configuration')
    axes[1].set_title('Model Performance by Configuration')
    
    # Add values on top of bars
    for i, v in enumerate(quantization_errors):
        axes[1].text(i, v - 0.01, f"{v:.4f}", ha='center')
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'env_config_results.png', dpi=300)
    print(f"Saved environment config results to {RESULTS_DIR / 'env_config_results.png'}")
    plt.close()
    
    # Create a table visualization of the configurations
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    # Load the env files to display their content
    basic_env = dotenv_values(TEMP_ENV_DIR / "basic.env")
    medium_env = dotenv_values(TEMP_ENV_DIR / "medium.env")
    large_env = dotenv_values(TEMP_ENV_DIR / "large.env")
    
    # Create a table with the configurations
    table_text = """
    Configuration Comparison
    -----------------------
    
    Parameter       | Basic Config  | Medium Config | Large Config
    ---------------|--------------|--------------|-------------
    SOM_WIDTH       | {0:<12} | {1:<12} | {2:<12}
    SOM_HEIGHT      | {3:<12} | {4:<12} | {5:<12}
    SOM_ITERATIONS  | {6:<12} | {7:<12} | {8:<12}
    SOM_SAMPLES     | {9:<12} | {10:<12} | {11:<12}
    SOM_BATCH_SIZE  | {12:<12} | {13:<12} | {14:<12}
    
    Performance Metrics:
    ------------------
    Training Time    | {15:<12.2f}s | {16:<12.2f}s | {17:<12.2f}s
    Final Error      | {18:<12.4f} | {19:<12.4f} | {20:<12.4f}
    """.format(
        basic_env.get('SOM_WIDTH', ''), medium_env.get('SOM_WIDTH', ''), large_env.get('SOM_WIDTH', ''),
        basic_env.get('SOM_HEIGHT', ''), medium_env.get('SOM_HEIGHT', ''), large_env.get('SOM_HEIGHT', ''),
        basic_env.get('SOM_ITERATIONS', ''), medium_env.get('SOM_ITERATIONS', ''), large_env.get('SOM_ITERATIONS', ''),
        basic_env.get('SOM_SAMPLES', ''), medium_env.get('SOM_SAMPLES', ''), large_env.get('SOM_SAMPLES', ''),
        basic_env.get('SOM_BATCH_SIZE', 'None'), medium_env.get('SOM_BATCH_SIZE', 'None'), large_env.get('SOM_BATCH_SIZE', 'None'),
        results['training_times'][0], results['training_times'][1], results['training_times'][2],
        results['metrics'][0]['quantization_error'], results['metrics'][1]['quantization_error'], results['metrics'][2]['quantization_error']
    )
    
    ax.text(0.1, 0.5, table_text, fontsize=12, family='monospace', va='center')
    
    plt.savefig(RESULTS_DIR / 'config_table.png', dpi=300, bbox_inches='tight')
    print(f"Saved configuration table to {RESULTS_DIR / 'config_table.png'}")
    plt.close()


def cleanup():
    """Clean up temporary directory"""
    shutil.rmtree(TEMP_ENV_DIR)
    print(f"Cleaned up temporary directory: {TEMP_ENV_DIR}")


if __name__ == "__main__":
    print("Running Kohonen SOM Environment Configuration Demonstration")
    
    # 1. Create diagrams to explain environment configuration
    create_comparison_diagrams()
    
    # 2. Create example Docker artifacts
    create_docker_artifacts()
    
    # 3. Run training with different environment configurations
    results = run_env_comparison()
    
    # 4. Plot comparison results
    plot_comparison_results(results)
    
    # 5. Clean up
    cleanup()
    
    print("\nEnvironment configuration demonstration complete!") 