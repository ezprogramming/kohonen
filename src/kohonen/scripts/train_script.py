#!/usr/bin/env python3
"""
Script to train a SOM model and log it to MLflow.
"""
import numpy as np
import os
import logging
import argparse
import mlflow
from dotenv import load_dotenv
from kohonen import SelfOrganizingMap
from kohonen.mlflow_utils import log_som_model

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    # Get default values from environment variables
    env_width = int(os.getenv('SOM_WIDTH', 30))
    env_height = int(os.getenv('SOM_HEIGHT', 30))
    env_input_dim = int(os.getenv('SOM_INPUT_DIM', 3))
    env_iterations = int(os.getenv('SOM_ITERATIONS', 100))
    env_samples = int(os.getenv('SOM_SAMPLES', 1000))
    env_learning_rate = float(os.getenv('SOM_LEARNING_RATE', 0.1))
    env_sigma = os.getenv('SOM_SIGMA', None)
    if env_sigma and env_sigma.strip():
        env_sigma = float(env_sigma)
    else:
        env_sigma = None
    env_random_state = os.getenv('SOM_RANDOM_STATE', None)
    if env_random_state and env_random_state.strip():
        env_random_state = int(env_random_state)
    else:
        env_random_state = None
    env_run_name = os.getenv('SOM_RUN_NAME', 'som_model')
    env_verbose = os.getenv('SOM_VERBOSE', 'false').lower() in ('true', '1', 't', 'yes', 'y')
    
    # Parse batch size environment variable
    env_batch_size = os.getenv('SOM_BATCH_SIZE', None)
    if env_batch_size and env_batch_size.strip():
        env_batch_size = int(env_batch_size)
    else:
        env_batch_size = None
    
    # Print debug information
    logger.info(f"Environment variables: WIDTH={env_width}, HEIGHT={env_height}, ITERATIONS={env_iterations}, SAMPLES={env_samples}")
    
    parser = argparse.ArgumentParser(description="Train a SOM model and log it to MLflow.")
    parser.add_argument("--width", type=int, default=env_width, help=f"Width of the SOM grid (default: {env_width})")
    parser.add_argument("--height", type=int, default=env_height, help=f"Height of the SOM grid (default: {env_height})")
    parser.add_argument("--input-dim", type=int, default=env_input_dim, help=f"Input dimension for the SOM (default: {env_input_dim})")
    parser.add_argument("--iterations", type=int, default=env_iterations, help=f"Number of training iterations (default: {env_iterations})")
    parser.add_argument("--samples", type=int, default=env_samples, help=f"Number of random samples to generate for training (default: {env_samples})")
    parser.add_argument("--learning-rate", type=float, default=env_learning_rate, help=f"Initial learning rate (default: {env_learning_rate})")
    parser.add_argument("--sigma", type=float, default=env_sigma, help=f"Initial neighborhood radius (default: {env_sigma if env_sigma is not None else 'max(width, height)/2'})")
    parser.add_argument("--random-state", type=int, default=env_random_state, help=f"Random seed for reproducibility (default: {env_random_state})")
    parser.add_argument("--run-name", type=str, default=env_run_name, help=f"Name for the MLflow run (default: {env_run_name})")
    parser.add_argument("--verbose", action="store_true", default=env_verbose, help=f"Enable verbose output during training (default: {env_verbose})")
    parser.add_argument("--batch-size", type=int, default=env_batch_size, help=f"Batch size for memory-efficient training (default: {env_batch_size if env_batch_size is not None else 'None (fully vectorized)'})")
    
    args = parser.parse_args()
    
    # Print parsed arguments
    logger.info(f"Parsed arguments: WIDTH={args.width}, HEIGHT={args.height}, ITERATIONS={args.iterations}, SAMPLES={args.samples}")
    
    return args

def train_and_log_model(args):
    """Train a SOM model and log it to MLflow."""
    # Configure MLflow
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    logger.info(f"Setting MLflow tracking URI to: {mlflow_uri}")
    mlflow.set_tracking_uri(mlflow_uri)
    
    # Create and train SOM
    logger.info(f"Training SOM with dimensions {args.width}x{args.height}, input dimension {args.input_dim}")
    som = SelfOrganizingMap(
        width=args.width, 
        height=args.height, 
        input_dim=args.input_dim,
        learning_rate=args.learning_rate,
        sigma=args.sigma,
        random_state=args.random_state
    )
    
    # Generate random training data
    logger.info(f"Generating {args.samples} random training samples")
    data = np.random.rand(args.samples, args.input_dim)
    
    # Train the model
    logger.info(f"Starting training for {args.iterations} iterations")
    metrics = som.train(data, n_iterations=args.iterations, verbose=args.verbose, batch_size=args.batch_size)
    
    # Log the model to MLflow
    logger.info("Logging model to MLflow...")
    run_id = log_som_model(
        model=som,
        input_data=data,
        params={
            "width": args.width,
            "height": args.height,
            "input_dim": args.input_dim,
            "n_iterations": args.iterations,
            "n_samples": args.samples,
            "learning_rate": args.learning_rate,
            "sigma": args.sigma if args.sigma is not None else f"auto ({som.sigma_0})",
            "random_state": args.random_state,
            "batch_size": args.batch_size if args.batch_size is not None else "None (fully vectorized)"
        },
        training_metrics=metrics,
        run_name=args.run_name,
        save_visualizations=True
    )
    
    # Save run ID to shared volume
    logger.info(f"Model trained and logged with run ID: {run_id}")
    run_id_file = os.getenv("RUN_ID_FILE", "/app/mlflow_data/run_id.txt")
    with open(run_id_file, "w") as f:
        f.write(run_id.strip())  # Ensure no extra characters
    
    return run_id

if __name__ == "__main__":
    args = parse_args()
    run_id = train_and_log_model(args)
    
    # Print MLflow URLs for convenience
    mlflow_url = os.environ.get("MLFLOW_UI_URL", "http://localhost:5050")
    print(f"\n[SUCCESS] Model training complete!")
    print(f"[INFO] View model in MLflow UI: {mlflow_url}")
    print(f"[INFO] Run ID: {run_id}") 