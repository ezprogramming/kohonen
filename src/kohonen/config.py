"""
Configuration settings for the Kohonen SOM package.

This module provides centralized configuration using Pydantic.
"""
import os
import logging
from typing import Optional, Dict, Any
import json

import mlflow

# Configure logging
logger = logging.getLogger(__name__)

# Settings class as a simple Python object for compatibility
class Settings:
    """
    Settings for the Kohonen SOM package loaded from environment variables.
    """
    def __init__(self):
        # MLflow Configuration
        self.mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "")
        self.mlflow_experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "som-experiments")
        
        # Model Selection Configuration
        self.metric_key = os.environ.get("SOM_METRIC_KEY", "quantization_error")
        
        # Handle boolean env var
        metric_asc = os.environ.get("SOM_METRIC_ASCENDING", "true").lower()
        self.metric_ascending = metric_asc in ("true", "1", "t", "yes", "y")
        
        # SOM Configuration
        self.som_width = int(os.environ.get("SOM_WIDTH", "50"))
        self.som_height = int(os.environ.get("SOM_HEIGHT", "50"))
        self.som_input_dim = int(os.environ.get("SOM_INPUT_DIM", "3"))
        self.som_iterations = int(os.environ.get("SOM_ITERATIONS", "300"))
        self.som_samples = int(os.environ.get("SOM_SAMPLES", "1000"))
        self.som_learning_rate = float(os.environ.get("SOM_LEARNING_RATE", "0.1"))
        
        # Handle batch size (can be None)
        batch_size = os.environ.get("SOM_BATCH_SIZE", "")
        self.som_batch_size = int(batch_size) if batch_size.strip() else None
        
        # Handle sigma (can be None)
        sigma = os.environ.get("SOM_SIGMA", "")
        self.som_sigma = float(sigma) if sigma.strip() else None
        
        # Handle random state (can be None)
        rand_state = os.environ.get("SOM_RANDOM_STATE", "42")
        self.som_random_state = int(rand_state) if rand_state.strip() else None
        
        self.som_run_name = os.environ.get("SOM_RUN_NAME", "som_model")
        
        # Handle boolean env var
        verbose = os.environ.get("SOM_VERBOSE", "true").lower()
        self.som_verbose = verbose in ("true", "1", "t", "yes", "y")
        
        # Load settings from .env file if it exists
        self._load_from_env_file()
        
        logger.info(f"Settings loaded. Metric key: {self.metric_key}, Ascending: {self.metric_ascending}")
    
    def _load_from_env_file(self):
        """Load settings from .env file if available."""
        env_file = ".env"
        if os.path.exists(env_file):
            logger.info(f"Loading settings from {env_file}")
            with open(env_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        key, value = line.split("=", 1)
                        # Remove comments from value
                        if "#" in value:
                            value = value.split("#", 1)[0].strip()
                        
                        # Set if not already set by environment variable
                        if os.environ.get(key) is None:
                            os.environ[key] = value
                            
    def as_dict(self):
        """Convert settings to a dictionary for logging."""
        return {
            "mlflow_tracking_uri": self.mlflow_tracking_uri,
            "mlflow_experiment_name": self.mlflow_experiment_name,
            "metric_key": self.metric_key,
            "metric_ascending": self.metric_ascending,
            "som_width": self.som_width,
            "som_height": self.som_height,
            "som_input_dim": self.som_input_dim,
            "som_iterations": self.som_iterations,
            "som_samples": self.som_samples,
            "som_learning_rate": self.som_learning_rate,
            "som_batch_size": self.som_batch_size,
            "som_sigma": self.som_sigma,
            "som_random_state": self.som_random_state,
            "som_run_name": self.som_run_name,
            "som_verbose": self.som_verbose
        }
    
    def __str__(self):
        """Return a string representation of the settings."""
        return json.dumps(self.as_dict(), indent=2)


# Create global settings instance
settings = Settings()

# Configure MLflow with the tracking URI
if settings.mlflow_tracking_uri:
    logger.info(f"Setting MLflow tracking URI to: {settings.mlflow_tracking_uri}")
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
else:
    logger.warning("No MLflow tracking URI provided in settings")


def get_model_params_from_settings() -> Dict[str, Any]:
    """
    Get model parameters from settings.
    
    Returns:
    --------
    Dict[str, Any]
        Dictionary of model parameters
    """
    return {
        "width": settings.som_width,
        "height": settings.som_height,
        "input_dim": settings.som_input_dim,
        "learning_rate": settings.som_learning_rate,
        "sigma": settings.som_sigma,
        "random_state": settings.som_random_state
    }


def get_training_params_from_settings() -> Dict[str, Any]:
    """
    Get training parameters from settings.
    
    Returns:
    --------
    Dict[str, Any]
        Dictionary of training parameters
    """
    return {
        "n_iterations": settings.som_iterations,
        "n_samples": settings.som_samples,
        "batch_size": settings.som_batch_size
    } 