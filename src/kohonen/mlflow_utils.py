"""
MLflow utilities for logging and loading SOM models.
"""
from __future__ import annotations
import os
import logging
import time
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
import tempfile
import matplotlib.pyplot as plt
import mlflow
import pandas as pd  # Add pandas for DataFrame handling
from mlflow.models import ModelSignature, infer_signature
from mlflow.types import Schema, ColSpec, TensorSpec
from mlflow.pyfunc import PythonModel, PythonModelContext

try:
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from .som import SelfOrganizingMap
from .visualization import visualize_som_grid, visualize_component_planes

# Set up logging based on environment variable
log_level = os.environ.get("LOG_LEVEL", "INFO")
logging.basicConfig(level=getattr(logging, log_level))
logger = logging.getLogger(__name__)


def get_active_experiment_id() -> str:
    """Get the ID of the active MLflow experiment.
    
    Returns:
        str: The ID of the active experiment
    """
    # Get active experiment
    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "som-experiments")
    experiment = mlflow.get_experiment_by_name(experiment_name)
    
    # If not found or deleted, try to get latest experiment with similar name
    if not experiment or experiment.lifecycle_stage == "deleted":
        # Find experiments starting with the base name
        experiments = mlflow.search_experiments()
        active_experiments = [exp for exp in experiments if exp.name.startswith(experiment_name) and exp.lifecycle_stage == "active"]
        
        if active_experiments:
            # Sort by name to get the latest one (assuming timestamp in name)
            active_experiments.sort(key=lambda x: x.name, reverse=True)
            experiment = active_experiments[0]
            mlflow.set_experiment(experiment.name)
            logger.info(f"Using existing experiment: {experiment.name} (ID: {experiment.experiment_id})")
        else:
            # Create a new experiment as a fallback
            timestamp = int(time.time())
            new_name = f"{experiment_name}-{timestamp}"
            experiment_id = mlflow.create_experiment(new_name)
            mlflow.set_experiment(new_name)
            logger.info(f"Created new experiment: {new_name} (ID: {experiment_id})")
            return experiment_id
    
    return experiment.experiment_id


def log_som_model(
    model: SelfOrganizingMap,
    input_data: np.ndarray,
    params: Optional[Dict[str, Any]] = None,
    training_metrics: Optional[Dict[str, List[float]]] = None,
    run_name: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
    component_names: Optional[List[str]] = None,
    save_visualizations: bool = True,
) -> str:
    """
    Log a SOM model to MLflow.
    
    Parameters:
    -----------
    model : SelfOrganizingMap
        The trained SOM model to log
    input_data : numpy.ndarray
        Sample input data used for training (used for visualizations)
    params : Dict[str, Any], optional
        Model parameters to log
    training_metrics : Dict[str, List[float]], optional
        Training metrics to log (e.g., quantization errors)
    run_name : str, optional
        Name for the MLflow run
    tags : Dict[str, str], optional
        Tags to add to the MLflow run
    component_names : List[str], optional
        Names for the input dimensions (for component plane visualization)
    save_visualizations : bool, default=True
        Whether to save visualizations as artifacts
        
    Returns:
    --------
    str
        The ID of the MLflow run
    """
    if not MLFLOW_AVAILABLE:
        logger.warning("MLflow is not available, model logging skipped")
        return "mlflow-not-available"
    
    # Get active experiment ID
    experiment_id = get_active_experiment_id()
    
    # Start a new run
    # First set the tracking URI from environment variable if available
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "mlflow_data")
    mlflow.set_tracking_uri(tracking_uri)
    
    # Get default experiment name from environment variable or use default
    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "som-experiments")
    
    # Get or create experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id
    
    # Start a new run
    with mlflow.start_run(run_name=run_name, experiment_id=experiment_id) as run:
        # Log parameters
        if params:
            mlflow.log_params(params)
        
        # Log model info
        model_info = {
            "width": model.width,
            "height": model.height,
            "input_dim": model.input_dim,
            "initial_learning_rate": model.learning_rate_0,
            "initial_sigma": model.sigma_0,
        }
        mlflow.log_params(model_info)
        
        # Log tags
        if tags:
            for key, value in tags.items():
                mlflow.set_tag(key, value)
        
        # Log metrics
        if training_metrics:
            if "quantization_errors" in training_metrics:
                for i, qe in enumerate(training_metrics["quantization_errors"]):
                    mlflow.log_metric("quantization_error", qe, step=i)
            
            if "topographic_errors" in training_metrics:
                for i, te in enumerate(training_metrics["topographic_errors"]):
                    mlflow.log_metric("topographic_error", te, step=i)
        
        # Save visualizations if requested
        if save_visualizations:
            # Create visualizations in a temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                # Get the weights for visualization
                weights = model.get_weights()
                
                # Unified weight grid visualization
                fig_weights = visualize_som_grid(weights)
                weights_path = os.path.join(temp_dir, "weights_grid.png")
                fig_weights.savefig(weights_path)
                plt.close(fig_weights)
                mlflow.log_artifact(weights_path, "visualizations")
                
                # Component planes visualization
                if component_names is None:
                    component_names = [f"Component {i+1}" for i in range(model.input_dim)]
                
                fig_components = visualize_component_planes(weights, component_names)
                components_path = os.path.join(temp_dir, "component_planes.png")
                fig_components.savefig(components_path)
                plt.close(fig_components)
                mlflow.log_artifact(components_path, "visualizations")
                
                # If we have input data, visualize BMU heatmap
                if input_data is not None and input_data.shape[0] > 0:
                    try:
                        # Create BMU heatmap
                        fig_heatmap = plt.figure(figsize=(10, 8))
                        ax = fig_heatmap.add_subplot(111)
                        
                        # Get BMUs for all input data points
                        bmu_counts = np.zeros((model.width, model.height))
                        for vec in input_data:
                            bmu_x, bmu_y = model.predict_bmu(vec)
                            bmu_counts[bmu_x, bmu_y] += 1
                        
                        # Plot heatmap
                        im = ax.imshow(bmu_counts.T, cmap='viridis', interpolation='nearest')
                        ax.set_title("BMU Activation Heatmap")
                        ax.set_xlabel("X")
                        ax.set_ylabel("Y")
                        plt.colorbar(im, ax=ax, label="Number of activations")
                        
                        # Save heatmap
                        heatmap_path = os.path.join(temp_dir, "bmu_heatmap.png")
                        fig_heatmap.savefig(heatmap_path)
                        plt.close(fig_heatmap)
                        mlflow.log_artifact(heatmap_path, "visualizations")
                    except Exception as e:
                        logger.warning(f"Could not create BMU heatmap: {e}")
        
        # Save model weights as numpy array
        with tempfile.TemporaryDirectory() as temp_dir:
            models_dir = os.path.join(temp_dir, "models")
            os.makedirs(models_dir, exist_ok=True)
            weights_path = os.path.join(models_dir, "weights.npy")
            np.save(weights_path, model.weights)
            
            # Log the weights file
            mlflow.log_artifact(weights_path, "models")
        
        # Log model signature
        try:
            from mlflow.models.signature import ModelSignature
            from mlflow.types import Schema, TensorSpec
            
            # Define input and output schema
            input_schema = Schema([
                TensorSpec(np.dtype(np.float32), (-1, model.input_dim), name="inputs")
            ])
            output_schema = Schema([
                TensorSpec(np.dtype(np.int32), (-1, 2), name="bmu_coordinates")
            ])
            
            signature = ModelSignature(inputs=input_schema, outputs=output_schema)
            
            # Create a simple model wrapper
            class SOMWrapper(mlflow.pyfunc.PythonModel):
                def __init__(self, som_model):
                    self.som_model = som_model
                
                def predict(
                    self,
                    context: PythonModelContext,
                    model_input: "pd.DataFrame",
                ) -> "pd.DataFrame":
                    """Predict BMU coordinates for input vectors.
                    
                    Args:
                        context: MLflow model context
                        model_input: Input data as a pandas DataFrame. Each row is an input vector.
                    Returns
                    -------
                    pd.DataFrame
                        DataFrame with columns ["bmu_x", "bmu_y"].
                    """
                    import numpy as np
                    import pandas as pd
                    # Ensure input is a DataFrame
                    if not isinstance(model_input, pd.DataFrame):
                        model_input = pd.DataFrame(model_input)
                    vectors = model_input.values.astype(np.float32)
                    # Get BMUs
                    results = [self.som_model.predict_bmu(vec) for vec in vectors]
                    return pd.DataFrame(results, columns=["bmu_x", "bmu_y"])
            
            # Log the model
            try:
                # Ensure data types match exactly what MLflow expects
                sample_input = input_data[:1].astype(np.float32)
                
                # First try to log with inferred signature
                logger.info("Logging model with inferred signature")
                logger.info(f"Sample input shape: {sample_input.shape}, dtype: {sample_input.dtype}")
                
                # Create example predictions for signature inference (shape: [n_samples, 2])
                sample_output = np.array([model.predict_bmu(vec) for vec in sample_input], dtype=np.int32)
                logger.info(f"Sample output shape: {sample_output.shape}, dtype: {sample_output.dtype}")
                
                # Try logging with inferred signature first
                mlflow.pyfunc.log_model(
                    artifact_path="som_model",
                    python_model=SOMWrapper(model),
                    artifacts=None,
                    conda_env=None,
                    code_path=None,
                    signature=infer_signature(sample_input, sample_output),
                    input_example=sample_input,
                )
            except Exception as e:
                logger.warning(f"Could not log model with inferred signature: {e}")
                
                try:
                    # Try with explicit signature and dict input format
                    logger.info("Trying with explicit signature and dict input")
                    
                    input_schema = Schema([
                        TensorSpec(np.dtype(np.float32), (-1, model.input_dim), name="inputs")
                    ])
                    output_schema = Schema([
                        TensorSpec(np.dtype(np.int32), (-1, 2), name="bmu_coordinates")
                    ])
                    
                    signature = ModelSignature(inputs=input_schema, outputs=output_schema)
                    
                    # Keep as numpy array, no conversion
                    dict_input = {"inputs": input_data[:1].astype(np.float32)}
                    
                    mlflow.pyfunc.log_model(
                        artifact_path="som_model",
                        python_model=SOMWrapper(model),
                        signature=signature,
                        input_example=dict_input,
                    )
                except Exception as e:
                    logger.warning(f"Could not log model with explicit signature: {e}")
                    
                    # Final fallback: log without signature or example
                    logger.info("Logging model without signature as fallback")
                    mlflow.pyfunc.log_model(
                        artifact_path="som_model",
                        python_model=SOMWrapper(model),
                    )
            
        except Exception as e:
            logger.warning(f"Could not log model with signature: {e}")
        
        return run.info.run_id


def load_som_model(
    run_id: str,
    artifact_path: str = "models"
) -> SelfOrganizingMap:
    """
    Load a SOM model from MLflow.
    
    Parameters:
    -----------
    run_id : str
        The ID of the MLflow run containing the model
    artifact_path : str
        Path within the run's artifact directory where the model is stored
        
    Returns:
    --------
    SelfOrganizingMap
        The loaded SOM model
    """
    if not MLFLOW_AVAILABLE:
        logger.warning("MLflow is not available, cannot load model")
        raise ImportError("MLflow is required to load models")
    
    try:
        # Set tracking URI from environment variable
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "mlflow_data")
        mlflow.set_tracking_uri(tracking_uri)
        
        # Try to download artifacts using the MLflow client
        local_dir = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path=artifact_path
        )
        
        # Load weights
        weights_path = os.path.join(local_dir, "weights.npy")
        weights = np.load(weights_path)
        
    except (OSError, FileNotFoundError) as e:
        logger.error(f"Error downloading artifacts: {e}")
        
        # Fallback: Try to access artifacts directly if we're in Docker environment
        # Check for environment variable indicating we're using local file paths
        artifact_root = os.environ.get("MLFLOW_ARTIFACT_ROOT")
        if artifact_root:
            logger.info(f"Trying direct artifact access at {artifact_root}")
            experiment_id = mlflow.get_run(run_id).info.experiment_id
            artifact_location = os.path.join(artifact_root, experiment_id, run_id, "artifacts", artifact_path)
            
            if os.path.exists(artifact_location):
                weights_path = os.path.join(artifact_location, "weights.npy")
                weights = np.load(weights_path)
            else:
                raise FileNotFoundError(f"Could not find artifacts at {artifact_location}")
        else:
            raise
    
    # Create model with the same dimensions
    width, height, input_dim = weights.shape
    model = SelfOrganizingMap(width, height, input_dim)
    
    # Set weights
    model.weights = weights
    
    return model 