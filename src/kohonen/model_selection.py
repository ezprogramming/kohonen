"""
Model selection utilities for finding the best SOM model.

This module provides functionality to select the optimal SOM model
from MLflow based on specified metrics.
"""
import os
import logging
from typing import Dict, List, Optional, Union, Any, Tuple

import mlflow
import pandas as pd
import numpy as np
from mlflow.entities import Run
from mlflow.tracking import MlflowClient

from .config import settings
from .som import SelfOrganizingMap
from .mlflow_utils import load_som_model

# Configure logging
logger = logging.getLogger(__name__)


def get_runs_for_experiment(
    experiment_name: Optional[str] = None,
    max_results: int = 100
) -> List[Run]:
    """
    Get all runs for a specified experiment.
    
    Parameters:
    -----------
    experiment_name : Optional[str]
        Name of the experiment to retrieve runs from.
        If None, uses default from settings.
    max_results : int
        Maximum number of runs to retrieve
        
    Returns:
    --------
    List[Run]
        List of MLflow runs
    """
    # Use provided experiment name or default from settings
    experiment_name = experiment_name or settings.mlflow_experiment_name
    
    # Set up MLflow tracking
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    
    # Get experiment ID
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        logger.warning(f"Experiment '{experiment_name}' not found. No models available.")
        return []
    
    # Get all runs for this experiment
    client = MlflowClient()
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        max_results=max_results,
        order_by=["metrics.quantization_error ASC"]
    )
    
    logger.info(f"Found {len(runs)} runs for experiment '{experiment_name}'")
    return runs


def find_best_run(
    experiment_name: Optional[str] = None,
    metric_key: Optional[str] = None,
    ascending: Optional[bool] = None,
    min_training_iterations: int = 50,
    run_status: str = "FINISHED"
) -> Optional[str]:
    """
    Find the best run based on the specified metric.
    
    Parameters:
    -----------
    experiment_name : Optional[str]
        Name of the experiment to search in.
        If None, uses default from settings.
    metric_key : Optional[str]
        Metric to optimize. If None, uses default from settings.
    ascending : Optional[bool]
        Whether to minimize (True) or maximize (False) the metric.
        If None, uses default from settings.
    min_training_iterations : int
        Minimum number of training iterations a model must have completed
    run_status : str
        Filter runs by status (e.g., "FINISHED", "RUNNING")
        
    Returns:
    --------
    Optional[str]
        Run ID of the best model, or None if no suitable models found
    """
    # Use defaults from settings if not specified
    metric_key = metric_key or settings.metric_key
    ascending = ascending if ascending is not None else settings.metric_ascending
    
    # Get all runs
    runs = get_runs_for_experiment(experiment_name)
    
    if not runs:
        logger.warning("No runs found for the experiment")
        return None
    
    # Filter runs by status and metrics availability
    valid_runs = []
    for run in runs:
        if run.info.status != run_status:
            continue
            
        # Check if the run has the required metric
        if metric_key not in run.data.metrics:
            continue
            
        # Check minimum training iterations via parameters
        params = run.data.params
        if "n_iterations" in params and int(params["n_iterations"]) < min_training_iterations:
            continue
            
        valid_runs.append(run)
    
    if not valid_runs:
        logger.warning(f"No valid runs found with metric '{metric_key}' and status '{run_status}'")
        return None
    
    # Sort runs by the metric
    sorted_runs = sorted(
        valid_runs,
        key=lambda r: r.data.metrics[metric_key],
        reverse=not ascending  # If ascending=True (minimize), reverse=False
    )
    
    best_run = sorted_runs[0]
    best_metric_value = best_run.data.metrics[metric_key]
    best_run_id = best_run.info.run_id
    
    logger.info(f"Found best model run_id={best_run_id} with {metric_key}={best_metric_value}")
    return best_run_id


def load_best_model(
    experiment_name: Optional[str] = None,
    metric_key: Optional[str] = None,
    ascending: Optional[bool] = None
) -> Optional[SelfOrganizingMap]:
    """
    Load the best SOM model based on the specified metric.
    
    Parameters:
    -----------
    experiment_name : Optional[str]
        Name of the experiment to search in.
        If None, uses default from settings.
    metric_key : Optional[str]
        Metric to optimize. If None, uses default from settings.
    ascending : Optional[bool]
        Whether to minimize (True) or maximize (False) the metric.
        If None, uses default from settings.
        
    Returns:
    --------
    Optional[SelfOrganizingMap]
        The best SOM model, or None if no suitable model found
    """
    best_run_id = find_best_run(experiment_name, metric_key, ascending)
    
    if not best_run_id:
        logger.warning("Could not find a suitable model to load")
        return None
    
    try:
        logger.info(f"Loading best model with run_id={best_run_id}")
        model = load_som_model(best_run_id)
        return model
    except Exception as e:
        logger.error(f"Error loading best model: {e}")
        return None


def get_model_metrics_summary(
    experiment_name: Optional[str] = None,
    metric_keys: Optional[List[str]] = None,
    max_results: int = 10
) -> pd.DataFrame:
    """
    Get a summary of model metrics for all runs.
    
    Parameters:
    -----------
    experiment_name : Optional[str]
        Name of the experiment to search in.
        If None, uses default from settings.
    metric_keys : Optional[List[str]]
        List of metrics to include in the summary.
        If None, includes all available metrics.
    max_results : int
        Maximum number of runs to include
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with run metrics and parameters
    """
    # Get all runs
    runs = get_runs_for_experiment(experiment_name, max_results=max_results)
    
    if not runs:
        return pd.DataFrame()
    
    # Extract run information
    runs_data = []
    for run in runs:
        run_data = {
            "run_id": run.info.run_id,
            "status": run.info.status,
            "start_time": pd.to_datetime(run.info.start_time, unit="ms"),
            "end_time": pd.to_datetime(run.info.end_time, unit="ms")
        }
        
        # Add metrics
        if metric_keys:
            for key in metric_keys:
                run_data[f"metric_{key}"] = run.data.metrics.get(key, np.nan)
        else:
            for key, value in run.data.metrics.items():
                run_data[f"metric_{key}"] = value
        
        # Add parameters
        for key, value in run.data.params.items():
            run_data[f"param_{key}"] = value
            
        runs_data.append(run_data)
    
    # Create DataFrame
    df = pd.DataFrame(runs_data)
    
    # Add run duration
    if "start_time" in df.columns and "end_time" in df.columns:
        df["duration_seconds"] = (df["end_time"] - df["start_time"]).dt.total_seconds()
    
    return df 