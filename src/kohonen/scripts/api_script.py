#!/usr/bin/env python3
"""
Script to start the API service with a trained model.
"""
import os
import time
import sys
import logging
import argparse
import uvicorn
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args(args=None):
    """
    Parse command line arguments.
    
    Parameters:
    -----------
    args : list, optional
        List of command line arguments for testing.
        If None, uses sys.argv.
    """
    parser = argparse.ArgumentParser(description="Start the API server with a trained model.")
    
    parser.add_argument(
        "--run-id", 
        type=str, 
        default=None, 
        help="Specific MLflow run ID to use. If not provided, the best model will be selected."
    )
    parser.add_argument(
        "--experiment-name", 
        type=str, 
        default=None, 
        help="MLflow experiment name to use. If not provided, uses value from settings."
    )
    parser.add_argument(
        "--metric", 
        type=str, 
        default=None, 
        help="Metric to use for model selection. If not provided, uses value from settings."
    )
    parser.add_argument(
        "--ascending", 
        action="store_true", 
        default=None,
        help="Whether to minimize (True) or maximize (False) the metric. If not provided, uses value from settings."
    )
    parser.add_argument(
        "--force-best", 
        action="store_true", 
        help="Force using the best model even if a run ID is available."
    )
    parser.add_argument(
        "--host", 
        type=str, 
        default="0.0.0.0", 
        help="Host to bind the API server to."
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=None, 
        help="Port to bind the API server to."
    )
    
    return parser.parse_args(args)


def start_api_server(args):
    """
    Start the API server with a trained model.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command line arguments
    """
    # Check if we should force using the best model
    if args.force_best:
        logger.info("Force-best flag set: Will use best model selection regardless of run ID availability")
        run_id = None
    else:
        # Check for run_id in arguments first
        run_id = args.run_id
        
        # If no run_id provided, check the file
        if run_id is None:
            run_id_file = os.environ.get("SOM_RUN_ID_FILE", "/app/mlflow_data/run_id.txt")
            logger.info(f"Checking for run ID in file: {run_id_file}")
            
            if os.path.exists(run_id_file):
                # Read the run ID from file
                with open(run_id_file, "r") as f:
                    run_id = f.read().strip()
                
                # Validate run ID - remove any trailing % or whitespace
                if run_id:
                    run_id = run_id.strip().rstrip('%')
                    
                    if run_id:
                        logger.info(f"Using specific model run ID from file: {run_id}")
                    else:
                        logger.info("Empty run ID found in file. Will use best model selection.")
                else:
                    logger.info("Empty run ID found in file. Will use best model selection.")
            else:
                logger.info("No run ID file found. Will use best model selection.")
    
    # If we have a specific run ID, use it
    if run_id:
        os.environ["SOM_RUN_ID"] = run_id
        logger.info(f"Setting SOM_RUN_ID environment variable to: {run_id}")
    else:
        # No specific run ID, we'll use model selection to find the best one
        logger.info("No specific run ID provided. Will use automatic model selection to find best model.")
        
        # Set experiment name and metric config if provided
        if args.experiment_name:
            os.environ["MLFLOW_EXPERIMENT_NAME"] = args.experiment_name
            logger.info(f"Setting experiment name: {args.experiment_name}")
            
        if args.metric:
            os.environ["SOM_METRIC_KEY"] = args.metric
            logger.info(f"Setting metric key: {args.metric}")
            
        if args.ascending is not None:
            os.environ["SOM_METRIC_ASCENDING"] = str(args.ascending).lower()
            logger.info(f"Setting metric ascending: {args.ascending}")
        
        # Unset SOM_RUN_ID to ensure model selection is used
        if "SOM_RUN_ID" in os.environ:
            del os.environ["SOM_RUN_ID"]
            logger.info("Unsetting SOM_RUN_ID to enable automatic model selection")
    
    # Start the API server
    port = args.port or int(os.environ.get("PORT", 8000))
    host = args.host or os.environ.get("HOST", "0.0.0.0")
    logger.info(f"Starting API server on {host}:{port}")
    uvicorn.run("kohonen.api:app", host=host, port=port)


if __name__ == "__main__":
    args = parse_args()
    start_api_server(args) 