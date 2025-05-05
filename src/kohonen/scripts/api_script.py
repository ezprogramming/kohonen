#!/usr/bin/env python3
"""
Script to start the API service with a trained model.
"""
import os
import time
import sys
import logging
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def start_api_server():
    """Start the API server with a trained model."""
    # Wait for run_id.txt to be available
    run_id_file = os.environ.get("SOM_RUN_ID_FILE", "/app/mlflow_data/run_id.txt")
    logger.info(f"Checking for run ID in file: {run_id_file}")
    
    max_attempts = 12  # Wait for up to 60 seconds
    attempts = 0
    
    while not os.path.exists(run_id_file) and attempts < max_attempts:
        logger.info("Waiting for model training to complete...")
        time.sleep(5)
        attempts += 1
    
    if not os.path.exists(run_id_file):
        logger.error(f"ERROR: Run ID file {run_id_file} not found after waiting. Exiting.")
        sys.exit(1)
    
    # Read the run ID
    with open(run_id_file, "r") as f:
        run_id = f.read().strip()
    
    # Validate run ID
    if not run_id or '%' in run_id:  # Check for invalid characters
        logger.warning(f"Invalid run ID found in file: '{run_id}'. Cleaning up...")
        run_id = run_id.strip('%')
    
    logger.info(f"Using model run ID: {run_id}")
    
    # Set environment variable for the API
    os.environ["SOM_RUN_ID"] = run_id
    logger.info(f"Starting API server with model run ID: {run_id}")
    
    # Start the API server
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    logger.info(f"Starting API server on {host}:{port}")
    uvicorn.run("kohonen.api:app", host=host, port=port)

if __name__ == "__main__":
    start_api_server() 