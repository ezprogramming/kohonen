#!/usr/bin/env python3
"""
MLflow Data Inspector for Kohonen SOM

This script provides a way to inspect MLflow data and print information about stored SOM models.
It helps with debugging, monitoring experiments, and finding specific run IDs without using the
MLflow UI. The script can:

1. List all experiments and their runs
2. Show detailed information about a specific run, including:
   - Parameters used for training
   - Final metrics achieved
   - Metadata tags
   - Artifacts stored (models, visualizations)

Usage:
    python -m kohonen.scripts.inspect_mlflow [--mlflow-dir DIR] [--run-id RUN_ID]

Via Makefile:
    make inspect [run_id=<run_id>]
"""
import os
import sys
import logging
import json
from pathlib import Path
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Inspect MLflow data and stored models")
    parser.add_argument("--mlflow-dir", type=str, default="./mlflow_data",
                      help="Path to the MLflow data directory")
    parser.add_argument("--run-id", type=str, default=None,
                      help="Run ID to inspect in detail (optional)")
    return parser.parse_args()

def inspect_mlflow_data(mlflow_dir: str, run_id: str = None):
    """
    Inspect the MLflow data directory and print information about experiments and runs.
    
    Parameters:
    -----------
    mlflow_dir : str
        Path to the MLflow data directory
    run_id : str, optional
        Specific run ID to inspect in detail
    
    Returns:
    --------
    int
        Exit code (0 for success, 1 for errors)
    """
    mlflow_path = Path(mlflow_dir)
    
    # Check if the directory exists
    if not mlflow_path.exists():
        logger.error(f"MLflow directory {mlflow_dir} does not exist")
        return 1
    
    logger.info(f"Inspecting MLflow data at: {mlflow_path.absolute()}")
    
    # Structure of MLflow data
    logger.info("\n== MLflow Data Structure ==")
    logger.info("- /mlflow_data/")
    logger.info("  ├── [experiment_id]/                 # Directory for each experiment")
    logger.info("  │   └── [run_id]/                    # Directory for each run within experiment")
    logger.info("  │       ├── artifacts/               # Model artifacts")
    logger.info("  │       ├── metrics/                 # Run metrics")
    logger.info("  │       ├── params/                  # Run parameters")
    logger.info("  │       └── tags/                    # Run metadata tags")
    logger.info("  ├── artifacts/                       # Central artifact store")
    logger.info("  │   └── [experiment_id]/             # Artifacts by experiment")
    logger.info("  │       └── [run_id]/                # Artifacts by run")
    logger.info("  │           └── artifacts/")
    logger.info("  │               ├── models/          # Stored models")
    logger.info("  │               └── visualizations/  # Visualizations")
    logger.info("  └── .trash/                          # Deleted experiments")
    
    # List experiments
    experiments = [p for p in mlflow_path.iterdir() 
                  if p.is_dir() and p.name not in ['.trash', 'artifacts']]
    
    logger.info(f"\nFound {len(experiments)} experiments:")
    for exp_dir in experiments:
        exp_id = exp_dir.name
        
        # Try to get experiment name
        try:
            with open(exp_dir / "meta.yaml", "r") as f:
                meta = f.read()
                exp_name = meta.split("name: ")[1].split("\n")[0].strip()
        except:
            exp_name = "Unknown"
        
        # Count runs
        runs = [p for p in exp_dir.iterdir() if p.is_dir()]
        logger.info(f"- Experiment: {exp_name} (ID: {exp_id}) - {len(runs)} runs")
        
        # List runs
        for run_dir in runs[:5]:  # Show only first 5 runs
            run_id = run_dir.name
            
            # Try to get run name
            try:
                with open(run_dir / "tags" / "mlflow.runName", "r") as f:
                    run_name = f.read()
            except:
                run_name = "Unknown"
            
            logger.info(f"  └── Run: {run_name} (ID: {run_id})")
    
    # Show detailed info for a specific run if requested
    if run_id:
        logger.info(f"\n== Detailed information for run {run_id} ==")
        
        # Find the run in experiments
        run_found = False
        for exp_dir in experiments:
            run_dir = exp_dir / run_id
            if run_dir.exists():
                run_found = True
                
                # Get experiment ID and name
                exp_id = exp_dir.name
                try:
                    with open(exp_dir / "meta.yaml", "r") as f:
                        meta = f.read()
                        exp_name = meta.split("name: ")[1].split("\n")[0].strip()
                except:
                    exp_name = "Unknown"
                
                logger.info(f"Found in experiment: {exp_name} (ID: {exp_id})")
                
                # Parse params
                params = {}
                params_dir = run_dir / "params"
                if params_dir.exists():
                    for param_file in params_dir.iterdir():
                        with open(param_file, "r") as f:
                            params[param_file.name] = f.read()
                
                logger.info(f"Parameters: {json.dumps(params, indent=2)}")
                
                # Parse metrics
                metrics = {}
                metrics_dir = run_dir / "metrics"
                if metrics_dir.exists():
                    for metric_file in metrics_dir.iterdir():
                        with open(metric_file, "r") as f:
                            lines = f.readlines()
                            values = [float(line.split()[1]) for line in lines]
                            metrics[metric_file.name] = values[-1]  # Last value
                
                logger.info(f"Metrics: {json.dumps(metrics, indent=2)}")
                
                # Parse tags
                tags = {}
                tags_dir = run_dir / "tags"
                if tags_dir.exists():
                    for tag_file in tags_dir.iterdir():
                        with open(tag_file, "r") as f:
                            tags[tag_file.name] = f.read()
                
                logger.info(f"Tags: {json.dumps(tags, indent=2)}")
                
                # List artifacts
                artifacts_dir = mlflow_path / "artifacts" / exp_id / run_id / "artifacts"
                if artifacts_dir.exists():
                    logger.info("\nArtifacts:")
                    for root, dirs, files in os.walk(artifacts_dir):
                        rel_path = os.path.relpath(root, artifacts_dir)
                        if rel_path != ".":
                            logger.info(f"- {rel_path}/")
                        for file in files:
                            full_path = os.path.join(root, file)
                            file_size = os.path.getsize(full_path)
                            file_size_str = f"{file_size / 1024:.1f} KB" if file_size < 1024*1024 else f"{file_size / (1024*1024):.1f} MB"
                            logger.info(f"  └── {file} ({file_size_str})")
        
        if not run_found:
            logger.error(f"Run ID {run_id} not found in any experiment")
    
    return 0

if __name__ == "__main__":
    args = parse_args()
    sys.exit(inspect_mlflow_data(args.mlflow_dir, args.run_id)) 