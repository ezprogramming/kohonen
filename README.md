# Kohonen Self-Organizing Map

A high-performance implementation of the Kohonen Self-Organizing Map (SOM) algorithm, optimized for production use.

## Features

- ⚡ **Vectorized Implementation**: 30-50x faster than naive triple-loop implementations using NumPy broadcasting with memory-efficient batch processing
- 📦 **Production-Ready**: Clean, tested package with proper structure and documentation
- 📈 **MLflow Integration**: Track experiments, metrics, and artifacts with MLflow
- 🔍 **Model Selection**: Automatic selection of the best model based on configurable metrics
- 🐳 **Containerized**: Ready-to-use Docker and docker-compose configurations
- 🌐 **API Service**: FastAPI endpoint for serving trained models

## Installation

Install the base package:

```bash
pip install .
```

Install with all optional dependencies:

```bash
pip install ".[all]"
```

Or select just what you need:

```bash
pip install ".[mlflow]"  # For MLflow integration
pip install ".[api]"     # For FastAPI service
pip install ".[dev]"     # For development tools (pytest, ruff, mypy)
```

## Quick Start

### Basic SOM Usage

```python
import numpy as np
from kohonen import SelfOrganizingMap

# Create a SOM with configurable dimensions
# (these values can be customized via .env file when using Docker)
width, height = 50, 50  # Default project config is now 50x50 via .env
input_dim = 3
som = SelfOrganizingMap(width=width, height=height, input_dim=input_dim)

# Generate random training data
samples = 1000  # Configurable via SOM_SAMPLES in .env
data = np.random.rand(samples, input_dim)

# Train the SOM
iterations = 300  # Configurable via SOM_ITERATIONS in .env
# For smaller datasets, use fully vectorized approach (batch_size=None)
# For larger datasets, use batch processing to control memory usage
batch_size = None  # Set to a numeric value like 100 for large datasets
metrics = som.train(data, n_iterations=iterations, verbose=True, batch_size=batch_size)

# Get the trained weights
weights = som.get_weights()

# Find the Best Matching Unit (BMU) for a new data point
new_point = np.array([0.2, 0.5, 0.8])
bmu_x, bmu_y = som.predict_bmu(new_point)
print(f"BMU coordinates: ({bmu_x}, {bmu_y})")
```

### Memory-Efficient Training with Batch Processing

For large datasets, you can use batch processing to control memory usage:

```python
# For a dataset with 100,000 samples
large_data = np.random.rand(100000, input_dim)

# Use batch processing with a batch size of 500
# This reduces memory usage during training
metrics = som.train(large_data, n_iterations=iterations, verbose=True, batch_size=500)
```

### Visualization

```python
import matplotlib.pyplot as plt
from kohonen.visualization import plot_som_grid, plot_component_planes

# Visualize the trained SOM
fig = plot_som_grid(weights)
plt.savefig("som_grid.png")

# Visualize component planes
component_names = ["Feature 1", "Feature 2", "Feature 3"]
fig = plot_component_planes(weights, component_names=component_names)
plt.savefig("component_planes.png")
```

### MLflow Integration

```python
from kohonen.mlflow_utils import log_som_model, load_som_model

# Log a trained model to MLflow
run_id = log_som_model(
    model=som,
    input_data=data,
    params={
        "width": width,
        "height": height, 
        "input_dim": input_dim,
        "iterations": iterations,
        "samples": samples
    },
    training_metrics=metrics,
    run_name="my_som_model",
    save_visualizations=True
)

# Later, load the model from MLflow
loaded_som = load_som_model(run_id)
```

#### Inspecting MLflow Experiments

You can inspect your MLflow experiments and runs using the provided `inspect_mlflow.py` script:

```python
# Inspect all experiments and runs
python -m kohonen.scripts.inspect_mlflow

# Inspect a specific run with detailed information
python -m kohonen.scripts.inspect_mlflow --run-id <run_id>

# Specify a custom MLflow data directory
python -m kohonen.scripts.inspect_mlflow --mlflow-dir path/to/mlflow_data
```

Or use the Makefile shortcut:

```bash
# Inspect all experiments
make inspect

# Inspect a specific run
make inspect run_id=<run_id>
```

This tool helps you understand your experiment structure, view run parameters, metrics, and artifacts without needing to use the MLflow UI.

### API Service

Start the API service:

```bash
uvicorn kohonen.api:app --host 0.0.0.0 --port 8000
```

Make predictions:

```python
import requests
import numpy as np

# Single prediction
response = requests.post(
    "http://localhost:8000/predict-bmu",
    json={"data": [0.2, 0.5, 0.8].tolist()}
)
print(response.json())  # {"bmu_x": 15, "bmu_y": 20}

# Batch prediction
response = requests.post(
    "http://localhost:8000/predict-batch",
    json={"data": np.random.rand(5, 3).tolist()}
)
print(response.json())  # {"results": [{"bmu_x": 10, "bmu_y": 15}, ...]}
```

### API Service with Model Selection

The API service can automatically select the best model based on metrics:

```bash
# Start API with automatic model selection (automatically starts MLflow if not running)
make api

# Start API with specific run ID (from command line)
python -m kohonen.scripts.api_script --run-id <run_id>

# Start API with custom metric selection
python -m kohonen.scripts.api_script --metric quantization_error --ascending

# Force using the best model even if a run ID is available
python -m kohonen.scripts.api_script --force-best
```

Make predictions:

```python
import requests
import numpy as np

# Get info about the selected model
response = requests.get("http://localhost:8000/model-info")
print(response.json())

# List available models
response = requests.get("http://localhost:8000/models?max_results=5")
print(response.json())

# Make a prediction
response = requests.post(
    "http://localhost:8000/predict-bmu",
    json={"data": [0.2, 0.5, 0.8]}
)
print(response.json())  # {"bmu_x": 15, "bmu_y": 20}
```

## Docker Deployment

The recommended way to deploy the SOM is using Docker Compose with an `.env` file for configuration.

### Using Docker Compose (Recommended)

1. Create a `.env` file in the project root or use the provided example:

```
# SOM Configuration
SOM_WIDTH=50
SOM_HEIGHT=50
SOM_INPUT_DIM=3
SOM_ITERATIONS=300
SOM_SAMPLES=1000
SOM_LEARNING_RATE=0.1
SOM_BATCH_SIZE=
SOM_SIGMA=
SOM_RANDOM_STATE=42
SOM_RUN_NAME=som_model
SOM_VERBOSE=true

# MLflow Configuration
MLFLOW_TRACKING_URI=http://mlflow:5000
RUN_ID_FILE=/app/mlflow_data/run_id.txt

# API Configuration
PORT=8000
SOM_RUN_ID_FILE=/app/mlflow_data/run_id.txt

# Model Selection Configuration
# Metric used to select the best model
SOM_METRIC_KEY=quantization_error
# True -> minimize, False -> maximize  
SOM_METRIC_ASCENDING=true
# Force using best model regardless of run ID (default false)
SOM_FORCE_BEST=false
```

2. Start the Docker services:

```bash
# Build the Docker images (required for first run or after code changes)
make build

# Start all services (MLflow, training, and API)
make up

# Access the MLflow UI at http://localhost:5050
# Access the API at http://localhost:8000
```

## Model Selection & Configuration

The API service can be configured to use either a specific model or automatically select the best model based on performance metrics.

### Understanding Model Selection

The model selection process follows these rules:

1. When `SOM_FORCE_BEST=true`:
   - The API will ignore run_id.txt and always select the best model based on metrics
   - The metric used for selection is defined by `SOM_METRIC_KEY` (default: quantization_error)
   - Whether to minimize or maximize the metric is set by `SOM_METRIC_ASCENDING` (default: true, meaning lower values are better)

2. When `SOM_FORCE_BEST=false` (default):
   - The API first checks for a run_id passed as a command-line argument
   - If no argument is provided, it reads the run_id from run_id.txt
   - If a valid run ID is found, it uses that specific model
   - If no valid run ID is found, it falls back to selecting the best model

### Training and Model Updates

When you run training, the system automatically:
1. Trains a new SOM model with the parameters in .env
2. Logs it to MLflow for tracking
3. Writes the new run ID to run_id.txt
4. The API service (if started after training) will use this newly trained model

### Configuring Model Selection

You can use these Makefile commands to manage model selection:

```bash
# Configure API to always use the best model (ignore run_id.txt)
make use-best

# Configure API to use a specific run ID
make use-specific run_id=YOUR_RUN_ID
```

Or modify environment variables directly:

```bash
# In .env file
SOM_FORCE_BEST=true    # Always use best model
SOM_FORCE_BEST=false   # Use run_id.txt if available
```

### Inspecting Available Models

To see available models:

```bash
# View all runs in MLflow
make inspect

# View details of a specific run
make inspect run_id=YOUR_RUN_ID

# Check which model the API is currently using
curl http://localhost:8000/model-info
```

### Retraining & Docker Rebuild Guidelines

For different types of changes:

1. **Code changes in src/kohonen/**: You need to rebuild the Docker image
   ```bash
   make build
   make up
   ```

2. **Changes to environment variables in docker-compose.yml or .env**: No rebuild needed, just restart
   ```bash
   make down
   make up
   ```

3. **To train a new model with current settings**:
   ```bash
   make train
   ```
   This will automatically write the new run ID to run_id.txt

## Development and Testing

To run tests locally:

```bash
# Run all tests
python -m kohonen.scripts.test_script

# Activate Python linting
ruff src/kohonen/

# Check type annotations
mypy src/kohonen/
```

## License

This project is released under the MIT License. See the [LICENSE](LICENSE) file for details. 