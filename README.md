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
# Start API with automatic model selection
python -m kohonen.scripts.api_script

# Start API with specific run ID
python -m kohonen.scripts.api_script --run-id <run_id>

# Start API with custom metric selection
python -m kohonen.scripts.api_script --metric quantization_error --ascending
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

1. Create a `.env` file in the project root:

```bash
# Create a basic configuration
cat > .env << EOL
# SOM Configuration Parameters
SOM_WIDTH=50
SOM_HEIGHT=50
SOM_INPUT_DIM=3
SOM_ITERATIONS=300
SOM_SAMPLES=1000
SOM_LEARNING_RATE=0.1
SOM_BATCH_SIZE=100     # Set to control memory usage (omit for fully vectorized mode)
SOM_RUN_NAME=som_model
SOM_VERBOSE=true
EOL
```

2. Build and start the services:

```bash
# Build Docker images with current configuration
docker compose build

# Start all services (MLflow, training, API)
docker compose up -d
```

3. Monitor the training progress:

```bash
docker compose logs -f train
```

4. When training completes, the API service will automatically load the trained model
   and serve it at http://localhost:8000.

> **Important**: After updating the `.env` file, you must rebuild the Docker images with
> `docker compose build` for the changes to take effect.

### Single Container (Advanced)

For standalone container deployment:

```bash
docker build -t kohonen-som .
docker run -p 8000:8000 -e SOM_WIDTH=50 -e SOM_HEIGHT=50 -e SOM_RUN_ID=your_run_id kohonen-som
```

## Development

Run tests:

```bash
pytest
```

Run linting and type checking:

```bash
ruff check src/
mypy src/
```

## Project Structure

The project follows a clean, production-ready structure:

```
kohonen/
├── src/                  # Source code
│   └── kohonen/          # Main package
│       ├── scripts/      # Command-line scripts
│       │   ├── train_script.py    # Training script
│       │   ├── api_script.py      # API server script
│       │   ├── test_script.py     # Test runner and unit tests
│       │   └── inspect_mlflow.py  # MLflow data inspector
│       ├── api.py        # FastAPI implementation
│       ├── mlflow_utils.py # MLflow integration
│       ├── som.py        # Core SOM implementation
│       └── visualization.py # Visualization utilities
├── mlflow_data/          # MLflow tracking data
│   ├── artifacts/        # Model artifacts
│   └── [experiment_id]/  # Experiment metadata
├── docker-compose.yml    # Docker Compose configuration
├── Dockerfile            # Docker image definition
├── Makefile              # Common commands
├── pyproject.toml        # Package configuration
└── README.md             # This file
```

## MLflow Data Structure

MLflow stores experiment data in a structured way:

```
mlflow_data/
├── [experiment_id]/                 # Directory for each experiment
│   └── [run_id]/                    # Directory for each run within experiment
│       ├── artifacts/               # Model artifacts
│       ├── metrics/                 # Run metrics
│       ├── params/                  # Run parameters
│       └── tags/                    # Run metadata tags
├── artifacts/                       # Central artifact store
│   └── [experiment_id]/             # Artifacts by experiment
│       └── [run_id]/                # Artifacts by run
│           └── artifacts/
│               ├── models/          # Stored models
│               └── visualizations/  # Visualizations
└── .trash/                          # Deleted experiments
```

Each run ID is a unique identifier for a trained model. The system stores:

1. **Model Artifacts**: The trained SOM model, saved in a serialized format
2. **Visualizations**: Generated plots of the SOM grid, component planes, etc.
3. **Metrics**: Performance metrics like quantization error
4. **Parameters**: Model hyperparameters (width, height, learning rate, etc.)
5. **Tags**: Metadata about the run (name, timestamp, etc.)

## Using the API After Training

The API service automatically loads the latest trained model from MLflow. Here's how to use it:

1. Train a model:
   ```bash
   # Run training with parameters from .env file
   make train
   
   # Or with custom arguments
   make train TRAIN_ARGS="--width 100 --height 100 --iterations 500"
   ```

2. Start the API service (if not already running):
   ```bash
   make api
   ```

3. Send requests to the API:
   ```bash
   # Get information about the loaded model
   curl http://localhost:8000/model-info
   
   # Make a prediction for a single data point
   curl -X POST -H "Content-Type: application/json" \
     -d '{"data": [0.2, 0.5, 0.8]}' \
     http://localhost:8000/predict-bmu
   
   # Make predictions for multiple data points
   curl -X POST -H "Content-Type: application/json" \
     -d '{"data": [[0.2, 0.5, 0.8], [0.9, 0.1, 0.3]]}' \
     http://localhost:8000/predict-batch
   ```

## Makefile Commands

For convenience, common tasks are available as make commands:

```bash
# Build Docker images
make build

# Start all services (MLflow, training, API)
make up

# Run only the training service
make train

# Run only the API service
make api

# Run tests
make test

# Stop all services
make down

# Inspect MLflow data
make inspect

# Show logs
make logs
```

## License

MIT 

## Using Custom Parameters

### Environment-Based Configuration

The project now uses an `.env` file for configuration by default. Create a `.env` file in the project root with your desired parameters:

```
# SOM Configuration Parameters
SOM_WIDTH=50                # Default grid width is now 50 
SOM_HEIGHT=50               # Default grid height is now 50
SOM_INPUT_DIM=3             # Input vector dimension
SOM_ITERATIONS=300          # Training iterations
SOM_SAMPLES=1000            # Number of random training samples
SOM_LEARNING_RATE=0.1       # Initial learning rate
SOM_BATCH_SIZE=             # Batch size for memory-efficient training (leave empty for fully vectorized mode)
SOM_SIGMA=                  # Neighborhood radius (blank = auto calculated)
SOM_RANDOM_STATE=42         # Random seed for reproducibility
SOM_RUN_NAME=som_model      # Name for the MLflow run
SOM_VERBOSE=true            # Enable verbose output during training

# MLflow Configuration
MLFLOW_TRACKING_URI=http://mlflow:5000
MLFLOW_EXPERIMENT_NAME=som-experiments
RUN_ID_FILE=/app/mlflow_data/run_id.txt

# API Configuration
PORT=8000
SOM_RUN_ID_FILE=/app/mlflow_data/run_id.txt

# Model Selection Configuration
METRIC_KEY=quantization_error   # Metric used to select the best model
METRIC_ASCENDING=true           # True -> minimize, False -> maximize
```

These parameters will be automatically used by the Docker containers. Whenever you change parameters:

1. Edit the `.env` file with your new values
2. Rebuild the Docker images: `docker compose build`
3. Restart the services: `docker compose up -d`

This workflow ensures all services use the same configuration parameters.

### Docker Compose Workflow

To use the SOM with Docker Compose:

1. Create or modify your `.env` file with desired parameters
2. Build the Docker images:
   ```bash
   docker compose build
   ```
3. Start all services:
   ```bash
   docker compose up -d
   ```
4. Check the logs to monitor the training process:
   ```bash
   docker compose logs -f train
   ```
5. When training is complete, access the API at `http://localhost:8000`
6. Stop all services when done:
   ```bash
   docker compose down
   ```

> **Important**: After making code changes or updating the `.env` file, you must rebuild the Docker images with `docker compose build` for the changes to take effect.

### Command Line Arguments (Advanced)

You can still override the environment variables with command-line arguments:

```bash
# Using Docker Compose with custom arguments
docker compose run --rm train python -m kohonen.scripts.train_script --width 50 --height 50 --iterations 200
```

Available parameters:

- `--width`: Width of the SOM grid (default: from .env or 30)
- `--height`: Height of the SOM grid (default: from .env or 30)
- `--input-dim`: Input dimension (default: from .env or 3)
- `--iterations`: Number of training iterations (default: from .env or 100)
- `--samples`: Number of random samples to generate (default: from .env or 1000)
- `--learning-rate`: Initial learning rate (default: from .env or 0.1)
- `--sigma`: Initial neighborhood radius (default: from .env or max(width, height)/2)
- `--random-state`: Random seed for reproducibility (default: from .env or None)
- `--run-name`: Name for the MLflow run (default: from .env or "som_model")
- `--verbose`: Enable verbose output during training (default: from .env or false) 

## API Endpoints

The SOM API provides the following endpoints:

### GET /health
Health check endpoint.

### GET /model-info
Get information about the loaded model, including its dimensions, run ID, and metrics.

### POST /predict-bmu
Find the BMU for a single input vector.

```json
{
  "data": [0.2, 0.5, 0.8]
}
```

### POST /predict-batch
Find BMUs for multiple input vectors.

```json
{
  "data": [
    [0.2, 0.5, 0.8],
    [0.9, 0.1, 0.3]
  ]
}
```

### GET /weights/{x}/{y}
Get the weights for a specific node at coordinates (x, y).

### GET /models
List available models with their metrics and parameters.

Query parameters:
- `max_results`: Maximum number of models to return (default: 10)
- `experiment_name`: Filter models by experiment name 