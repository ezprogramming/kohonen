"""
Demonstration of FastAPI Integration in Kohonen SOM.

This script showcases:
1. How to serve a trained SOM model via REST API
2. Input validation with Pydantic models
3. Setting up endpoints for both single and batch predictions
4. Comparison with traditional model serving approaches

The script runs a local FastAPI server and demonstrates API requests.
"""
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import json
import requests
import subprocess
import atexit
import uvicorn
import threading
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional, Tuple, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the project root to the path so we can import the package
sys.path.append(str(Path(__file__).resolve().parents[2]))

from kohonen import SelfOrganizingMap
from kohonen.visualization import plot_som_grid

# Set random seed for reproducibility
np.random.seed(42)

# Directory for saving comparison results
RESULTS_DIR = Path(__file__).parent
os.makedirs(RESULTS_DIR, exist_ok=True)

# Create a sample SOM model for the demo
def create_sample_model():
    """Create and train a sample SOM model for the demo"""
    width, height = 20, 20
    input_dim = 3
    n_samples = 2000
    n_iterations = 100
    
    # Generate training data
    data = np.random.rand(n_samples, input_dim)
    
    # Create and train SOM
    som = SelfOrganizingMap(width, height, input_dim)
    som.train(data, n_iterations=n_iterations, verbose=True)
    
    return som

# The global model instance
MODEL = create_sample_model()

# ===== FastAPI App Definition =====

# Pydantic models for request and response validation
class InputVector(BaseModel):
    """Single input vector for SOM prediction"""
    data: List[float] = Field(..., example=[0.2, 0.5, 0.8])
    
    @validator('data')
    def validate_dimensions(cls, v):
        if len(v) != 3:  # Our demo model expects 3D input
            raise ValueError('Input vector must have 3 dimensions')
        return v

class BatchInputVectors(BaseModel):
    """Batch of input vectors for SOM prediction"""
    data: List[List[float]] = Field(..., example=[[0.2, 0.5, 0.8], [0.7, 0.3, 0.1]])
    
    @validator('data')
    def validate_dimensions(cls, v):
        if not v:
            raise ValueError('Batch cannot be empty')
        dims = len(v[0])
        if dims != 3:  # Our demo model expects 3D input
            raise ValueError('Input vectors must have 3 dimensions')
        for vector in v:
            if len(vector) != dims:
                raise ValueError('All input vectors must have the same dimensions')
        return v

class BMUResponse(BaseModel):
    """Response containing BMU coordinates"""
    bmu_x: int
    bmu_y: int

class BatchBMUResponse(BaseModel):
    """Response containing BMU coordinates for multiple inputs"""
    results: List[BMUResponse]

class ModelInfoResponse(BaseModel):
    """Response containing model information"""
    width: int
    height: int
    input_dim: int
    version: str = "1.0.0"
    description: str = "Kohonen Self-Organizing Map"

# Create FastAPI app
app = FastAPI(
    title="Kohonen SOM API",
    description="API for Self-Organizing Map predictions",
    version="1.0.0"
)

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint providing basic information"""
    return {
        "message": "Kohonen SOM API is running",
        "docs_url": "/docs",
        "model_info_url": "/model-info"
    }

@app.get("/model-info", response_model=ModelInfoResponse)
async def model_info():
    """Get information about the loaded model"""
    return {
        "width": MODEL.width,
        "height": MODEL.height,
        "input_dim": MODEL.input_dim,
        "version": "1.0.0",
        "description": "Kohonen Self-Organizing Map"
    }

@app.post("/predict-bmu", response_model=BMUResponse)
async def predict_bmu(input_data: InputVector):
    """
    Find the Best Matching Unit (BMU) for a single input vector
    """
    try:
        # Convert input data to numpy array
        input_vector = np.array(input_data.data)
        
        # Get BMU coordinates
        bmu_x, bmu_y = MODEL.predict_bmu(input_vector)
        
        return {"bmu_x": int(bmu_x), "bmu_y": int(bmu_y)}
    except Exception as e:
        logger.error(f"Error in predict_bmu: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-batch", response_model=BatchBMUResponse)
async def predict_batch(input_data: BatchInputVectors):
    """
    Find BMUs for a batch of input vectors
    """
    try:
        # Convert input data to numpy array
        input_batch = np.array(input_data.data)
        
        # Get BMU coordinates for each input vector
        bmu_coords = MODEL.predict_batch(input_batch)
        
        # Format results
        results = [
            {"bmu_x": int(x), "bmu_y": int(y)} 
            for x, y in bmu_coords
        ]
        
        return {"results": results}
    except Exception as e:
        logger.error(f"Error in predict_batch: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ===== Server Management =====

def start_server(port=8000, host="127.0.0.1"):
    """Start the FastAPI server in a separate thread"""
    server_thread = threading.Thread(
        target=uvicorn.run,
        kwargs={"app": app, "host": host, "port": port, "log_level": "info"},
        daemon=True
    )
    server_thread.start()
    logger.info(f"Server started on http://{host}:{port}")
    
    # Wait for server to start
    time.sleep(2)
    return server_thread


# ===== Comparison Functions =====

def traditional_prediction(model, input_data):
    """Traditional direct model prediction"""
    start_time = time.time()
    result = model.predict_bmu(input_data)
    end_time = time.time()
    return result, end_time - start_time

def api_prediction(input_data, url="http://127.0.0.1:8000/predict-bmu"):
    """Prediction via API call"""
    start_time = time.time()
    response = requests.post(url, json={"data": input_data.tolist()})
    result = response.json()
    end_time = time.time()
    return (result["bmu_x"], result["bmu_y"]), end_time - start_time

def run_latency_comparison(n_samples=100, port=8000):
    """Compare latency of direct prediction vs API prediction"""
    print("Running prediction latency comparison...")
    
    # Generate test data
    test_data = np.random.rand(n_samples, MODEL.input_dim)
    
    # Direct prediction timing
    direct_times = []
    for i in range(n_samples):
        _, time_taken = traditional_prediction(MODEL, test_data[i])
        direct_times.append(time_taken)
    
    # Start API server
    server_thread = start_server(port=port)
    
    # API prediction timing
    api_times = []
    for i in range(n_samples):
        _, time_taken = api_prediction(test_data[i], f"http://127.0.0.1:{port}/predict-bmu")
        api_times.append(time_taken)
    
    return {
        "direct_times": direct_times,
        "api_times": api_times,
        "direct_avg": np.mean(direct_times),
        "api_avg": np.mean(api_times)
    }

def batch_api_test(batch_size=10, url="http://127.0.0.1:8000/predict-batch"):
    """Test batch prediction API performance"""
    # Generate a batch of test vectors
    test_batch = np.random.rand(batch_size, MODEL.input_dim)
    
    # Time the batch API call
    start_time = time.time()
    response = requests.post(url, json={"data": test_batch.tolist()})
    results = response.json()
    end_time = time.time()
    
    return results, end_time - start_time


# ===== Visualization Functions =====

def plot_comparison_results(results):
    """Plot comparison between direct and API prediction latency"""
    direct_times = np.array(results["direct_times"]) * 1000  # Convert to ms
    api_times = np.array(results["api_times"]) * 1000        # Convert to ms
    
    # Create histogram of latencies
    plt.figure(figsize=(12, 6))
    
    plt.hist(direct_times, bins=30, alpha=0.7, label='Direct Prediction', color='#66B2FF')
    plt.hist(api_times, bins=30, alpha=0.7, label='API Prediction', color='#FF9999')
    
    plt.axvline(results["direct_avg"] * 1000, color='#0066CC', linestyle='--', 
                label=f'Direct Avg: {results["direct_avg"]*1000:.2f} ms')
    plt.axvline(results["api_avg"] * 1000, color='#CC0000', linestyle='--', 
                label=f'API Avg: {results["api_avg"]*1000:.2f} ms')
    
    plt.xlabel('Latency (ms)')
    plt.ylabel('Frequency')
    plt.title('Prediction Latency Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(RESULTS_DIR / 'api_latency_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved latency comparison to {RESULTS_DIR / 'api_latency_comparison.png'}")
    plt.close()
    
    # Create bar chart for average latencies
    plt.figure(figsize=(10, 6))
    
    methods = ['Direct Prediction', 'API Prediction']
    avg_times = [results["direct_avg"] * 1000, results["api_avg"] * 1000]
    
    plt.bar(methods, avg_times, color=['#66B2FF', '#FF9999'])
    
    # Add values on top of bars
    for i, v in enumerate(avg_times):
        plt.text(i, v + 0.5, f"{v:.2f} ms", ha='center')
    
    plt.ylabel('Average Latency (ms)')
    plt.title('Average Prediction Latency by Method')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.savefig(RESULTS_DIR / 'api_avg_latency.png', dpi=300, bbox_inches='tight')
    print(f"Saved average latency comparison to {RESULTS_DIR / 'api_avg_latency.png'}")
    plt.close()


def create_api_diagrams():
    """Create diagrams explaining the API integration"""
    # Create API architecture diagram
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    api_arch_text = """
    Traditional Model Serving vs. FastAPI Integration
    -----------------------------------------------
    
    Traditional Approach:
    -------------------
    1. Load model in same process as application
    2. Make direct function calls to the model
    3. No separation between application and model
    4. Limited scalability and flexibility
    
    FastAPI Integration:
    ------------------
    Client → HTTP Request → FastAPI Server → SOM Model → Response → Client
    
    Benefits:
    --------
    1. Decoupling - separates model from application logic
    2. Scalability - can deploy multiple API instances
    3. Language-agnostic - any language can call the API
    4. Input validation - automatic using Pydantic
    5. Documentation - automatic Swagger/OpenAPI docs
    6. Versioning - easily support multiple model versions
    """
    
    ax.text(0.1, 0.5, api_arch_text, fontsize=12, family='monospace', va='center')
    
    plt.savefig(RESULTS_DIR / 'api_architecture.png', dpi=300, bbox_inches='tight')
    print(f"Saved API architecture diagram to {RESULTS_DIR / 'api_architecture.png'}")
    plt.close()
    
    # Create API endpoints diagram
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    endpoints_text = """
    FastAPI Endpoints for Kohonen SOM
    -------------------------------
    
    GET /
    - Basic API information
    - Links to documentation and model info
    
    GET /model-info
    - Grid dimensions (width, height)
    - Input dimensions
    - Model version and description
    
    POST /predict-bmu
    - Input: Single vector JSON {"data": [0.2, 0.5, 0.8]}
    - Output: BMU coordinates {"bmu_x": 10, "bmu_y": 15}
    - Validation: Ensures vector has correct dimensions
    
    POST /predict-batch
    - Input: Multiple vectors {"data": [[0.2, 0.5, 0.8], [0.7, 0.3, 0.1]]}
    - Output: Multiple BMU coordinates
      {"results": [{"bmu_x": 10, "bmu_y": 15}, {"bmu_x": 5, "bmu_y": 8}]}
    - More efficient for multiple predictions
    """
    
    ax.text(0.1, 0.5, endpoints_text, fontsize=12, family='monospace', va='center')
    
    plt.savefig(RESULTS_DIR / 'api_endpoints.png', dpi=300, bbox_inches='tight')
    print(f"Saved API endpoints diagram to {RESULTS_DIR / 'api_endpoints.png'}")
    plt.close()
    
    # Create API vs direct code diagram
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis('off')
    
    code_comparison_text = """
    Code Comparison: Direct Model Use vs. API Integration
    --------------------------------------------------
    
    Direct Model Use:
    ---------------
    ```python
    # Import the model package directly
    from kohonen import SelfOrganizingMap
    
    # Load model (tied to specific Python environment)
    model = SelfOrganizingMap(width=20, height=20, input_dim=3)
    model.train(training_data, n_iterations=100)
    
    # Make prediction (tied to Python)
    input_vector = np.array([0.2, 0.5, 0.8])
    bmu_x, bmu_y = model.predict_bmu(input_vector)
    ```
    
    API Integration (Python client):
    -----------------------------
    ```python
    # Simple HTTP request, no model package dependency
    import requests
    
    # Make prediction (language-agnostic)
    response = requests.post(
        "http://model-api:8000/predict-bmu",
        json={"data": [0.2, 0.5, 0.8]}
    )
    result = response.json()
    bmu_x, bmu_y = result["bmu_x"], result["bmu_y"]
    ```
    
    API Integration (JavaScript client):
    --------------------------------
    ```javascript
    // Make same prediction from JavaScript
    fetch("http://model-api:8000/predict-bmu", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ data: [0.2, 0.5, 0.8] })
    })
    .then(response => response.json())
    .then(result => {
        const bmuX = result.bmu_x;
        const bmuY = result.bmu_y;
        console.log(`BMU: (${bmuX}, ${bmuY})`);
    });
    ```
    """
    
    ax.text(0.1, 0.5, code_comparison_text, fontsize=12, family='monospace', va='center')
    
    plt.savefig(RESULTS_DIR / 'api_code_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved API code comparison to {RESULTS_DIR / 'api_code_comparison.png'}")
    plt.close()


if __name__ == "__main__":
    print("Running Kohonen SOM FastAPI Integration Demonstration")
    
    # 1. Create diagrams to explain API integration
    create_api_diagrams()
    
    # 2. Run latency comparison
    port = 8765  # Use a non-standard port to avoid conflicts
    latency_results = run_latency_comparison(n_samples=50, port=port)
    
    # 3. Plot latency comparison results
    plot_comparison_results(latency_results)
    
    # 4. Test batch API
    print("\nTesting batch API endpoint...")
    batch_results, batch_time = batch_api_test(batch_size=20, url=f"http://127.0.0.1:{port}/predict-batch")
    print(f"Processed 20 vectors in {batch_time:.4f} seconds ({batch_time/20*1000:.2f} ms per vector)")
    
    # 5. Display model information
    model_info_response = requests.get(f"http://127.0.0.1:{port}/model-info").json()
    print("\nModel Information:")
    for key, value in model_info_response.items():
        print(f"  {key}: {value}")
    
    print("\nFastAPI integration demonstration complete!") 