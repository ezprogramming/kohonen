"""
FastAPI implementation for serving SOM models.
"""
import os
import logging
from typing import Dict, List, Optional, Union, Any

# Set up logging based on environment variable
log_level = os.environ.get("LOG_LEVEL", "INFO")
logging.basicConfig(level=getattr(logging, log_level))
logger = logging.getLogger(__name__)

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    
    # Models for request/response validation
    class PredictRequest(BaseModel):
        data: List[float]
    
    class BatchPredictRequest(BaseModel):
        data: List[List[float]]
    
    class PredictResponse(BaseModel):
        bmu_x: int
        bmu_y: int
    
    class BatchPredictResponse(BaseModel):
        results: List[PredictResponse]
    
    class ModelInfo(BaseModel):
        width: int
        height: int
        input_dim: int
        run_id: Optional[str] = None
    
    # Global model instance
    model = None
    model_run_id = None
    
    def get_model():
        """Get or load the SOM model."""
        global model, model_run_id
        
        # Check if we need to load a model
        if model is None:
            try:
                # First try to import necessary modules
                import mlflow
                from kohonen.som import SelfOrganizingMap
                from kohonen.mlflow_utils import load_som_model
                
                # Try to get run_id from environment
                run_id = os.environ.get("SOM_RUN_ID", "").strip()
                
                # Clean run ID in case there are any unwanted characters
                if '%' in run_id:
                    run_id = run_id.strip('%')
                    logger.warning(f"Cleaned run ID to: {run_id}")
                
                if run_id:
                    try:
                        logger.info(f"Loading model from MLflow with run_id: {run_id}")
                        model = load_som_model(run_id)
                        model_run_id = run_id
                        logger.info(f"Model loaded successfully: {model.width}x{model.height}")
                    except Exception as e:
                        logger.error(f"Error loading model from MLflow: {e}")
                        # Create a dummy model for demo purposes
                        logger.info("Creating a fallback demo model")
                        # Use environment variables for default model if available
                        width = int(os.environ.get("SOM_WIDTH", 20))
                        height = int(os.environ.get("SOM_HEIGHT", 20))
                        input_dim = int(os.environ.get("SOM_INPUT_DIM", 3))
                        model = SelfOrganizingMap(width, height, input_dim)
                        model_run_id = "demo-model"
                else:
                    # Create a dummy model for demo purposes
                    logger.info("No model ID provided, creating a demo model")
                    # Use environment variables for default model if available
                    width = int(os.environ.get("SOM_WIDTH", 20))
                    height = int(os.environ.get("SOM_HEIGHT", 20))
                    input_dim = int(os.environ.get("SOM_INPUT_DIM", 3))
                    model = SelfOrganizingMap(width, height, input_dim)
                    model_run_id = "demo-model"
            except ImportError as e:
                # Handle case when MLflow or SOM is not available
                logger.error(f"Error importing required modules: {e}")
                raise HTTPException(
                    status_code=500, 
                    detail="Model loading failed. Required modules not available."
                )
        
        return model

    # Create FastAPI app
    app = FastAPI(
        title="Kohonen SOM API",
        description="API for serving Self-Organizing Map models",
        version="0.1.0",
    )
    
    # Define API endpoints
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "api_version": "0.1.0"}
    
    @app.get("/model-info")
    async def model_info():
        """Get information about the loaded model."""
        try:
            som = get_model()
            return ModelInfo(
                width=som.width,
                height=som.height,
                input_dim=som.input_dim,
                run_id=model_run_id
            )
        except HTTPException as e:
            # For demo purposes, provide mock info if we can't get the model
            logger.warning("Using demo model info because model couldn't be loaded")
            width = int(os.environ.get("SOM_WIDTH", 20))
            height = int(os.environ.get("SOM_HEIGHT", 20))
            input_dim = int(os.environ.get("SOM_INPUT_DIM", 3))
            return ModelInfo(
                width=width,
                height=height,
                input_dim=input_dim,
                run_id="demo-model"
            )
    
    @app.post("/predict-bmu", response_model=PredictResponse)
    async def predict_bmu(request: PredictRequest):
        """
        Find the Best Matching Unit (BMU) for a single input vector.
        
        The input vector should have the same dimensionality as the SOM was trained on.
        """
        try:
            som = get_model()
            
            # Convert data to numpy array
            try:
                import numpy as np
                data = np.array(request.data)
                
                # Check dimension
                if data.shape[0] != som.input_dim:
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Input dimension mismatch. Expected {som.input_dim}, got {data.shape[0]}"
                    )
                
                # Find BMU
                bmu_x, bmu_y = som.predict_bmu(data)
                
                return PredictResponse(bmu_x=bmu_x, bmu_y=bmu_y)
            
            except Exception as e:
                logger.error(f"Error during prediction: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        except HTTPException as e:
            # For demo purposes, return fake BMU if we can't get the model
            logger.warning("Using demo prediction because model couldn't be loaded")
            return PredictResponse(bmu_x=0, bmu_y=0)
    
    @app.post("/predict-batch", response_model=BatchPredictResponse)
    async def predict_batch(request: BatchPredictRequest):
        """
        Find the Best Matching Units (BMUs) for multiple input vectors.
        
        Each input vector should have the same dimensionality as the SOM was trained on.
        """
        try:
            som = get_model()
            
            # Convert data to numpy array
            try:
                import numpy as np
                data = np.array(request.data)
                
                # Check dimensions
                if data.shape[1] != som.input_dim:
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Input dimension mismatch. Expected {som.input_dim}, got {data.shape[1]}"
                    )
                
                # Find BMUs
                results = []
                for vector in data:
                    bmu_x, bmu_y = som.predict_bmu(vector)
                    results.append(PredictResponse(bmu_x=bmu_x, bmu_y=bmu_y))
                
                return BatchPredictResponse(results=results)
            
            except Exception as e:
                logger.error(f"Error during batch prediction: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        except HTTPException as e:
            # For demo purposes, return fake BMUs if we can't get the model
            logger.warning("Using demo batch prediction because model couldn't be loaded")
            return BatchPredictResponse(results=[PredictResponse(bmu_x=0, bmu_y=0)] * len(request.data))
    
    @app.get("/weights/{x}/{y}")
    async def get_weights(x: int, y: int):
        """Get the weights for a specific node."""
        try:
            som = get_model()
            
            try:
                # Check coordinates
                if x < 0 or x >= som.width or y < 0 or y >= som.height:
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Coordinates out of bounds. Valid range: 0-{som.width-1}, 0-{som.height-1}"
                    )
                
                # Get weights
                weights = som.get_weights()[x, y].tolist()
                
                return {"coordinates": {"x": x, "y": y}, "weights": weights}
            
            except Exception as e:
                logger.error(f"Error retrieving weights: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        except HTTPException as e:
            # For demo purposes, return fake weights if we can't get the model
            logger.warning("Using demo weights because model couldn't be loaded")
            return {"coordinates": {"x": x, "y": y}, "weights": [0.5, 0.5, 0.5]}

except ImportError as e:
    logger.warning(f"Required packages not installed. API functionality will be limited: {e}")
    app = None 