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
    from fastapi import FastAPI, HTTPException, Query
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
        metrics: Optional[Dict[str, float]] = None
        experiment_name: Optional[str] = None
    
    # Global model instance
    model = None
    model_run_id = None
    model_metrics = {}
    
    def get_model():
        """Get or load the SOM model."""
        global model, model_run_id, model_metrics
        
        # Check if we need to load a model
        if model is None:
            try:
                # First try to import necessary modules
                import mlflow
                import numpy as np
                from kohonen.som import SelfOrganizingMap
                from kohonen.mlflow_utils import load_som_model
                from kohonen.model_selection import load_best_model, find_best_run
                from kohonen.config import settings
                
                # Try to get run_id from environment or find the best model
                run_id = os.environ.get("SOM_RUN_ID", "").strip()
                
                # Clean run ID in case there are any unwanted characters
                if '%' in run_id:
                    run_id = run_id.strip('%')
                    logger.warning(f"Cleaned run ID to: {run_id}")
                
                if run_id:
                    # Explicit run ID provided - use it
                    try:
                        logger.info(f"Loading model from MLflow with run_id: {run_id}")
                        model = load_som_model(run_id)
                        model_run_id = run_id
                        logger.info(f"Model loaded successfully: {model.width}x{model.height}")
                        
                        # Try to get metrics
                        try:
                            client = mlflow.tracking.MlflowClient()
                            run = client.get_run(run_id)
                            model_metrics = {k: v for k, v in run.data.metrics.items()}
                            logger.info(f"Loaded metrics for model: {model_metrics}")
                        except Exception as e:
                            logger.warning(f"Could not load metrics for model: {e}")
                    except Exception as e:
                        logger.error(f"Error loading model from MLflow with run_id={run_id}: {e}")
                        model = _create_fallback_model()
                else:
                    # No run ID provided - find best model
                    logger.info("No specific run ID provided, finding best model...")
                    try:
                        # Find best run ID based on settings
                        best_run_id = find_best_run(
                            experiment_name=settings.mlflow_experiment_name,
                            metric_key=settings.metric_key,
                            ascending=settings.metric_ascending
                        )
                        
                        if best_run_id:
                            logger.info(f"Found best model with run_id: {best_run_id}")
                            model = load_som_model(best_run_id)
                            model_run_id = best_run_id
                            
                            # Get metrics for the best model
                            try:
                                client = mlflow.tracking.MlflowClient()
                                run = client.get_run(best_run_id)
                                model_metrics = {k: v for k, v in run.data.metrics.items()}
                                logger.info(f"Loaded metrics for best model: {model_metrics}")
                            except Exception as e:
                                logger.warning(f"Could not load metrics for best model: {e}")
                        else:
                            logger.warning("No suitable model found, creating fallback model")
                            model = _create_fallback_model()
                    except Exception as e:
                        logger.error(f"Error finding/loading best model: {e}")
                        model = _create_fallback_model()
            except ImportError as e:
                # Handle case when MLflow or SOM is not available
                logger.error(f"Error importing required modules: {e}")
                raise HTTPException(
                    status_code=500, 
                    detail="Model loading failed. Required modules not available."
                )
        
        return model
    
    def _create_fallback_model():
        """Create a fallback model for demo purposes."""
        from kohonen.som import SelfOrganizingMap
        from kohonen.config import settings
        
        global model_run_id
        
        # Use settings for default model
        logger.info("Creating a fallback demo model")
        width = settings.som_width
        height = settings.som_height
        input_dim = settings.som_input_dim
        model = SelfOrganizingMap(width, height, input_dim)
        model_run_id = "demo-model"
        
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
            
            # Get experiment name from settings if available
            experiment_name = None
            try:
                from kohonen.config import settings
                experiment_name = settings.mlflow_experiment_name
            except ImportError:
                pass
                
            return ModelInfo(
                width=som.width,
                height=som.height,
                input_dim=som.input_dim,
                run_id=model_run_id,
                metrics=model_metrics,
                experiment_name=experiment_name
            )
        except HTTPException as e:
            # For demo purposes, provide mock info if we can't get the model
            logger.warning("Using demo model info because model couldn't be loaded")
            from kohonen.config import settings
            
            return ModelInfo(
                width=settings.som_width,
                height=settings.som_height,
                input_dim=settings.som_input_dim,
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
            
    @app.get("/models")
    async def list_models(
        max_results: int = Query(10, ge=1, le=100),
        experiment_name: Optional[str] = None
    ):
        """List available models with their metrics."""
        try:
            from kohonen.model_selection import get_model_metrics_summary
            from kohonen.config import settings
            
            # Use provided experiment name or default from settings
            experiment_name = experiment_name or settings.mlflow_experiment_name
            
            # Get model metrics summary
            df = get_model_metrics_summary(
                experiment_name=experiment_name,
                max_results=max_results
            )
            
            if df.empty:
                return {"models": []}
            
            # Convert to records
            records = df.to_dict(orient="records")
            return {"models": records, "experiment_name": experiment_name}
            
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            raise HTTPException(status_code=500, detail=str(e))

except ImportError as e:
    logger.warning(f"Required packages not installed. API functionality will be limited: {e}")
    app = None 