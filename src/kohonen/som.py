"""
Self-Organizing Map implementation module.
"""
import numpy as np
import logging
from typing import Tuple, Optional, Union, Dict, Any

logger = logging.getLogger(__name__)


class SelfOrganizingMap:
    """
    Self-Organizing Map (Kohonen Network) implementation.
    
    This class implements the Kohonen learning algorithm to create a self-organizing map
    from input data. The SOM arranges its nodes in a 2D grid, where each node has a weight
    vector matching the dimensionality of the input data.
    
    Parameters:
    -----------
    width : int
        Width of the SOM grid
    height : int
        Height of the SOM grid
    input_dim : int
        Dimensionality of the input vectors
    learning_rate : float, default=0.1
        Initial learning rate
    sigma : float, optional
        Initial neighborhood radius. If None, it's set to max(width, height)/2
    random_state : Optional[int], default=None
        Random seed for reproducibility
    """

    def __init__(
        self,
        width: int,
        height: int,
        input_dim: int,
        learning_rate: float = 0.1,
        sigma: Optional[float] = None,
        random_state: Optional[int] = None
    ):
        # Validate parameters
        if width <= 0 or height <= 0:
            raise ValueError("Width and height must be positive integers")
        if input_dim <= 0:
            raise ValueError("Input dimension must be a positive integer")
        if learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        
        self.width = width
        self.height = height
        self.input_dim = input_dim
        self.learning_rate_0 = learning_rate
        self.sigma_0 = sigma if sigma is not None else max(width, height) / 2
        
        # Set random state for reproducibility
        if random_state is not None:
            np.random.seed(random_state)
        
        # Initialize weight vectors
        self.weights = np.random.random((width, height, input_dim))
        
        # Pre-compute grid positions for all nodes (for faster BMU calculation)
        self.grid_positions = np.array([(i, j) for i in range(width) for j in range(height)])
        self.grid_positions_reshaped = self.grid_positions.reshape(width, height, 2)
        
        # Training metrics
        self.training_quantization_errors = []
        
    def train(
        self,
        input_data: np.ndarray,
        n_iterations: int,
        verbose: bool = False,
        batch_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Train the SOM on the given input data.
        
        Parameters:
        -----------
        input_data : np.ndarray
            Training vectors with shape (n_samples, input_dim)
        n_iterations : int
            Number of training iterations
        verbose : bool, default=False
            Whether to print progress information
        batch_size : Optional[int], default=None
            If provided, process input vectors in batches of this size for reduced
            memory usage. If None, processes all vectors at once (fully vectorized).
            
            - For small to medium datasets, use None for fastest performance.
            - For large datasets, use a batch size (e.g., 100-500) to reduce memory usage.
            - Memory usage scales with O(n_samples) when batch_size=None and 
              O(batch_size) when batch_size is set to a value.
            
        Returns:
        --------
        dict
            Dictionary containing training metrics
        """
        # Validate input
        if not isinstance(input_data, np.ndarray):
            raise TypeError("Input data must be a numpy array")
        if input_data.ndim != 2:
            raise ValueError("Input data must be 2-dimensional with shape (n_samples, input_dim)")
        if input_data.shape[1] != self.input_dim:
            raise ValueError(f"Input data has {input_data.shape[1]} features, but SOM expects {self.input_dim}")
        if n_iterations <= 0:
            raise ValueError("Number of iterations must be positive")
        
        # Decay parameter
        λ = n_iterations / np.log(self.sigma_0)
        n_samples = input_data.shape[0]
        
        # Reset training metrics
        self.training_quantization_errors = []
        
        logger.info(f"Starting SOM training with {n_iterations} iterations on {n_samples} samples")
        
        # Create grid coordinate matrices once (vectorized)
        grid_x, grid_y = np.meshgrid(np.arange(self.width), np.arange(self.height), indexing='ij')
        
        # Training loop
        for t in range(n_iterations):
            σt = self.sigma_0 * np.exp(-t/λ)
            αt = self.learning_rate_0 * np.exp(-t/λ)
            
            if verbose and t % max(1, n_iterations // 10) == 0:
                logger.info(f"Iteration {t}/{n_iterations}, σ={σt:.4f}, α={αt:.4f}")
            
            # Determine batch processing or full vectorization
            if batch_size is None or batch_size >= n_samples:
                # Fully vectorized approach - processes all input vectors at once
                # Warning: This may use a lot of memory for large datasets
                self._update_weights_fully_vectorized(input_data, αt, σt, grid_x, grid_y)
            else:
                # Batch processing - more memory efficient
                for i in range(0, n_samples, batch_size):
                    batch_end = min(i + batch_size, n_samples)
                    batch = input_data[i:batch_end]
                    self._update_weights_vectorized(batch, αt, σt, grid_x, grid_y)
            
            # Calculate metrics every 10% of iterations
            if t % max(1, n_iterations // 10) == 0:
                qe = self._quantization_error_vectorized(input_data)
                self.training_quantization_errors.append(qe)
                
                if verbose:
                    logger.info(f"Quantization error: {qe:.6f}")
        
        # Final metrics
        final_qe = self._quantization_error_vectorized(input_data)
        self.training_quantization_errors.append(final_qe)
        
        metrics = {
            'quantization_error': final_qe,
            'quantization_errors': self.training_quantization_errors
        }
        
        logger.info(f"Training completed. Final quantization error: {final_qe:.6f}")
        
        return metrics
    
    def _update_weights_vectorized(self, batch_data: np.ndarray, learning_rate: float, sigma: float, 
                                  grid_x: np.ndarray, grid_y: np.ndarray) -> None:
        """
        Update weights for a batch of input vectors using vectorized operations.
        
        Parameters:
        -----------
        batch_data : np.ndarray
            Batch of input vectors with shape (batch_size, input_dim)
        learning_rate : float
            Current learning rate
        sigma : float
            Current neighborhood radius
        grid_x : np.ndarray
            Grid x-coordinate matrix
        grid_y : np.ndarray
            Grid y-coordinate matrix
        """
        # Iterate through the batch (still a loop, but much smaller than the full dataset)
        for vt in batch_data:
            # Find Best Matching Unit (BMU)
            bmu_idx = self._find_bmu_fast(vt)
            bmu_x, bmu_y = np.unravel_index(bmu_idx, (self.width, self.height))
            
            # Calculate squared distance to BMU for each node
            dist_sq = (grid_x - bmu_x)**2 + (grid_y - bmu_y)**2
            
            # Calculate neighborhood influence for all nodes at once
            influence = np.exp(-dist_sq / (2 * sigma**2))
            
            # Update all weights at once using broadcasting
            influence = influence.reshape(self.width, self.height, 1)
            self.weights += learning_rate * influence * (vt - self.weights)
    
    def _update_weights_fully_vectorized(self, input_data: np.ndarray, learning_rate: float, sigma: float,
                                        grid_x: np.ndarray, grid_y: np.ndarray) -> None:
        """
        Fully vectorized weight update for all input vectors at once.
        
        This is faster but more memory intensive.
        
        Parameters:
        -----------
        input_data : np.ndarray
            All input vectors with shape (n_samples, input_dim)
        learning_rate : float
            Current learning rate
        sigma : float
            Current neighborhood radius
        grid_x : np.ndarray
            Grid x-coordinate matrix
        grid_y : np.ndarray
            Grid y-coordinate matrix
        """
        # Find BMUs for all input vectors at once
        bmu_indices = self._find_bmus_batch(input_data)
        bmu_coords = np.array(np.unravel_index(bmu_indices, (self.width, self.height))).T
        
        # For each input vector, update the weights
        for i, vt in enumerate(input_data):
            bmu_x, bmu_y = bmu_coords[i]
            
            # Calculate squared distance to BMU for each node
            dist_sq = (grid_x - bmu_x)**2 + (grid_y - bmu_y)**2
            
            # Calculate neighborhood influence for all nodes at once
            influence = np.exp(-dist_sq / (2 * sigma**2))
            
            # Update all weights at once using broadcasting
            influence = influence.reshape(self.width, self.height, 1)
            self.weights += learning_rate * influence * (vt - self.weights)
    
    def _find_bmus_batch(self, input_vectors: np.ndarray) -> np.ndarray:
        """
        Find BMUs for multiple input vectors at once.
        
        Parameters:
        -----------
        input_vectors : np.ndarray
            Input vectors with shape (n_samples, input_dim)
            
        Returns:
        --------
        np.ndarray
            Array of flattened BMU indices for each input vector
        """
        # Reshape weights for distance calculation
        reshaped_weights = self.weights.reshape(-1, self.input_dim)  # shape: (width*height, input_dim)
        
        # Calculate distances for all input vectors to all neurons at once
        # Using broadcasting to avoid loops
        # Reshape input_vectors to (n_samples, 1, input_dim) for broadcasting
        expanded_inputs = input_vectors.reshape(input_vectors.shape[0], 1, self.input_dim)
        
        # Calculate squared differences for all combinations
        # Result shape: (n_samples, width*height, input_dim)
        squared_diffs = (expanded_inputs - reshaped_weights.reshape(1, -1, self.input_dim))**2
        
        # Sum over features and find minimum distance indices
        distances = np.sum(squared_diffs, axis=2)  # shape: (n_samples, width*height)
        
        # Find indices of minimum distances for each input vector
        return np.argmin(distances, axis=1)  # shape: (n_samples,)
    
    def predict_batch(self, input_vectors: np.ndarray) -> np.ndarray:
        """
        Find the BMUs for a batch of input vectors.
        
        Parameters:
        -----------
        input_vectors : np.ndarray
            Input vectors of shape (n_samples, input_dim)
            
        Returns:
        --------
        np.ndarray
            Array of shape (n_samples, 2) where each row is the (x, y) coordinates of the BMU
        """
        # Use the fully vectorized method to find BMUs
        bmu_indices = self._find_bmus_batch(input_vectors)
        
        # Convert flat indices to (x, y) coordinates
        bmu_coords = np.array(np.unravel_index(bmu_indices, (self.width, self.height))).T
        
        return bmu_coords
    
    def _quantization_error_vectorized(self, input_vectors: np.ndarray) -> float:
        """
        Calculate the quantization error for the given input vectors using vectorized operations.
        
        Parameters:
        -----------
        input_vectors : np.ndarray
            Input vectors of shape (n_samples, input_dim)
            
        Returns:
        --------
        float
            The quantization error
        """
        # Find BMUs for all input vectors at once
        bmu_coords = self.predict_batch(input_vectors)
        
        # Calculate distances between each input vector and its BMU weight
        errors = np.zeros(input_vectors.shape[0])
        
        for i, (x, y) in enumerate(bmu_coords):
            errors[i] = np.sqrt(np.sum((input_vectors[i] - self.weights[x, y])**2))
        
        return np.mean(errors)
    
    def _find_bmu_fast(self, input_vector: np.ndarray) -> int:
        """
        Optimized method to find the Best Matching Unit (BMU) for the given input vector.
        
        Parameters:
        -----------
        input_vector : np.ndarray
            Input vector of shape (input_dim,)
            
        Returns:
        --------
        int
            Flattened index of the BMU in the SOM grid
        """
        # Reshape the weights for efficient distance calculation
        reshaped_weights = self.weights.reshape(-1, self.input_dim)
        
        # Calculate squared Euclidean distance to all nodes at once
        dist_sq = np.sum((reshaped_weights - input_vector)**2, axis=1)
        
        # Find BMU
        return np.argmin(dist_sq)
    
    def _find_bmu(self, input_vector: np.ndarray) -> Tuple[int, int]:
        """
        Find the Best Matching Unit (BMU) for the given input vector.
        
        Parameters:
        -----------
        input_vector : np.ndarray
            Input vector of shape (input_dim,)
            
        Returns:
        --------
        Tuple[int, int]
            (x, y) coordinates of the BMU in the SOM grid
        """
        bmu_idx = self._find_bmu_fast(input_vector)
        bmu_x, bmu_y = np.unravel_index(bmu_idx, (self.width, self.height))
        
        return bmu_x, bmu_y
    
    def predict_bmu(self, input_vector: np.ndarray) -> Tuple[int, int]:
        """
        Find the BMU for an input vector - same as _find_bmu but exposed as a public API.
        
        Parameters:
        -----------
        input_vector : np.ndarray
            Input vector of shape (input_dim,)
            
        Returns:
        --------
        Tuple[int, int]
            (x, y) coordinates of the BMU in the SOM grid
        """
        return self._find_bmu(input_vector)
    
    def get_weights(self) -> np.ndarray:
        """
        Get the weights of the SOM.
        
        Returns:
        --------
        np.ndarray
            Copy of the SOM weights with shape (width, height, input_dim)
        """
        return self.weights.copy() 