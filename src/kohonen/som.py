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
        verbose: bool = False
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
        
        # Compute all grid positions as a 2D array for fast distance calculation
        grid_positions = np.zeros((self.width * self.height, 2))
        for i in range(self.width):
            for j in range(self.height):
                grid_positions[i * self.height + j] = [i, j]
        
        # Training loop
        for t in range(n_iterations):
            σt = self.sigma_0 * np.exp(-t/λ)
            αt = self.learning_rate_0 * np.exp(-t/λ)
            
            if verbose and t % max(1, n_iterations // 10) == 0:
                logger.info(f"Iteration {t}/{n_iterations}, σ={σt:.4f}, α={αt:.4f}")
            
            # For each input vector
            for vt in input_data:
                # Find Best Matching Unit (BMU)
                bmu_idx = self._find_bmu_fast(vt)
                bmu_x, bmu_y = np.unravel_index(bmu_idx, (self.width, self.height))
                
                # Calculate distance to BMU for all nodes (vectorized)
                # Reshape grid positions for broadcasting
                grid_x = np.arange(self.width).reshape(self.width, 1)
                grid_y = np.arange(self.height).reshape(1, self.height)
                
                # Calculate squared distance to BMU for each node
                dist_sq = (grid_x - bmu_x)**2 + (grid_y - bmu_y)**2
                
                # Calculate neighborhood influence for all nodes at once
                influence = np.exp(-dist_sq / (2 * σt**2))
                
                # Update all weights at once using broadcasting
                # influence shape: (width, height)
                # vt shape: (input_dim,)
                # weights shape: (width, height, input_dim)
                influence = influence.reshape(self.width, self.height, 1)
                self.weights += αt * influence * (vt - self.weights)
            
            # Calculate metrics every 10% of iterations
            if t % max(1, n_iterations // 10) == 0:
                qe = self._quantization_error(input_data)
                self.training_quantization_errors.append(qe)
                
                if verbose:
                    logger.info(f"Quantization error: {qe:.6f}")
        
        # Final metrics
        final_qe = self._quantization_error(input_data)
        self.training_quantization_errors.append(final_qe)
        
        metrics = {
            'quantization_error': final_qe,
            'quantization_error_history': self.training_quantization_errors
        }
        
        logger.info(f"Training completed. Final quantization error: {final_qe:.6f}")
        
        return metrics
    
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
        bmus = np.zeros((input_vectors.shape[0], 2), dtype=int)
        
        for i, vector in enumerate(input_vectors):
            bmu_x, bmu_y = self._find_bmu(vector)
            bmus[i, 0] = bmu_x
            bmus[i, 1] = bmu_y
        
        return bmus
    
    def _quantization_error(self, input_vectors: np.ndarray) -> float:
        """
        Calculate the quantization error for the given input vectors.
        
        The quantization error is the average distance between each input vector
        and its BMU's weight vector.
        
        Parameters:
        -----------
        input_vectors : np.ndarray
            Input vectors of shape (n_samples, input_dim)
            
        Returns:
        --------
        float
            The quantization error
        """
        error_sum = 0.0
        
        for vector in input_vectors:
            bmu_x, bmu_y = self._find_bmu(vector)
            error_sum += np.sqrt(np.sum((vector - self.weights[bmu_x, bmu_y])**2))
        
        return error_sum / input_vectors.shape[0]
    
    def get_weights(self) -> np.ndarray:
        """
        Get the weights of the SOM.
        
        Returns:
        --------
        np.ndarray
            Copy of the SOM weights with shape (width, height, input_dim)
        """
        return self.weights.copy() 