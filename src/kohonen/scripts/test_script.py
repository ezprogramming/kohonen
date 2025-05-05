#!/usr/bin/env python3
"""
Test script for the Self-Organizing Map implementation.

This file contains both unit tests for the SOM implementation and
a function to run the tests from the command line.
"""
import os
import sys
import pytest
import logging
import numpy as np
from kohonen.som import SelfOrganizingMap

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#---------------------------
# Unit Tests for SOM
#---------------------------

def test_som_initialization():
    """Test SOM initialization with valid parameters."""
    width, height, input_dim = 10, 10, 3
    
    # Basic initialization
    som = SelfOrganizingMap(width, height, input_dim)
    
    assert som.width == width
    assert som.height == height
    assert som.input_dim == input_dim
    assert som.weights.shape == (width, height, input_dim)
    # Check that grid positions were calculated
    assert hasattr(som, 'grid_positions')
    assert som.grid_positions.shape == (width * height, 2)
    assert hasattr(som, 'grid_positions_reshaped')
    assert som.grid_positions_reshaped.shape == (width, height, 2)


def test_som_initialization_validation():
    """Test SOM initialization parameter validation."""
    # Invalid width
    with pytest.raises(ValueError):
        SelfOrganizingMap(0, 10, 3)
    
    # Invalid height
    with pytest.raises(ValueError):
        SelfOrganizingMap(10, -1, 3)
    
    # Invalid input dimension
    with pytest.raises(ValueError):
        SelfOrganizingMap(10, 10, 0)
    
    # Invalid learning rate
    with pytest.raises(ValueError):
        SelfOrganizingMap(10, 10, 3, learning_rate=0)


def test_find_bmu():
    """Test finding the Best Matching Unit."""
    width, height, input_dim = 3, 3, 2
    som = SelfOrganizingMap(width, height, input_dim, random_state=42)
    
    # Simple case - create an input vector that exactly matches a weight vector
    target_x, target_y = 1, 2
    target_value = som.weights[target_x, target_y].copy()
    
    # Find the BMU for this vector
    bmu_x, bmu_y = som._find_bmu(target_value)
    
    # Assert the BMU is the expected node
    assert bmu_x == target_x
    assert bmu_y == target_y


def test_train_validation():
    """Test validation of training parameters."""
    som = SelfOrganizingMap(10, 10, 3)
    
    # Invalid input data type
    with pytest.raises(TypeError):
        som.train("not a numpy array", 100)
    
    # Invalid input data shape (1D)
    with pytest.raises(ValueError):
        som.train(np.array([1, 2, 3]), 100)
    
    # Invalid input data dimensions
    with pytest.raises(ValueError):
        som.train(np.random.random((10, 4)), 100)  # Expected 3 features
    
    # Invalid iterations
    with pytest.raises(ValueError):
        som.train(np.random.random((10, 3)), 0)


def test_train_simple_case():
    """Test training on a simple case with deterministic results."""
    width, height, input_dim = 3, 3, 2
    som = SelfOrganizingMap(width, height, input_dim, random_state=42)
    
    # Simple 2D data with 4 samples forming a square
    input_data = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    
    # Save initial weights for comparison
    initial_weights = som.weights.copy()
    
    # Train for a few iterations
    metrics = som.train(input_data, n_iterations=10)
    
    # Verify training changed the weights
    assert not np.array_equal(initial_weights, som.weights)
    
    # Verify metrics are returned
    assert 'quantization_error' in metrics
    assert 'quantization_errors' in metrics
    
    # Verify quantization error is computed and valid
    assert np.isscalar(metrics['quantization_error'])
    assert metrics['quantization_error'] >= 0
    assert len(metrics['quantization_errors']) > 0


def test_predict_batch():
    """Test batch prediction."""
    width, height, input_dim = 3, 3, 2
    som = SelfOrganizingMap(width, height, input_dim, random_state=42)
    
    # Simple 2D data with 4 samples
    input_data = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    
    # Get BMUs for all samples
    bmus = som.predict_batch(input_data)
    
    # Check shape of result
    assert bmus.shape == (4, 2)
    
    # Check if coordinates are within grid bounds
    assert np.all(bmus[:, 0] >= 0) and np.all(bmus[:, 0] < width)
    assert np.all(bmus[:, 1] >= 0) and np.all(bmus[:, 1] < height)
    
    # Check individual predictions match batch predictions
    for i, sample in enumerate(input_data):
        bmu_x, bmu_y = som.predict_bmu(sample)
        assert bmu_x == bmus[i, 0]
        assert bmu_y == bmus[i, 1]


def test_get_weights():
    """Test the get_weights method returns a copy of the weights."""
    som = SelfOrganizingMap(10, 10, 3, random_state=42)
    
    # Get weights
    weights = som.get_weights()
    
    # Verify it's a copy (modifying it shouldn't affect original)
    original = som.weights.copy()
    weights[0, 0, 0] = 999
    
    assert som.weights[0, 0, 0] != 999
    assert np.array_equal(som.weights, original)


#---------------------------
# Test Runner Function
#---------------------------

def run_tests():
    """Run all SOM tests with pytest."""
    logger.info("Running SOM tests...")
    
    # Get the directory of this script
    this_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Run pytest with coverage
    test_args = [
        "-xvs",                   # Verbose output, exit on first failure
        "--cov=kohonen",         # Measure code coverage for the kohonen package
        "--cov-report=term",     # Print coverage report to terminal
        __file__                 # Test this file itself
    ]
    
    logger.info(f"Running pytest with arguments: {test_args}")
    exit_code = pytest.main(test_args)
    
    if exit_code == 0:
        logger.info("All tests passed!")
    else:
        logger.error(f"Tests failed with exit code: {exit_code}")
    
    return exit_code

if __name__ == "__main__":
    sys.exit(run_tests()) 