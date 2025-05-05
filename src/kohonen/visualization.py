"""
Visualization utilities for Self-Organizing Maps.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Optional, Tuple, Union, List


def plot_som_grid(
    weights: np.ndarray,
    figsize: Tuple[int, int] = (10, 10),
    cmap: str = 'viridis',
    title: Optional[str] = None
) -> Figure:
    """
    Plot the SOM grid as an image.
    
    Parameters:
    -----------
    weights : np.ndarray
        The SOM weight vectors with shape (width, height, 3) for RGB or (width, height, 1) for grayscale
    figsize : Tuple[int, int], default=(10, 10)
        Figure size (width, height) in inches
    cmap : str, default='viridis'
        Colormap for grayscale images
    title : Optional[str], default=None
        Title for the plot
        
    Returns:
    --------
    Figure
        The matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # If input depth is 3, assume RGB
    if weights.shape[2] == 3:
        # Ensure RGB values are within [0, 1] range
        img = np.clip(weights, 0, 1)
        ax.imshow(img)
    # If input depth is 1, use grayscale
    elif weights.shape[2] == 1:
        ax.imshow(weights.squeeze(), cmap=cmap)
    else:
        # For higher dimensions, use the first 3 for RGB if available, otherwise grayscale
        if weights.shape[2] >= 3:
            img = np.clip(weights[:, :, :3], 0, 1)
            ax.imshow(img)
        else:
            # Use the first dimension as grayscale
            ax.imshow(weights[:, :, 0], cmap=cmap)
    
    if title:
        ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.tight_layout()
    
    return fig


def plot_component_planes(
    weights: np.ndarray,
    component_names: Optional[List[str]] = None,
    figsize: Optional[Tuple[int, int]] = None,
    cmap: str = 'viridis',
    title: Optional[str] = None
) -> Figure:
    """
    Plot the component planes of the SOM.
    
    Component planes visualize each dimension of the weight vectors separately.
    
    Parameters:
    -----------
    weights : np.ndarray
        The SOM weight vectors with shape (width, height, input_dim)
    component_names : Optional[List[str]], default=None
        Names for the components/dimensions
    figsize : Optional[Tuple[int, int]], default=None
        Figure size (width, height) in inches
    cmap : str, default='viridis'
        Colormap for the component planes
    title : Optional[str], default=None
        Overall title for the plot
        
    Returns:
    --------
    Figure
        The matplotlib figure
    """
    n_components = weights.shape[2]
    
    # Calculate grid layout
    n_cols = min(3, n_components)
    n_rows = (n_components + n_cols - 1) // n_cols
    
    # Set default figsize based on components
    if figsize is None:
        figsize = (n_cols * 4, n_rows * 4)
    
    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Handle case with only one component
    if n_components == 1:
        axs = np.array([axs])
    
    axs = axs.flatten()
    
    # Component names
    if component_names is None:
        component_names = [f"Component {i+1}" for i in range(n_components)]
    
    # Plot each component
    for i in range(n_components):
        ax = axs[i]
        component = weights[:, :, i]
        im = ax.imshow(component, cmap=cmap)
        ax.set_title(component_names[i])
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax)
    
    # Hide unused subplots
    for i in range(n_components, len(axs)):
        axs[i].set_visible(False)
    
    if title:
        plt.suptitle(title)
    
    plt.tight_layout()
    if title:
        plt.subplots_adjust(top=0.9)
    
    return fig


def plot_sample_positions(
    som_width: int,
    som_height: int,
    bmu_positions: np.ndarray,
    labels: Optional[np.ndarray] = None,
    figsize: Tuple[int, int] = (10, 10),
    cmap: str = 'tab10',
    alpha: float = 0.7,
    title: Optional[str] = None
) -> Figure:
    """
    Plot the positions of input samples on the SOM grid.
    
    Parameters:
    -----------
    som_width : int
        Width of the SOM grid
    som_height : int
        Height of the SOM grid
    bmu_positions : np.ndarray
        Array of shape (n_samples, 2) containing the (x, y) BMU coordinates for each sample
    labels : Optional[np.ndarray], default=None
        Array of shape (n_samples,) containing the labels for each sample
    figsize : Tuple[int, int], default=(10, 10)
        Figure size (width, height) in inches
    cmap : str, default='tab10'
        Colormap for the labels
    alpha : float, default=0.7
        Alpha/transparency value for points
    title : Optional[str], default=None
        Title for the plot
        
    Returns:
    --------
    Figure
        The matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set up 2D grid
    ax.set_xlim(-0.5, som_width - 0.5)
    ax.set_ylim(-0.5, som_height - 0.5)
    
    # Draw grid lines
    for i in range(som_width + 1):
        ax.axvline(i - 0.5, color='gray', linestyle='--', alpha=0.3)
    for i in range(som_height + 1):
        ax.axhline(i - 0.5, color='gray', linestyle='--', alpha=0.3)
    
    # Plot data points based on their BMU
    if labels is not None:
        scatter = ax.scatter(
            bmu_positions[:, 0], 
            bmu_positions[:, 1], 
            c=labels, 
            cmap=cmap, 
            s=50, 
            alpha=alpha,
            edgecolors='w',
        )
        # Add legend for class labels
        unique_labels = np.unique(labels)
        if len(unique_labels) <= 10:  # Only add legend if reasonable number of classes
            handles, _ = scatter.legend_elements()
            ax.legend(handles, [f"Class {i}" for i in unique_labels], 
                     title="Classes", loc="upper right")
    else:
        ax.scatter(
            bmu_positions[:, 0], 
            bmu_positions[:, 1], 
            s=50, 
            alpha=alpha,
            edgecolors='w',
        )
    
    if title:
        ax.set_title(title)
    
    ax.set_aspect('equal')
    ax.set_xticks(range(som_width))
    ax.set_yticks(range(som_height))
    
    plt.tight_layout()
    
    return fig

# Add aliases for backward compatibility and improved naming
visualize_som_grid = plot_som_grid
visualize_component_planes = plot_component_planes
visualize_sample_positions = plot_sample_positions 