# %%
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
import matplotlib.gridspec as gridspec

# Get the project root directory
project_root = Path(__file__).resolve().parent.parent  # Adjust if needed
sys.path.append(str(project_root))

# Import from vis_utils
from vis_utils import (
    convert_xzy_hmaps,
    CONTROLLER_PATH_PREFIX,
    CONTROLLER_NAME,
    WORLD_NAME,
    OUTPUT_DIR
)

def load_grid_cell_data(scale):
    """
    Load grid cell data for a specific scale.
    
    Args:
        scale (int): Scale to load
        
    Returns:
        tuple: (hmap_loc, hmap_gcn)
    """
    import pickle
    import os
    import numpy as np
    
    # Define the path for hmaps
    hmap_directory = os.path.join(
        CONTROLLER_PATH_PREFIX, CONTROLLER_NAME, "pkl", WORLD_NAME, "hmaps"
    )
    
    # Load location data
    hmap_file = "hmap_loc.pkl"
    file_path = os.path.join(hmap_directory, hmap_file)
    with open(file_path, "rb") as f:
        hmap_loc = np.array(pickle.load(f))
        # remove first element
        hmap_loc = hmap_loc[1:]
    print(f"Loaded hmap_loc from {file_path}")
    
    # Load GCN data for the specified scale
    hmap_file = f"hmap_gcn_scale_{scale}.pkl"
    file_path = os.path.join(hmap_directory, hmap_file)
    try:
        with open(file_path, "rb") as f:
            hmap_gcn = np.array(pickle.load(f))
            # remove first element
            hmap_gcn = hmap_gcn[1:]
            print(f"Loaded hmap_gcn_scale_{scale} from {file_path}")
    except FileNotFoundError:
        print(f"Error: {file_path} not found. Cannot process scale {scale}.")
        return None, None
    
    return hmap_loc, hmap_gcn

def plot_average_grid_activation(
    hmap_x,
    hmap_y,
    hmap_data,
    scale,
    ax=None,
    cmap='viridis',
):
    """
    Plots a hexbin plot for the average activation of all grid cells.
    
    Args:
        hmap_x: X coordinates of the grid.
        hmap_y: Y coordinates of the grid.
        hmap_data: Activation data for the cells.
        scale: Scale identifier for title
        ax: Matplotlib axis to plot on (optional)
        cmap: Colormap to use
    
    Returns:
        matplotlib.axes.Axes: The axis with the plot
    """
    # Create plot if no axis is provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # Calculate the average activation across all grid cells
    average_activations = np.mean(hmap_data, axis=1)
    
    # Use hexbin for the plot
    hb = ax.hexbin(
        hmap_x,
        hmap_y,
        C=average_activations,
        gridsize=100,
        reduce_C_function=np.mean,
        cmap=cmap,
        edgecolors="none",
    )
    
    # Add colorbar
    cbar = plt.colorbar(hb, ax=ax)
    cbar.set_label('Average Activation')
    
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(f"Scale {scale}: Average Activation Across All Grid Cells")
    
    return ax

def plot_single_grid_cell(
    hmap_x,
    hmap_y,
    hmap_data,
    cell_index,
    scale,
    ax=None,
    cmap='viridis',
):
    """
    Plots a hexbin plot for a single grid cell.
    
    Args:
        hmap_x: X coordinates of the grid.
        hmap_y: Y coordinates of the grid.
        hmap_data: Activation data for all cells.
        cell_index: Index of the cell to plot
        scale: Scale identifier for title
        ax: Matplotlib axis to plot on (optional)
        cmap: Colormap to use
    
    Returns:
        matplotlib.axes.Axes: The axis with the plot
    """
    # Create plot if no axis is provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # Get activations for this specific cell
    cell_activations = hmap_data[:, cell_index]
    
    # Use hexbin for the plot
    hb = ax.hexbin(
        hmap_x,
        hmap_y,
        C=cell_activations,
        gridsize=100,
        reduce_C_function=np.mean,
        cmap=cmap,
        edgecolors="none",
    )
    
    # Add colorbar
    cbar = plt.colorbar(hb, ax=ax)
    cbar.set_label('Activation')
    
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(f"Scale {scale}: Grid Cell {cell_index}")
    
    return ax

def find_best_grid_cell(hmap_gcn):
    """
    Find a grid cell with strong activation patterns.
    
    Args:
        hmap_gcn: Grid cell activations
        
    Returns:
        int: Index of a grid cell with strong activation
    """
    # Calculate total activation for each cell
    total_activations = np.sum(np.abs(hmap_gcn), axis=0)
    
    # Get indices of cells with non-zero activation, sorted by activation strength
    active_cells = np.where(total_activations > 0)[0]
    
    if len(active_cells) == 0:
        return 0  # Default to first cell if none are active
    
    # Take one of the top cells with highest activation
    top_cells = active_cells[np.argsort(-total_activations[active_cells])[:10]]
    
    # Return a randomly selected cell from the top cells
    return np.random.choice(top_cells)

def plot_multi_scale_grid_activations(
    scales=None,
    save_path=None,
    show_plot=True,
    cmap='viridis',
):
    """
    Create a figure with grid cell visualizations for multiple scales.
    
    Args:
        scales: List of scale indices to plot
        save_path: Path to save the figure
        show_plot: Whether to show the figure
        cmap: Colormap to use
    """
    if scales is None:
        scales = [0, 1, 2]
    
    # Create figure
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 3)
    
    # Process each scale
    for i, scale in enumerate(scales):
        print(f"\n==== Processing Scale {scale} ====")
        
        # Load data
        hmap_loc, hmap_gcn = load_grid_cell_data(scale)
        
        # Skip if data couldn't be loaded
        if hmap_loc is None or hmap_gcn is None:
            print(f"Skipping scale {scale} due to missing data")
            continue
        
        # Convert coordinates
        hmap_x, hmap_z, hmap_y = convert_xzy_hmaps(hmap_loc)
        
        # Find a good grid cell to visualize
        cell_index = find_best_grid_cell(hmap_gcn)
        
        # Plot single grid cell (top row)
        ax_single = plt.subplot(gs[0, i])
        plot_single_grid_cell(
            hmap_x=hmap_x,
            hmap_y=hmap_y,
            hmap_data=hmap_gcn,
            cell_index=cell_index,
            scale=scale,
            ax=ax_single,
            cmap=cmap,
        )
        
        # Plot average activation (bottom row)
        ax_avg = plt.subplot(gs[1, i])
        plot_average_grid_activation(
            hmap_x=hmap_x,
            hmap_y=hmap_y,
            hmap_data=hmap_gcn,
            scale=scale,
            ax=ax_avg,
            cmap=cmap,
        )
    
    # Add overall title
    plt.suptitle("Grid Cell Activations Across Scales", fontsize=16, y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure if requested
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    # Show figure if requested
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    return fig

# %%
if __name__ == "__main__":
    print("Starting multi-scale grid cell activation visualization...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create output directory
    output_dir = os.path.join(OUTPUT_DIR, "grid_cells")
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate and show the plot
    plot_multi_scale_grid_activations(
        scales=[0, 1, 2],
        save_path=os.path.join(output_dir, "grid_cell_activations_multi_scale.png"),
        show_plot=True,
        cmap='viridis',
    )
    
    print("Visualization complete!")