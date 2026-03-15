# %%
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
import matplotlib.gridspec as gridspec
import glob

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


def _hmap_directory():
    return os.path.join(
        CONTROLLER_PATH_PREFIX, CONTROLLER_NAME, "pkl", WORLD_NAME, "hmaps"
    )


def discover_available_gcn_scales():
    """Discover all scales with saved GCN hmaps (supports prefixed trial files)."""
    hmap_directory = _hmap_directory()
    if not os.path.exists(hmap_directory):
        return []

    scales = set()
    for file_path in glob.glob(os.path.join(hmap_directory, "*hmap_gcn_scale_*.pkl")):
        stem = os.path.splitext(os.path.basename(file_path))[0]
        try:
            scales.add(int(stem.split("_")[-1]))
        except ValueError:
            continue
    return sorted(scales)


def _resolve_hmap_pair_for_scale(scale):
    """
    Resolve matching (loc, gcn) files for a scale.
    Supports:
    - non-prefixed: hmap_loc.pkl + hmap_gcn_scale_{scale}.pkl
    - prefixed: {prefix}hmap_loc.pkl + {prefix}hmap_gcn_scale_{scale}.pkl
    """
    hmap_directory = _hmap_directory()
    base_gcn = os.path.join(hmap_directory, f"hmap_gcn_scale_{scale}.pkl")
    base_loc = os.path.join(hmap_directory, "hmap_loc.pkl")
    if os.path.exists(base_gcn) and os.path.exists(base_loc):
        return base_loc, base_gcn

    matches = sorted(
        glob.glob(os.path.join(hmap_directory, f"*hmap_gcn_scale_{scale}.pkl"))
    )
    for gcn_file in matches:
        suffix = f"hmap_gcn_scale_{scale}.pkl"
        prefix = os.path.basename(gcn_file)[: -len(suffix)]
        loc_file = os.path.join(hmap_directory, f"{prefix}hmap_loc.pkl")
        if os.path.exists(loc_file):
            return loc_file, gcn_file

    return None, None

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
    
    loc_path, gcn_path = _resolve_hmap_pair_for_scale(scale)
    if loc_path is None or gcn_path is None:
        hmap_directory = _hmap_directory()
        compact_path = os.path.join(hmap_directory, "hmap_compact_stats.pkl")
        if os.path.exists(compact_path):
            print(
                f"Error: no full GCN hmap for scale {scale} in {hmap_directory}. "
                f"Only compact stats found ({compact_path}); spatial GC plotting requires full hmaps."
            )
        else:
            print(f"Error: no matching hmap_loc + hmap_gcn_scale_{scale} files found in {hmap_directory}.")
        return None, None

    with open(loc_path, "rb") as f:
        hmap_loc = np.array(pickle.load(f))
        if len(hmap_loc) > 1:
            hmap_loc = hmap_loc[1:]
    print(f"Loaded hmap_loc from {loc_path}")

    with open(gcn_path, "rb") as f:
        hmap_gcn = np.array(pickle.load(f))
        if len(hmap_gcn) > 1:
            hmap_gcn = hmap_gcn[1:]
    print(f"Loaded hmap_gcn_scale_{scale} from {gcn_path}")
    
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
        scales = discover_available_gcn_scales()
    if not scales:
        print("No GCN hmap scales discovered. Nothing to plot.")
        return None
    
    # Create figure
    num_scales = len(scales)
    fig = plt.figure(figsize=(6 * num_scales, 12))
    gs = gridspec.GridSpec(2, num_scales)
    
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
    
    discovered = discover_available_gcn_scales()
    print(f"Discovered GCN scales: {discovered}")

    # Generate and show the plot
    plot_multi_scale_grid_activations(
        scales=discovered if discovered else [0, 1, 2],
        save_path=os.path.join(output_dir, "grid_cell_activations_multi_scale.png"),
        show_plot=True,
        cmap='viridis',
    )
    
    print("Visualization complete!")
