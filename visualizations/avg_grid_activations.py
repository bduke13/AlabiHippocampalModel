import numpy as np
import matplotlib.pyplot as plt
from vis_utils import load_hmaps, convert_xzy_hmaps

def plot_average_grid_activation(
    hmap_x,
    hmap_y,
    hmap_data,
    save_path=None,
):
    """
    Plots a hexbin plot for the average activation of all grid cells.
    
    Args:
        hmap_x: X coordinates of the grid.
        hmap_y: Y coordinates of the grid.
        hmap_data: Activation data for the cells.
        save_path: Path to save the plot (if None, plot is not saved).
    """
    # Calculate the average activation across all grid cells
    average_activations = np.mean(hmap_data, axis=1)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.suptitle("Average Grid Cell Activation", fontsize=16)
    
    # Use hexbin for the plot
    hb = ax.hexbin(
        hmap_x,
        hmap_y,
        C=average_activations,
        gridsize=100,
        reduce_C_function=np.mean,
        cmap='viridis',
        edgecolors="none",
    )
    
    # Add colorbar
    cbar = plt.colorbar(hb, ax=ax)
    cbar.set_label('Average Activation')
    
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Average Activation Across All Grid Cells")
    
    plt.tight_layout()
    
    # Save the plot if a path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=500)
        print(f"Saved average grid cell activation plot to {save_path}")
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    # Load grid cell data
    hmap_loc, hmap_grid = load_hmaps(hmap_names=["hmap_loc", "hmap_gcn"])
    hmap_x, hmap_z, hmap_y = convert_xzy_hmaps(hmap_loc)
    
    # Generate and show the average activation plot
    plot_average_grid_activation(hmap_x, hmap_y, hmap_grid)