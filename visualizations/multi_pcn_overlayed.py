# %%
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from typing import Optional, Dict, List, Tuple
import matplotlib.gridspec as gridspec
from pathlib import Path

# Get the project root directory
project_root = Path(__file__).resolve().parent.parent  # Adjust this if needed
sys.path.append(str(project_root))

# Import from vis_utils
from vis_utils import (
    convert_xzy_hmaps,
    generate_random_colors,
    OUTPUT_DIR,
    CONTROLLER_PATH_PREFIX,
    CONTROLLER_NAME,
    WORLD_NAME
)

def plot_scale_cells(
    hmap_pcn: np.ndarray,
    hmap_x: np.ndarray,
    hmap_y: np.ndarray,
    scale: int,
    gridsize: int = 200,
    num_cells_to_sample: Optional[int] = 10,
    show_plot: bool = False,
    ax = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Plot place cells for a specific scale.
    
    Args:
        hmap_pcn: Place cell network activation data
        hmap_x: X coordinates
        hmap_y: Y coordinates
        scale: Scale identifier (for title)
        gridsize: Resolution of the grid
        num_cells_to_sample: Number of random cells to sample
        show_plot: Whether to show the plot
        ax: Matplotlib axis to plot on
        
    Returns:
        Tuple of (image array, cell indices used)
    """
    # Calculate total activation per cell
    total_activation_per_cell = np.sum(hmap_pcn, axis=0)

    # Get indices of cells with non-zero activation
    nonzero_activation_indices = np.where(total_activation_per_cell > 0)[0]

    # If num_cells_to_sample is None, use all cells with non-zero activation
    if num_cells_to_sample is None:
        num_cells_to_plot = len(nonzero_activation_indices)
    else:
        num_cells_to_plot = min(num_cells_to_sample, len(nonzero_activation_indices))
    
    print(f"Scale {scale}: Plotting {num_cells_to_plot} cells out of {len(nonzero_activation_indices)} active cells")

    # Randomly select the specified number of place cells with non-zero activation
    cell_indices = np.random.choice(
        nonzero_activation_indices, size=num_cells_to_plot, replace=False
    )

    # Define the grid boundaries and resolution
    xmin, xmax = np.min(hmap_x), np.max(hmap_x)
    ymin, ymax = np.min(hmap_y), np.max(hmap_y)

    # Create grid edges
    xedges = np.linspace(xmin, xmax, gridsize + 1)
    yedges = np.linspace(ymin, ymax, gridsize + 1)

    # Initialize arrays to store total activations and counts per bin per cell
    total_activations_per_bin = np.zeros((gridsize, gridsize, num_cells_to_plot))
    counts_per_bin = np.zeros((gridsize, gridsize, num_cells_to_plot))

    # Process each randomly selected cell
    for idx, cell_index in enumerate(cell_indices):
        # Get activations for this cell
        activations = hmap_pcn[:, cell_index]

        # Positions where activation is greater than zero
        mask = activations > 0
        if not np.any(mask):
            continue  # Skip cells with zero activation
        x, y, a = hmap_x[mask], hmap_y[mask], activations[mask]

        # Bin the positions
        ix = np.digitize(x, xedges) - 1  # indices start from 0
        iy = np.digitize(y, yedges) - 1

        # Clip indices to valid range
        ix = np.clip(ix, 0, gridsize - 1)
        iy = np.clip(iy, 0, gridsize - 1)

        # For each bin, accumulate activations and counts
        for i, j, activation in zip(ix, iy, a):
            total_activations_per_bin[i, j, idx] += activation
            counts_per_bin[i, j, idx] += 1

    # Compute mean activation per bin per cell, handling division by zero
    mean_activation_per_bin = np.zeros_like(total_activations_per_bin)
    nonzero_counts = counts_per_bin > 0
    mean_activation_per_bin[nonzero_counts] = (
        total_activations_per_bin[nonzero_counts] / counts_per_bin[nonzero_counts]
    )

    # For each bin, find the cell with the maximum mean activation
    max_mean_activation_per_bin = np.max(mean_activation_per_bin, axis=2)
    cell_with_max_activation = np.argmax(mean_activation_per_bin, axis=2)

    # Normalize activations to [0, 1] for scaling colors
    max_activation = np.max(max_mean_activation_per_bin)
    max_activation = (
        max_activation if max_activation > 0 else 1
    )  # Avoid division by zero
    normalized_activation = max_mean_activation_per_bin / max_activation

    # Generate random vibrant colors for each cell
    colors_rgb = generate_random_colors(num_cells_to_plot)

    # Now, create an image array to store RGB values
    image = np.zeros((gridsize, gridsize, 3))

    # Assign colors to bins
    for i in range(gridsize):
        for j in range(gridsize):
            activation_level = normalized_activation[i, j]
            if activation_level > 0:
                idx = cell_with_max_activation[i, j]
                adjusted_color = activation_level * np.array(colors_rgb[idx])
                image[i, j, :] = adjusted_color

    # Transpose the image because imshow expects the first axis to be the y-axis
    image = np.transpose(image, (1, 0, 2))

    # Plot the overlay image if an axis is provided
    if ax is not None:
        extent = [xmin, xmax, ymin, ymax]
        ax.imshow(image, extent=extent, origin="lower")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(f"Scale {scale}: {num_cells_to_plot} Place Cells")

    return image, cell_indices

def plot_all_multi_scale_cells(
    scales: List[int] = None,
    gridsize: int = 200,
    num_cells_to_sample: Optional[int] = 10,
    show_plot: bool = True,
    save_path: Optional[str] = None,
    plot_all_cells_too: bool = True
):
    """
    Plot place cells for multiple scales.
    
    Args:
        scales: List of scales to plot
        gridsize: Resolution of the grid
        num_cells_to_sample: Number of random cells to sample
        show_plot: Whether to show the plot
        save_path: Path to save the plot
        plot_all_cells_too: Whether to also plot all cells
    """
    if scales is None:
        scales = [0, 1, 2]
    
    # Load data
    hmap_loc, hmap_pcn_dict = load_multi_scale_hmaps(scales)
    hmap_x, hmap_z, hmap_y = convert_xzy_hmaps(hmap_loc)
    
    # Determine number of subplot rows
    num_rows = 1
    if plot_all_cells_too:
        num_rows = 2
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 5 * num_rows))
    gs = gridspec.GridSpec(num_rows, len(scales))
    
    # Plot sampled cells for each scale
    for i, scale in enumerate(scales):
        ax = plt.subplot(gs[0, i])
        plot_scale_cells(
            hmap_pcn=hmap_pcn_dict[scale],
            hmap_x=hmap_x,
            hmap_y=hmap_y,
            scale=scale,
            gridsize=gridsize,
            num_cells_to_sample=num_cells_to_sample,
            ax=ax
        )
    
    # Plot all cells for each scale if requested
    if plot_all_cells_too:
        for i, scale in enumerate(scales):
            ax = plt.subplot(gs[1, i])
            # Set num_cells_to_sample to None to plot all cells
            plot_scale_cells(
                hmap_pcn=hmap_pcn_dict[scale],
                hmap_x=hmap_x,
                hmap_y=hmap_y,
                scale=scale,
                gridsize=gridsize,
                num_cells_to_sample=None,  # Plot all cells
                ax=ax
            )
    
    plt.tight_layout()
    
    # Save the plot if a save path is provided
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")
    
    # Show the plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()

def save_multi_scale_plots(
    base_filename: str = "multi_scale_pcn",
    scales: List[int] = None,
    gridsize: int = 200,
    num_cells_to_sample: int = 10
):
    """
    Generate and save multi-scale place cell plots.
    
    Args:
        base_filename: Base name for output files
        scales: List of scales to plot
        gridsize: Resolution of the grid
        num_cells_to_sample: Number of random cells to sample
    """
    if scales is None:
        scales = [0, 1, 2]
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate path for the multi-scale plot
    save_path = os.path.join(OUTPUT_DIR, f"{base_filename}.png")
    
    # Generate and save the plot
    plot_all_multi_scale_cells(
        scales=scales,
        gridsize=gridsize,
        num_cells_to_sample=num_cells_to_sample,
        show_plot=False,
        save_path=save_path,
        plot_all_cells_too=True
    )
    
    print(f"Multi-scale plots saved to {save_path}")

# Add load_multi_scale_hmaps directly here to avoid import issues
def load_multi_scale_hmaps(scales=None):
    """
    Load history map (hmap) data for multiple scales.
    
    Args:
        scales (List[int], optional): List of scales to load. Defaults to [0, 1, 2].
        
    Returns:
        tuple: (hmap_loc, dict of hmap_pcn by scale)
    """
    import pickle
    import os
    import numpy as np
    
    if scales is None:
        scales = [0, 1, 2]
    
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
    
    # Load PCN data for each scale
    hmap_pcn_dict = {}
    for scale in scales:
        hmap_file = f"hmap_pcn_scale_{scale}.pkl"
        file_path = os.path.join(hmap_directory, hmap_file)
        try:
            with open(file_path, "rb") as f:
                scale_data = np.array(pickle.load(f))
                # remove first element
                scale_data = scale_data[1:]
                hmap_pcn_dict[scale] = scale_data
                print(f"Loaded hmap_pcn_scale_{scale} from {file_path}")
        except FileNotFoundError:
            print(f"Warning: {file_path} not found. Skipping scale {scale}.")
            continue
    
    return hmap_loc, hmap_pcn_dict

# Run immediately when file is executed
if __name__ == "__main__":
    print("Starting multi-scale place cell visualization...")
    
    # Show plot directly (this will block until plot is closed)
    plot_all_multi_scale_cells(
        scales=[0, 1, 2],
        gridsize=200,
        num_cells_to_sample=10,
        show_plot=True,
        save_path=os.path.join(OUTPUT_DIR, "multi_scale_pcn.png"),
        plot_all_cells_too=True
    )
    
    print("Visualization complete!")