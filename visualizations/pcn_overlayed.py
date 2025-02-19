# %%
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from typing import Optional

# Hardcoded world name
WORLD_NAME = "10x10"

def load_pickle(file_path):
    """Load a pickle file from the given path."""
    with open(file_path, "rb") as f:
        return pickle.load(f)

def load_hmaps_from_world():
    """Load hmap data based on the hardcoded world name."""
    base_path = f"webots/controllers/create3_base/pkl/{WORLD_NAME}/hmaps"
    hmap_loc = load_pickle(os.path.join(base_path, "hmap_loc.pkl"))
    hmap_pcn = load_pickle(os.path.join(base_path, "hmap_pcn.pkl"))
    return hmap_loc, hmap_pcn

def convert_xzy_hmaps(hmap_loc: np.ndarray):
    """
    Convert hmap location array to separate x, z, and y components.
    Note: In the new model, getSFVec3f() returns [x, y, z] where y is the vertical coordinate.
    For plotting the horizontal plane, we use x and z.
    """
    hmap_x = hmap_loc[:, 0]
    hmap_z = hmap_loc[:, 2]  # use the third column as the horizontal coordinate (z)
    hmap_y = hmap_loc[:, 1]  # vertical, if needed later
    return hmap_x, hmap_z, hmap_y

def generate_random_colors(num_colors):
    """Generate an array of random vibrant colors in RGB format."""
    return np.random.rand(num_colors, 3)

def plot_overlayed_cells(
    hmap_pcn: np.ndarray,
    hmap_x: np.ndarray,
    hmap_z: np.ndarray,   # renamed parameter to reflect horizontal z
    gridsize: int = 100,
    num_cells_to_sample: Optional[int] = None,
    show_plot: bool = True,
    save_path: Optional[str] = None,
):
    # Calculate total activation per cell
    total_activation_per_cell = np.sum(hmap_pcn, axis=0)

    # Get indices of cells with non-zero activation
    nonzero_activation_indices = np.where(total_activation_per_cell > 0)[0]

    # Determine how many cells to plot
    if num_cells_to_sample is None:
        num_cells_to_plot = len(nonzero_activation_indices)
    else:
        num_cells_to_plot = min(num_cells_to_sample, len(nonzero_activation_indices))
    print(
        f"Plotting {num_cells_to_plot} cells out of {len(nonzero_activation_indices)} active cells"
    )

    # Randomly select the specified number of place cells with non-zero activation
    cell_indices = np.random.choice(
        nonzero_activation_indices, size=num_cells_to_plot, replace=False
    )

    # Define grid boundaries and resolution using x and z coordinates
    xmin, xmax = np.min(hmap_x), np.max(hmap_x)
    zmin, zmax = np.min(hmap_z), np.max(hmap_z)

    # Create grid edges
    xedges = np.linspace(xmin, xmax, gridsize + 1)
    zedges = np.linspace(zmin, zmax, gridsize + 1)

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
        x = hmap_x[mask]
        z = hmap_z[mask]
        a = activations[mask]

        # Bin the positions (using x and z)
        ix = np.digitize(x, xedges) - 1  # indices start from 0
        iz = np.digitize(z, zedges) - 1

        # Clip indices to valid range
        ix = np.clip(ix, 0, gridsize - 1)
        iz = np.clip(iz, 0, gridsize - 1)

        # For each bin, accumulate activations and counts
        for i, j, activation in zip(ix, iz, a):
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
    max_activation = max_activation if max_activation > 0 else 1  # Avoid division by zero
    normalized_activation = max_mean_activation_per_bin / max_activation

    # Generate random vibrant colors for each cell
    colors_rgb = generate_random_colors(num_cells_to_plot)

    # Create an image array to store RGB values
    image = np.zeros((gridsize, gridsize, 3))

    # Assign colors to bins
    for i in range(gridsize):
        for j in range(gridsize):
            activation_level = normalized_activation[i, j]
            if activation_level > 0:
                idx = cell_with_max_activation[i, j]
                adjusted_color = activation_level * np.array(colors_rgb[idx])
                image[i, j, :] = adjusted_color

    # Transpose the image for correct orientation (imshow expects first axis as y-axis)
    image = np.transpose(image, (1, 0, 2))

    # Plot the overlay image using x and z coordinates
    plt.figure(figsize=(8, 8))
    extent = [xmin, xmax, zmin, zmax]
    plt.imshow(image, extent=extent, origin="lower")
    plt.xlabel("X Coordinate")
    plt.ylabel("Z Coordinate")
    plt.title(f"Overlay of {num_cells_to_plot} Place Cells with Fading Colors")

    # Save the plot if a save path is provided
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")

    # Show or close the plot
    if show_plot:
        plt.show()
    else:
        plt.close()

if __name__ == "__main__":
    from vis_utils import load_hmaps, convert_xzy_hmaps

    # Load hmap data from the hardcoded world name
    hmap_loc, hmap_pcn = load_hmaps()
    # Updated conversion: hmap_x and hmap_z now represent the horizontal plane
    hmap_x, hmap_z, _ = convert_xzy_hmaps(hmap_loc)

    # Example usage with updated parameters (using hmap_x and hmap_z)
    plot_overlayed_cells(
        hmap_pcn=hmap_pcn,
        hmap_x=hmap_x,
        hmap_y=hmap_y,
        gridsize=60,
        num_cells_to_sample=None,  # Use all cells with non-zero activation
        show_plot=True,
        save_path=None,
    )
