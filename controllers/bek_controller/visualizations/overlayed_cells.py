# %%
import numpy as np
import matplotlib.pyplot as plt
import pickle
import json
import matplotlib.colors as mcolors
from pathlib import Path


def plot_overlayed_cells(
    hmap_x,
    hmap_y,
    hmap_z,
    colors_path=None,
    gridsize=50,
    output_dir=None,
    suffix="",
):
    """
    Create an overlay plot of place cells with fading colors.

    Args:
        hmap_x_path (str): Path to the x coordinates pickle file
        hmap_y_path (str): Path to the y coordinates pickle file
        hmap_z_path (str): Path to the PCN data pickle file
        colors_path (str): Path to the colors JSON file
        gridsize (int): Size of the grid for binning
    """
    # Total number of place cells based on hmap_z's shape (dimension 1)
    num_cells_to_plot = hmap_z.shape[1]

    # Generate or load colors
    if colors_path is None:
        # Generate random vibrant colors with higher saturation
        colors_rgb = []
        for _ in range(num_cells_to_plot):
            # Generate colors with higher minimum values for more vibrancy
            color = np.random.uniform(0.6, 1, 3)
            # Ensure at least one channel is very bright
            bright_channel = np.random.randint(0, 3)
            color[bright_channel] = np.random.uniform(0.9, 1.0)
            colors_rgb.append(color)
    else:
        # Load the colors list from file
        with open(colors_path, "r") as f:
            colors = json.load(f)
        # Convert hex colors to RGB format
        colors_rgb = [mcolors.to_rgb(c) for c in colors]

    # Generate or load colors for all cells
    if colors_path is None:
        # Generate random bright colors for better visibility
        colors_rgb = [np.random.uniform(0.4, 1, 3) for _ in range(num_cells_to_plot)]
    else:
        # Load the colors list from file
        with open(colors_path, "r") as f:
            colors = json.load(f)
        # Convert hex colors to RGB format
        colors_rgb = [mcolors.to_rgb(c) for c in colors]

    # Calculate total activation per cell for cell selection
    total_activation_per_cell = np.sum(hmap_z, axis=0)

    # Get indices of cells with non-zero activation
    nonzero_activation_indices = np.where(total_activation_per_cell > 0)[0]

    # Check if there are enough cells with non-zero activation
    if len(nonzero_activation_indices) < num_cells_to_plot:
        print(f"{len(nonzero_activation_indices)} cells have non-zero activation.")
        num_cells_to_plot = len(nonzero_activation_indices)

    # Randomly select the specified number of place cells with non-zero activation
    cell_indices = np.random.choice(
        nonzero_activation_indices, size=num_cells_to_plot, replace=False
    )

    # --- Part 1: Overlay Plot of All Cells with Fading Colors ---

    # Define the grid boundaries and resolution
    xmin = np.min(hmap_x)
    xmax = np.max(hmap_x)
    ymin = np.min(hmap_y)
    ymax = np.max(hmap_y)

    # Create grid edges
    xedges = np.linspace(xmin, xmax, gridsize + 1)
    yedges = np.linspace(ymin, ymax, gridsize + 1)

    # Initialize arrays to store total activations and counts per bin per cell
    # Shape: (gridsize, gridsize, num_cells_to_plot)
    total_activations_per_bin = np.zeros((gridsize, gridsize, num_cells_to_plot))
    counts_per_bin = np.zeros((gridsize, gridsize, num_cells_to_plot))

    # Process each randomly selected cell
    for idx, cell_index in enumerate(cell_indices):
        # Get activations for this cell
        activations = hmap_z[:, cell_index]

        # Positions where activation is greater than zero
        mask = activations > 0
        if not np.any(mask):
            continue  # Skip cells with zero activation (shouldn't occur with the updated selection)
        x = hmap_x[mask]
        y = hmap_y[mask]
        a = activations[mask]

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
    if max_activation == 0:
        max_activation = 1  # To avoid division by zero
    normalized_activation = max_mean_activation_per_bin / max_activation

    # Now, create an image array to store RGB values
    image = np.zeros((gridsize, gridsize, 3))

    # Assign colors to bins
    for i in range(gridsize):
        for j in range(gridsize):
            activation_level = normalized_activation[i, j]
            if activation_level > 0:
                idx = cell_with_max_activation[i, j]
                color_rgb = colors_rgb[cell_indices[idx]]
                # Enhance color vibrancy:
                # 1. Increase minimum brightness to 0.3
                # 2. Apply power function to activation level to make mid-levels brighter
                # 3. Scale the remaining range from 0.3 to 1.0
                enhanced_activation = 0.3 + 0.7 * (activation_level**0.5)
                adjusted_color = enhanced_activation * np.array(color_rgb)
                # Ensure we don't exceed 1.0
                adjusted_color = np.clip(adjusted_color, 0, 1)
                image[i, j, :] = adjusted_color
            # else:
            # No activation in this bin, leave as black (0,0,0)

    # Transpose the image because imshow expects the first axis to be the y-axis
    image = np.transpose(image, (1, 0, 2))

    # Plot the overlay image
    plt.figure(figsize=(8, 8))
    extent = [xmin, xmax, ymin, ymax]
    plt.imshow(image, extent=extent, origin="lower")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title(f"Overlay of {num_cells_to_plot} Place Cells with Fading Colors")

    fig = plt.gcf()

    if output_dir is not None:
        # Convert to Path object and create directory if it doesn't exist
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Construct filename with suffix
        filename = f"overlayed_cells{suffix}.png"
        output_path = output_dir / filename

        # Save the plot
        fig.savefig(output_path)

    # Always show the plot
    plt.show()
    plt.close(fig)

    return fig  # Return the figure object


# %%
if __name__ == "__main__":
    from controllers.bek_controller.visualizations.analysis_utils import *
    from controllers.bek_controller.visualizations.path_view import plot_trajectory

    data_path = "controllers/bek_controller/IJCNN/2D_250/upright/"
    hmap_x, hmap_y, hmap_z = load_hmaps(data_path)
    # %%
    plot_trajectory(
        hmap_x,
        hmap_y,
    )
    # %%
    plot_overlayed_cells(hmap_x, hmap_y, hmap_z, gridsize=80, output_dir=data_path)
