# %%
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.colors as mcolors
from typing import Optional, List


# Custom function to calculate weighted mean
def weighted_mean(data, weights):
    return np.sum(data * weights) / np.sum(weights)


# Function to plot combined place cells with centers
def plot_combined_place_cells_with_centers(
    hmap_x, hmap_y, hmap_pcn, gridsize=80, specific_cells=None
):
    """
    Plots combined place cells with the place field centers overlayed.

    Args:
    - hmap_x: X coordinates.
    - hmap_y: Y coordinates.
    - hmap_pcn: Activation map (z-axis data).
    - gridsize: Size of the grid for the plot.
    - specific_cells: Optional list of specific cell indices to plot.
    """
    num_cells_to_plot = hmap_pcn.shape[1]

    if specific_cells is not None:
        num_cells_to_plot = len(specific_cells)
        cell_indices = specific_cells
    else:
        total_activation_per_cell = np.sum(hmap_pcn, axis=0)
        nonzero_activation_indices = np.where(total_activation_per_cell > 0)[0]
        cell_indices = nonzero_activation_indices

    # Define grid boundaries
    xmin, xmax = np.min(hmap_x), np.max(hmap_x)
    ymin, ymax = np.min(hmap_y), np.max(hmap_y)
    xedges = np.linspace(xmin, xmax, gridsize + 1)
    yedges = np.linspace(ymin, ymax, gridsize + 1)

    # Initialize storage arrays
    total_activations_per_bin = np.zeros((gridsize, gridsize, num_cells_to_plot))
    counts_per_bin = np.zeros((gridsize, gridsize, num_cells_to_plot))
    means = np.empty([num_cells_to_plot, 2])

    # Generate random vibrant colors for each cell using our utility function
    colors_rgb = generate_random_colors(num_cells_to_plot)

    for idx, cell_index in enumerate(cell_indices):
        activations = hmap_pcn[:, cell_index]
        mask = activations > 0
        if not np.any(mask):
            continue

        x, y, a = hmap_x[mask], hmap_y[mask], activations[mask]
        ix = np.clip(np.digitize(x, xedges) - 1, 0, gridsize - 1)
        iy = np.clip(np.digitize(y, yedges) - 1, 0, gridsize - 1)

        for i, j, activation in zip(ix, iy, a):
            total_activations_per_bin[i, j, idx] += activation
            counts_per_bin[i, j, idx] += 1

        # Compute weighted mean (center) for each cell
        try:
            x_mean = weighted_mean(hmap_x, weights=hmap_pcn[:, cell_index])
            y_mean = weighted_mean(hmap_y, weights=hmap_pcn[:, cell_index])
            means[idx] = x_mean, y_mean
        except:
            means[idx] = np.nan, np.nan

    # Compute mean activation per bin per cell
    nonzero_counts = counts_per_bin > 0
    mean_activation_per_bin = np.zeros_like(total_activations_per_bin)
    mean_activation_per_bin[nonzero_counts] = (
        total_activations_per_bin[nonzero_counts] / counts_per_bin[nonzero_counts]
    )
    max_mean_activation_per_bin = np.max(mean_activation_per_bin, axis=2)
    cell_with_max_activation = np.argmax(mean_activation_per_bin, axis=2)

    max_activation = np.max(max_mean_activation_per_bin)
    normalized_activation = (
        max_mean_activation_per_bin / max_activation if max_activation > 0 else 1
    )

    # Create an image array to store RGB values
    image = np.zeros((gridsize, gridsize, 3))

    for i in range(gridsize):
        for j in range(gridsize):
            activation_level = normalized_activation[i, j]
            if activation_level > 0:
                idx = cell_with_max_activation[i, j]
                color_rgb = colors_rgb[idx]
                adjusted_color = activation_level * np.array(color_rgb)
                image[i, j, :] = adjusted_color

    # Transpose for correct orientation
    image = np.transpose(image, (1, 0, 2))

    # Plot the overlay image
    plt.figure(figsize=(8, 8))
    extent = [xmin, xmax, ymin, ymax]
    plt.imshow(image, extent=extent, origin="lower")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title(f"Overlay of {num_cells_to_plot} Place Cells with Centers")

    # Overlay the center points
    valid_cells = ~np.isnan(means).any(axis=1)
    valid_means = means[valid_cells]
    plt.scatter(
        valid_means[:, 0],
        valid_means[:, 1],
        c="red",
        marker="x",
        s=100,
        label="Centers",
    )
    plt.legend()
    plt.show()


if __name__ == "__main__":
    from vis_utils import (
        load_hmaps,
        convert_xzy_hmaps,
        generate_random_colors,
    )

    # Load hmap data
    base_path = "webots/controllers/create3_base/"
    hmap_loc, hmap_pcn = load_hmaps(
        prefix=base_path, hmap_names=["hmap_loc", "hmap_pcn"]
    )
    hmap_x, hmap_z, hmap_y = convert_xzy_hmaps(hmap_loc)

    gridsize = 80

    specific_cells = None  # Replace with specific cell indices if needed
    plot_combined_place_cells_with_centers(
        hmap_x, hmap_y, hmap_pcn, gridsize, specific_cells
    )
