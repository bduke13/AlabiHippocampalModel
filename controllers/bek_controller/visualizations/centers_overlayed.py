import numpy as np
import matplotlib.pyplot as plt
import pickle
import json
import os
import matplotlib.colors as mcolors


# Custom function to calculate weighted mean
def weighted_mean(data, weights):
    return np.sum(data * weights) / np.sum(weights)


# Part 1: Combined Plot of All Place Cells
def plot_combined_place_cells_with_centers(
    hmap_x, hmap_y, hmap_z, gridsize=80, colors_rgb=None, specific_cells=None
):
    """
    Plots combined place cells with the place field centers overlayed.

    Args:
    - hmap_x: X coordinates.
    - hmap_y: Y coordinates.
    - hmap_z: Activation map (z-axis data).
    - gridsize: Size of the hexagonal grid for the plot.
    - colors_rgb: List of RGB colors for each place cell.
    - specific_cells: Optional list of specific cell indices to plot.
    """
    num_cells_to_plot = hmap_z.shape[1]

    # If specific cells are provided, limit the plot to those cells
    if specific_cells is not None:
        num_cells_to_plot = len(specific_cells)
        cell_indices = specific_cells
    else:
        total_activation_per_cell = np.sum(hmap_z, axis=0)
        nonzero_activation_indices = np.where(total_activation_per_cell > 0)[0]
        cell_indices = nonzero_activation_indices

    # Create grid boundaries and resolution
    xmin, xmax = np.min(hmap_x), np.max(hmap_x)
    ymin, ymax = np.min(hmap_y), np.max(hmap_y)
    xedges = np.linspace(xmin, xmax, gridsize + 1)
    yedges = np.linspace(ymin, ymax, gridsize + 1)

    total_activations_per_bin = np.zeros((gridsize, gridsize, num_cells_to_plot))
    counts_per_bin = np.zeros((gridsize, gridsize, num_cells_to_plot))

    # Initialize to store center points (weighted mean)
    means = np.empty([num_cells_to_plot, 2])

    # Process each cell
    for idx, cell_index in enumerate(cell_indices):
        activations = hmap_z[:, cell_index]

        mask = activations > 0
        if not np.any(mask):
            continue  # Skip cells with zero activation

        x = hmap_x[mask]
        y = hmap_y[mask]
        a = activations[mask]

        # Bin the positions
        ix = np.digitize(x, xedges) - 1
        iy = np.digitize(y, yedges) - 1
        ix = np.clip(ix, 0, gridsize - 1)
        iy = np.clip(iy, 0, gridsize - 1)

        for i, j, activation in zip(ix, iy, a):
            total_activations_per_bin[i, j, idx] += activation
            counts_per_bin[i, j, idx] += 1

        # Calculate the weighted mean (center) for the cell
        try:
            x_mean = weighted_mean(hmap_x, weights=hmap_z[:, cell_index])
            y_mean = weighted_mean(hmap_y, weights=hmap_z[:, cell_index])
            means[idx] = x_mean, y_mean
        except:
            means[idx] = np.nan, np.nan  # Handle case with no firing

    # Compute mean activation per bin and normalize activations
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

    # Assign colors to bins
    for i in range(gridsize):
        for j in range(gridsize):
            activation_level = normalized_activation[i, j]
            if activation_level > 0:
                idx = cell_with_max_activation[i, j]
                color_rgb = colors_rgb[cell_indices[idx]]
                adjusted_color = activation_level * np.array(color_rgb)
                image[i, j, :] = adjusted_color

    # Transpose the image for correct orientation
    image = np.transpose(image, (1, 0, 2))

    # Plot the overlay image
    plt.figure(figsize=(8, 8))
    extent = [xmin, xmax, ymin, ymax]
    plt.imshow(image, extent=extent, origin="lower")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title(f"Overlay of {num_cells_to_plot} Place Cells with Fading Colors")

    # Overlay the center points for each valid cell
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

    # Show the combined plot
    plt.show()


if __name__ == "__main__":
    gridsize = 80

    # Load the colors list
    with open("c:/Users/jacks/OneDrive/Desktop/NeuroNav/AlabiHippocampalModel/controllers/bek_controller/visualizations/colors.json", "r") as f:
        colors = json.load(f)

    # Convert hex colors to RGB format
    colors_rgb = [mcolors.to_rgb(c) for c in colors]

    # Load hmap data
    with open("hmap_x.pkl", "rb") as f:
        hmap_x = np.array(pickle.load(f))
    with open("hmap_y.pkl", "rb") as f:
        hmap_y = np.array(pickle.load(f))
    with open("hmap_z.pkl", "rb") as f:
        hmap_z = np.asarray(pickle.load(f))

    specific_cells = None  # Replace with specific cells or set to None to plot all

    # Generate the combined plot with centers overlayed
    plot_combined_place_cells_with_centers(
        hmap_x, hmap_y, hmap_z, gridsize, colors_rgb, specific_cells
    )
