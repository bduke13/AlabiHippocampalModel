import numpy as np
import matplotlib.pyplot as plt
import pickle
import json
import matplotlib.colors as mcolors
import itertools

gridsize = 150

# Load the colors list
with open(r"controllers/bek_controller/visualizations/colors.json", "r") as f:
    colors = json.load(f)

# Convert hex colors to RGB format
colors_rgb = [mcolors.to_rgb(c) for c in colors]

# Load hmap data for small and large scales
base_path = r"controllers/bek_controller/pkl/hmaps/world0_20x20-obstacles"
with open(f"{base_path}/hmap_x.pkl", "rb") as f:
    hmap_x = np.array(pickle.load(f))
with open(f"{base_path}/hmap_y.pkl", "rb") as f:
    hmap_y = np.array(pickle.load(f))
with open(f"{base_path}/hmap_z_small.pkl", "rb") as f:
    hmap_z_small = np.asarray(pickle.load(f))
with open(f"{base_path}/hmap_z_large.pkl", "rb") as f:
    hmap_z_large = np.asarray(pickle.load(f))

def plot_hmap_overlay(hmap_z, title, gridsize, hmap_x, hmap_y, colors_rgb):
    # Total number of place cells based on hmap_z's shape
    num_cells_to_plot = hmap_z.shape[1]

    # Calculate total activation per cell
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

    # Ensure colors are sufficient
    if len(colors_rgb) < num_cells_to_plot:
        print("Insufficient colors provided. Cycling through available colors.")
        # Repeat colors as necessary to match the number of cells
        colors_rgb = list(itertools.islice(itertools.cycle(colors_rgb), num_cells_to_plot))

    # Define the grid boundaries and resolution
    xmin = np.min(hmap_x)
    xmax = np.max(hmap_x)
    ymin = np.min(hmap_y)
    ymax = np.max(hmap_y)

    # Create grid edges
    xedges = np.linspace(xmin, xmax, gridsize + 1)
    yedges = np.linspace(ymin, ymax, gridsize + 1)

    # Initialize arrays to store total activations and counts per bin per cell
    total_activations_per_bin = np.zeros((gridsize, gridsize, num_cells_to_plot))
    counts_per_bin = np.zeros((gridsize, gridsize, num_cells_to_plot))

    # Process each randomly selected cell
    for idx, cell_index in enumerate(cell_indices):
        # Get activations for this cell
        activations = hmap_z[:, cell_index]

        # Positions where activation is greater than zero
        mask = activations > 0
        if not np.any(mask):
            continue
        x = hmap_x[mask]
        y = hmap_y[mask]
        a = activations[mask]

        # Bin the positions
        ix = np.digitize(x, xedges) - 1
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
        max_activation = 1
    normalized_activation = max_mean_activation_per_bin / max_activation

    # Create an image array to store RGB values
    image = np.zeros((gridsize, gridsize, 3))

    # Assign colors to bins
    for i in range(gridsize):
        for j in range(gridsize):
            activation_level = normalized_activation[i, j]
            if activation_level > 0:
                idx = cell_with_max_activation[i, j]
                color_rgb = colors_rgb[cell_indices[idx] % len(colors_rgb)]
                adjusted_color = activation_level * np.array(color_rgb)
                image[i, j, :] = adjusted_color

    # Transpose the image for proper display
    image = np.transpose(image, (1, 0, 2))

    # Plot the overlay image
    plt.imshow(image, extent=[xmin, xmax, ymin, ymax], origin="lower")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title(title)


# Plot both scales side by side
plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plot_hmap_overlay(hmap_z_small, "Small Scale Place Cells", gridsize, hmap_x, hmap_y, colors_rgb)
plt.subplot(1, 2, 2)
plot_hmap_overlay(hmap_z_large, "Large Scale Place Cells", gridsize, hmap_x, hmap_y, colors_rgb)
plt.tight_layout()
plt.show()
