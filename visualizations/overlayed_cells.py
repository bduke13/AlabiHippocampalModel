# %%
from matplotlib.cbook import file_requires_unicode
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

print(os.listdir())


from analysis_utils import load_hmaps


gridsize = 80
num_cells_to_sample = 150  # Set the number of cells you want to plot


base_path = "webots/controllers/create3_base/"
# Load hmap data
hmap_loc, hmap_pcn = load_hmaps(prefix=base_path, hmap_names=["hmap_loc", "hmap_pcn"])
hmap_x = hmap_loc[:, 0]
hmap_y = hmap_loc[:, 2]

# %%
# Calculate total activation per cell
total_activation_per_cell = np.sum(hmap_pcn, axis=0)

# Get indices of cells with non-zero activation
nonzero_activation_indices = np.where(total_activation_per_cell > 0)[0]

# Set number of cells to plot based on available cells
num_cells_to_plot = min(num_cells_to_sample, len(nonzero_activation_indices))
print(
    f"Plotting {num_cells_to_plot} cells out of {len(nonzero_activation_indices)} active cells"
)

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
    activations = hmap_pcn[:, cell_index]

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

# Create a list of distinct colors for each cell
colors_rgb = plt.cm.rainbow(np.linspace(0, 1, num_cells_to_plot))[:, :3]

# Now, create an image array to store RGB values
image = np.zeros((gridsize, gridsize, 3))

# Assign colors to bins
for i in range(gridsize):
    for j in range(gridsize):
        activation_level = normalized_activation[i, j]
        if activation_level > 0:
            idx = cell_with_max_activation[i, j]
            # color_rgb = colors_rgb[idx]    # This is correct
            color_rgb = colors_rgb[idx]

            adjusted_color = activation_level * np.array(color_rgb)
            image[i, j, :] = adjusted_color  # else:
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

# Ensure the plot displays properly
plt.show()


if __name__ == "main":
    pass
