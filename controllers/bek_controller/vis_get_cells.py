import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import json
import matplotlib.colors as mcolors

# Load the colors list
with open("colors.json", "r") as f:
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

# Choose top N cells based on total activation
N = 100  # Modify this number to select top N cells
total_activations = np.sum(hmap_z, axis=0)
cell_indices = np.argsort(total_activations)[-N:]

# Directory to save images
output_dir = "place_cell_images"
individual_dir = os.path.join(output_dir, "individual_cells")
combined_dir = os.path.join(output_dir, "combined_plots")

if not os.path.exists(individual_dir):
    os.makedirs(individual_dir)

if not os.path.exists(combined_dir):
    os.makedirs(combined_dir)

# Save individual cells
for idx, cell_index in enumerate(cell_indices):
    # Get activations for this cell
    activations = hmap_z[:, cell_index]

    # Positions
    x = hmap_x
    y = hmap_y

    # Color for this cell
    color_rgb = colors_rgb[cell_index]

    # Create a new figure for each cell
    fig, ax = plt.subplots(figsize=(4, 4))

    # Create a hexbin plot
    hb = ax.hexbin(
        x,
        y,
        C=activations,
        gridsize=50,
        reduce_C_function=np.mean,
        cmap=None,
        edgecolors="none",
    )

    # Get aggregated activations per bin
    counts = hb.get_array()

    # Normalize counts for alpha values
    max_count = counts.max()
    if max_count > 0:
        counts_normalized = counts / max_count
    else:
        counts_normalized = counts

    # Create RGBA colors
    rgba_colors = np.zeros((len(counts), 4))
    rgba_colors[:, 0:3] = color_rgb  # Set RGB values
    rgba_colors[:, 3] = counts_normalized  # Set alpha based on activation

    # Set the facecolors of the hexbin collection
    hb.set_facecolors(rgba_colors)

    # Customize each plot
    ax.set_title(f"Cell {cell_index}", fontsize=10)
    ax.axis("off")

    # Save each cell image to individual_cells directory
    output_path = os.path.join(individual_dir, f"cell_{cell_index}.jpg")
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Cell {cell_index} image saved to {output_path}")

# Now create a 5x5 grid for combined plots and save it
grid_size = 5  # Adjust based on how many cells you want per row and column
fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))

for idx, cell_index in enumerate(cell_indices[: grid_size * grid_size]):
    row, col = divmod(idx, grid_size)

    # Get activations for this cell
    activations = hmap_z[:, cell_index]

    # Positions
    x = hmap_x
    y = hmap_y

    # Color for this cell
    color_rgb = colors_rgb[cell_index]

    # Create a hexbin plot in the grid
    hb = axes[row, col].hexbin(
        x,
        y,
        C=activations,
        gridsize=50,
        reduce_C_function=np.mean,
        cmap=None,
        edgecolors="none",
    )

    # Get aggregated activations per bin
    counts = hb.get_array()

    # Normalize counts for alpha values
    max_count = counts.max()
    if max_count > 0:
        counts_normalized = counts / max_count
    else:
        counts_normalized = counts

    # Create RGBA colors
    rgba_colors = np.zeros((len(counts), 4))
    rgba_colors[:, 0:3] = color_rgb  # Set RGB values
    rgba_colors[:, 3] = counts_normalized  # Set alpha based on activation

    # Set the facecolors of the hexbin collection
    hb.set_facecolors(rgba_colors)

    # Customize each subplot
    axes[row, col].set_title(f"Cell {cell_index}", fontsize=10)
    axes[row, col].axis("off")

# Save the combined plot to combined_plots directory
output_path_combined = os.path.join(combined_dir, "top_N_cells_grid.jpg")
plt.savefig(output_path_combined, dpi=300)
plt.close()

print(f"Grid of top {N} cells saved to {output_path_combined}")
