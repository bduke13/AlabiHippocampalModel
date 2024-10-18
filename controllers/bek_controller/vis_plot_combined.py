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
N = 200  # Modify this number to select top N cells
total_activations = np.sum(hmap_z, axis=0)

# Set a flag to sort cells by activation or just use them in order
top_activations = True

if top_activations:
    cell_indices = np.argsort(total_activations)[-N:]
else:
    cell_indices = np.arange(N)  # Just use first N cells

# Directory to save combined plots
combined_dir = "place_cell_images/combined_plots"
if not os.path.exists(combined_dir):
    os.makedirs(combined_dir)

# Set grid size
grid_size = 5  # 5x5 grid for subplots
cells_per_grid = grid_size * grid_size

# Loop over cells and save combined images in batches of 5x5
for i in range(0, len(cell_indices), cells_per_grid):
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))

    for idx in range(cells_per_grid):
        row, col = divmod(idx, grid_size)
        if i + idx < len(cell_indices):
            cell_index = cell_indices[i + idx]

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
        else:
            # Turn off axes for empty subplots
            axes[row, col].axis("off")

    # Save the combined grid plot
    output_path_combined = os.path.join(
        combined_dir, f"cells_{i+1}_to_{i+cells_per_grid}.jpg"
    )
    plt.savefig(output_path_combined, dpi=300)
    plt.close()

    print(
        f"Grid of cells {i+1} to {min(i+cells_per_grid, len(cell_indices))} saved to {output_path_combined}"
    )
