
import numpy as np
import matplotlib.pyplot as plt
import pickle
import json
import os
import matplotlib.colors as mcolors

# Create the directory if it doesn't exist
output_dir = 'place_cell_images/'
os.makedirs(output_dir, exist_ok=True)

# Load the colors list
with open('colors.json', 'r') as f:
    colors = json.load(f)

# Convert hex colors to RGB format
colors_rgb = [mcolors.to_rgb(c) for c in colors]

# Load hmap data
with open('hmap_x.pkl', 'rb') as f:
    hmap_x = np.array(pickle.load(f))
with open('hmap_y.pkl', 'rb') as f:
    hmap_y = np.array(pickle.load(f))
with open('hmap_z.pkl', 'rb') as f:
    hmap_z = np.asarray(pickle.load(f))

# Choose cells to plot
num_place_cells = hmap_z.shape[1]

# Option 1: Sample 20 random cells
# np.random.seed(0)
# cell_indices = np.random.choice(num_place_cells, size=20, replace=False)

# Option 2: Select top 100 cells based on total activation
total_activations = np.sum(hmap_z, axis=0)
cell_indices = np.argsort(total_activations)[-100:]

for cell_index in cell_indices:
    # Get activations for this cell
    activations = hmap_z[:, cell_index]

    # Positions
    x = hmap_x
    y = hmap_y

    # Color for this cell
    color_rgb = colors_rgb[cell_index % len(colors_rgb)]  # Use modulo to avoid index out of range

    # Create a hexbin plot
    plt.figure(figsize=(8, 6))
    # Use hexbin, specify C as activations
    hb = plt.hexbin(
        x, y, C=activations, gridsize=50, reduce_C_function=np.mean,
        cmap=None, edgecolors='none'
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

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(f'Place Cell {cell_index} Activation')

    # Save the plot to the specified folder
    file_path = os.path.join(output_dir, f'place_cell_{cell_index}.jpg')
    plt.savefig(file_path)

    # Close the figure to save memory
    plt.close()

print(f"Saved hexbin plots for {len(cell_indices)} cells to {output_dir}")
