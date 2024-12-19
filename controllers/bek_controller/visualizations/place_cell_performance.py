# %%
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import json
from scipy.spatial.distance import cdist
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap


def separate_subplots(
    hmap_x,
    hmap_y,
    hmap_z,
    colors_rgb,
    image_path="visualizations/environment_images/5x5_env_image.jpg",
    random_index=None,
    output_dir="place_cell_images/",
):
    """
    Creates a figure with two subplots:
      - Left subplot: Place cell hexbin plot for the most active cell at a chosen reference point.
      - Right subplot: Distance hexbin plot from the chosen reference point.

    Args:
        hmap_x (np.ndarray): X coordinates of points.
        hmap_y (np.ndarray): Y coordinates of points.
        hmap_z (np.ndarray): Activation matrix of shape (N, D).
        colors_rgb (list): List of RGB color tuples for place cells.
        image_path (str): Path to the environment background image.
        random_index (int or None): Index of reference point. If None, chosen randomly.
        output_dir (str): Directory to save images if needed.
    """
    # Choose a reference index if not provided
    if random_index is None:
        random_index = np.random.randint(0, len(hmap_x))

    # Reference activations
    ref_activation = hmap_z[random_index]
    # Identify the place cell with the maximum activation at this reference point
    cell_index = np.argmax(ref_activation)

    # Get activations for this cell
    activations = hmap_z[:, cell_index]

    # Color for this cell
    color_rgb = colors_rgb[cell_index % len(colors_rgb)]

    # Create a new figure with two subplots
    fig, (ax_pc, ax_dist) = plt.subplots(ncols=2, figsize=(20, 10))

    # Plot background image on both subplots if available
    if os.path.exists(image_path):
        img = plt.imread(image_path)
        for ax in [ax_pc, ax_dist]:
            ax.imshow(
                img,
                extent=[np.min(hmap_x), np.max(hmap_x), np.min(hmap_y), np.max(hmap_y)],
                origin="upper",
            )
    else:
        print(f"WARNING: {image_path} does not exist. Using blank background.")

    # ----- Left Subplot: Place Cell Hexbin -----
    # Create a hexbin plot using activations
    hb_pc = ax_pc.hexbin(
        hmap_x,
        hmap_y,
        C=activations,
        gridsize=50,
        reduce_C_function=np.mean,
        cmap=None,  # We'll handle colors ourselves
        edgecolors="none",
    )

    # Get aggregated activations per bin
    counts = hb_pc.get_array()
    max_count = counts.max() if len(counts) > 0 else 0
    counts_normalized = counts / max_count if max_count > 0 else counts

    # Create RGBA colors for the place cell hexbin
    rgba_colors = np.zeros((len(counts), 4))
    rgba_colors[:, 0:3] = color_rgb  # Set RGB values
    rgba_colors[:, 3] = counts_normalized  # Set alpha based on activation
    hb_pc.set_facecolors(rgba_colors)

    ax_pc.set_xlabel("X Coordinate")
    ax_pc.set_ylabel("Y Coordinate")
    ax_pc.set_title(f"Place Cell {cell_index} Activation")
    ax_pc.grid(True, alpha=0.3)
    ax_pc.set_aspect("equal", adjustable="box")

    # ----- Right Subplot: Distances Scatter Plot -----
    # Compute activation distances from ref_activation to all points
    activation_distances = cdist(
        hmap_z, ref_activation[np.newaxis, :], metric="euclidean"
    ).flatten()

    # Create a custom colormap: closer = red, farther = black
    cmap = LinearSegmentedColormap.from_list("red_black", ["red", "black"])

    # Create scatter plot using the axis object
    scatter = ax_dist.scatter(hmap_x, hmap_y, c=activation_distances, cmap=cmap, s=5)
    plt.colorbar(scatter, ax=ax_dist, label="Distance")

    ax_dist.set_title(f"Distances from point {random_index}")
    ax_dist.set_xlabel("X Coordinate")
    ax_dist.set_ylabel("Y Coordinate")
    ax_dist.grid(True, alpha=0.3)
    ax_dist.set_aspect("equal", adjustable="box")

    # Highlight the reference point
    ax_dist.scatter(
        hmap_x[random_index],
        hmap_y[random_index],
        color="blue",
        s=40,
        zorder=5,
        label="Reference Point",
    )
    ax_dist.legend()

    plt.tight_layout()
    plt.show()

    print(f"Plots separated. Reference index: {random_index}, Place cell: {cell_index}")


if __name__ == "__main__":
    # Load the data
    with open("hmap_x.pkl", "rb") as f:
        hmap_x = np.array(pickle.load(f))
    with open("hmap_y.pkl", "rb") as f:
        hmap_y = np.array(pickle.load(f))
    with open("hmap_z.pkl", "rb") as f:
        hmap_z = np.array(pickle.load(f))

    # Load colors list
    with open("visualizations/colors.json", "r") as f:
        colors = json.load(f)
    colors_rgb = [mcolors.to_rgb(c) for c in colors]

    # Run the separate subplots function
    separate_subplots(hmap_x, hmap_y, hmap_z, colors_rgb, random_index=None)
