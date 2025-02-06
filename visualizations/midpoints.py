import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import json
import matplotlib.colors as mcolors
from typing import Optional, List


# Custom function to calculate weighted mean
def weighted_mean(data, weights):
    return np.sum(data * weights) / np.sum(weights)


# Function to plot place fields with the hexbin and the place field centers overlay
def plot_place_fields_with_centers(
    hmap_x,
    hmap_y,
    hmap_z,
    cell_indices: Optional[List[int]] = None,
    colors_rgb=None,
    image_path="",
    save_plot=True,
    show_plot=False,
    output_dir="place_cell_images/",
):
    """
    Plots place fields using hexbin with the place field centers overlayed.

    Args:
    - hmap_x: Array of x coordinates.
    - hmap_y: Array of y coordinates.
    - hmap_z: Activation map (z-axis data).
    - cell_indices: Optional list of cell indices to plot. If None, plots all cells.
    - colors_rgb: List of RGB colors for the place cells.
    - image_path: Path to the background image (optional).
    - save_plot: Boolean flag to save the plot (default is True).
    - show_plot: Boolean flag to display the plot (default is False).
    - output_dir: Directory to save the plot (default is 'place_cell_images/').
    """
    num_cells = hmap_z.shape[-1]

    if cell_indices is None:
        cell_indices = list(
            range(num_cells)
        )  # Use all cells if no specific indices are provided

    means = np.empty([len(cell_indices), 2])

    # Calculate the place field centers (weighted mean)
    for i, cell_index in enumerate(cell_indices):
        try:
            x_mean = weighted_mean(hmap_x, weights=hmap_z[:, cell_index])
            y_mean = weighted_mean(hmap_y, weights=hmap_z[:, cell_index])
            means[i] = x_mean, y_mean
        except:
            means[i] = np.nan, np.nan  # Handle case with no firing

    # Filter out cells with no activation (nan values)
    valid_cells = ~np.isnan(means).any(axis=1)
    valid_means = means[valid_cells]

    plt.figure(figsize=(8, 8))
    ax = plt.gca()

    if image_path and os.path.exists(image_path):
        # Load and plot the background image if provided
        img = plt.imread(image_path)
        ax.imshow(
            img,
            extent=[np.min(hmap_x), np.max(hmap_x), np.min(hmap_y), np.max(hmap_y)],
            origin="upper",
        )

    # Overlay the hexbin plot for each place cell
    for i, cell_index in enumerate(cell_indices):
        if not valid_cells[i]:
            continue

        activations = hmap_z[:, cell_index]

        # Color for this cell
        color_rgb = (
            colors_rgb[cell_index % len(colors_rgb)]
            if colors_rgb is not None
            else "blue"
        )

        # Create the hexbin plot
        hb = plt.hexbin(
            hmap_x,
            hmap_y,
            C=activations,
            gridsize=50,
            reduce_C_function=np.mean,
            cmap=None,
            edgecolors="none",
        )

        # Overlay the place field center as a red "X"
        plt.scatter(valid_means[i, 0], valid_means[i, 1], c="red", marker="x", s=100)

    plt.title(f"Place Field Centers and Activation Hexbin")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)
    plt.gca().set_aspect("equal", adjustable="box")

    # Save the plot if save_plot is True
    if save_plot:
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, "place_field_centers_hexbin.jpg")
        plt.savefig(file_path)
        print(f"Saved plot to {file_path}")

    # Show the plot if show_plot is True
    if show_plot:
        plt.show()

    # Close the figure to save memory
    plt.close()


if __name__ == "__main__":
    # Load hmap data
    with open("../hmap_x.pkl", "rb") as f:
        hmap_x = np.array(pickle.load(f))[10:]
    with open("../hmap_y.pkl", "rb") as f:
        hmap_y = np.array(pickle.load(f))[10:]
    with open("../hmap_z.pkl", "rb") as f:
        hmap_z = np.asarray(pickle.load(f))[10:]

    # Load the colors list
    with open("colors.json", "r") as f:
        colors = json.load(f)

    # Convert hex colors to RGB format
    colors_rgb = [mcolors.to_rgb(c) for c in colors]

    # Optional specific cell indices (e.g., [0, 5, 10]) or None to plot all
    specific_cells = [5]  # Replace with a list of specific cell indices if needed

    # Plot with the environment image as background
    image_path = "environment_images/5x5_env_image.jpg"  # Set this to an empty string if no background is needed

    # Plot hexbin and place field centers
    plot_place_fields_with_centers(
        hmap_x,
        hmap_y,
        hmap_z,
        cell_indices=specific_cells,
        colors_rgb=colors_rgb,
        image_path=image_path,
        save_plot=False,
        show_plot=True,
    )

    print("Processed place field centers with hexbin.")
