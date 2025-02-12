# %%
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Optional, List


# Custom function to calculate weighted mean
def weighted_mean(data, weights):
    return np.sum(data * weights) / np.sum(weights)


# Function to plot place field centers with an optional background image
def plot_place_fields(
    valid_means,
    image_path="",
    output_dir="place_field_images/",
    save_plot=True,
    show_plot=False,
):
    """
    Plots the place field centers on a 2D plot with an optional background image.

    Args:
    - valid_means: A numpy array of valid place field centers (means).
    - image_path: Path to the background image (optional).
    - output_dir: Directory to save the plot (default is 'place_field_images/').
    - save_plot: Boolean flag to save the plot (default is True).
    - show_plot: Boolean flag to display the plot (default is False).
    """
    if not os.path.exists(image_path):
        print(f"WARNING: {image_path} does not exist. Using blank background.")
        image_path = None

    plt.figure(figsize=(8, 8))
    ax = plt.gca()

    if image_path:
        # Load and plot the background image if provided
        img = plt.imread(image_path)
        ax.imshow(
            img,
            extent=[np.min(hmap_x), np.max(hmap_x), np.min(hmap_y), np.max(hmap_y)],
            origin="upper",
        )

    # Plot the place field centers
    plt.scatter(valid_means[:, 0], valid_means[:, 1], c="red", marker="x", s=50)
    plt.title("Place Field Centers")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)
    plt.gca().set_aspect("equal", adjustable="box")

    # Save the plot if save_plot is True
    if save_plot:
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, "place_field_centers.jpg")
        plt.savefig(file_path)
        print(f"Saved plot to {file_path}")

    # Show the plot if show_plot is True
    if show_plot:
        plt.show()

    # Close the figure to save memory
    plt.close()


# Function to calculate and return or plot place field centers for selected cells
def get_place_field_centers(
    hmap_x,
    hmap_y,
    hmap_pcn,
    cell_indices: Optional[List[int]] = None,
    image_path="",
    save_plot=True,
    show_plot=False,
):
    """
    Returns or plots the place field centers for selected cells.

    Args:
    - hmap_x: Array of x coordinates.
    - hmap_y: Array of y coordinates.
    - hmap_pcn: Activation map (z-axis data).
    - cell_indices: Optional list of cell indices to plot. If None, plots all cells.
    - image_path: Path to the background image (optional).
    - save_plot: Boolean flag to save the plot (default is True).
    - show_plot: Boolean flag to display the plot (default is False).

    Returns:
    - valid_means: Numpy array of valid place field centers.
    """
    num_cells = hmap_pcn.shape[-1]

    if cell_indices is None:
        cell_indices = list(
            range(num_cells)
        )  # Use all cells if no specific indices are provided

    means = np.empty([len(cell_indices), 2])

    for i, cell_index in enumerate(cell_indices):
        try:
            x_mean = weighted_mean(hmap_x, weights=hmap_pcn[:, cell_index])
            y_mean = weighted_mean(hmap_y, weights=hmap_pcn[:, cell_index])
            means[i] = x_mean, y_mean
        except:
            means[i] = np.nan, np.nan  # Handle case with no firing

    # Filter out cells with no activation (nan values)
    valid_cells = ~np.isnan(means).any(axis=1)
    valid_means = means[valid_cells]

    # Plot the centers if needed
    plot_place_fields(valid_means, image_path, save_plot=save_plot, show_plot=show_plot)

    return valid_means


if __name__ == "__main__":
    from vis_utils import load_hmaps, convert_xzy_hmaps

    # Load hmap data
    base_path = "webots/controllers/create3_base/"
    # Load hmap data
    hmap_loc, hmap_pcn = load_hmaps(
        prefix=base_path, hmap_names=["hmap_loc", "hmap_pcn"]
    )
    hmap_x, hmap_z, hmap_y = convert_xzy_hmaps(hmap_loc)

    # Optional specific cell indices (e.g., [0, 5, 10]) or None to plot all
    specific_cells = None  # Replace with a list of specific cell indices if needed

    # Plot with the environment image as background
    image_path = "visualizations/environment_images/5x5_env_image.jpg"  # Set this to an empty string if no background is needed

    # Get and plot place field centers
    get_place_field_centers(
        hmap_x,
        hmap_y,
        hmap_pcn,
        cell_indices=specific_cells,
        image_path=image_path,
        save_plot=False,
        show_plot=True,
    )

    print("Processed place field centers.")
