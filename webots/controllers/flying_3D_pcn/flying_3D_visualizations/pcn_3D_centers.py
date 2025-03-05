# %%
import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional, List


def weighted_mean(data, weights):
    return np.sum(data * weights) / np.sum(weights)


def plot_place_fields_3d(
    valid_means,
    image_path="",
    output_dir="place_field_images/",
    save_plot=True,
    show_plot=False,
):
    """
    Plots the place field centers in a 3D space.

    Args:
    - valid_means: A numpy array of valid place field centers (means in 3D).
    - image_path: Path to the background image (optional, not used in 3D).
    - output_dir: Directory to save the plot (default is 'place_field_images/').
    - save_plot: Boolean flag to save the plot (default is True).
    - show_plot: Boolean flag to display the plot (default is False).
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot the place field centers
    ax.scatter(
        valid_means[:, 0],
        valid_means[:, 1],
        valid_means[:, 2],
        c="red",
        marker="x",
        s=50,
    )
    ax.set_title("3D Place Field Centers")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_zlabel("Z Coordinate")
    ax.grid(True)

    # Save the plot if save_plot is True
    if save_plot:
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, "place_field_centers_3d.jpg")
        plt.savefig(file_path)
        print(f"Saved 3D plot to {file_path}")

    # Show the plot if show_plot is True
    if show_plot:
        plt.show()

    plt.close()


def get_place_field_centers_3d(
    hmap_x,
    hmap_y,
    hmap_z,
    hmap_pcn,
    cell_indices: Optional[List[int]] = None,
    save_plot=True,
    show_plot=False,
):
    """
    Returns or plots the place field centers for selected cells in 3D.

    Args:
    - hmap_x: Array of x coordinates.
    - hmap_y: Array of y coordinates.
    - hmap_z: Array of z coordinates.
    - hmap_pcn: Activation map (z-axis data).
    - cell_indices: Optional list of cell indices to plot. If None, plots all cells.
    - save_plot: Boolean flag to save the plot (default is True).
    - show_plot: Boolean flag to display the plot (default is False).

    Returns:
    - valid_means: Numpy array of valid place field centers in 3D.
    """
    num_cells = hmap_pcn.shape[-1]

    if cell_indices is None:
        cell_indices = list(
            range(num_cells)
        )  # Use all cells if no specific indices are provided

    means = np.empty([len(cell_indices), 3])

    for i, cell_index in enumerate(cell_indices):
        try:
            x_mean = weighted_mean(hmap_x, weights=hmap_pcn[:, cell_index])
            y_mean = weighted_mean(hmap_y, weights=hmap_pcn[:, cell_index])
            z_mean = weighted_mean(hmap_z, weights=hmap_pcn[:, cell_index])
            means[i] = x_mean, y_mean, z_mean
        except:
            means[i] = np.nan, np.nan, np.nan  # Handle case with no firing

    # Filter out cells with no activation (nan values)
    valid_cells = ~np.isnan(means).any(axis=1)
    valid_means = means[valid_cells]

    # Plot the centers if needed
    plot_place_fields_3d(valid_means, save_plot=save_plot, show_plot=show_plot)

    return valid_means


if __name__ == "__main__":
    from visualizations.vis_utils import load_hmaps, convert_xzy_hmaps

    # Load hmap data
    hmap_loc, hmap_pcn = load_hmaps(hmap_names=["hmap_loc", "hmap_pcn"])
    hmap_x, hmap_z, hmap_y = convert_xzy_hmaps(hmap_loc)

    # Optional specific cell indices (e.g., [0, 5, 10]) or None to plot all
    specific_cells = range(
        0, 500, 1
    )  # Replace with a list of specific cell indices if needed

    # Get and plot 3D place field centers
    get_place_field_centers_3d(
        hmap_x,
        hmap_y,
        hmap_z,
        hmap_pcn,
        cell_indices=None,
        save_plot=False,
        show_plot=True,
    )

    print("Processed 3D place field centers.")
