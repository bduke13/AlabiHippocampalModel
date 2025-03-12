# %%
import os
import numpy as np
import matplotlib.pyplot as plt

# Import utility functions from your visualization module.
from visualizations.vis_utils import load_hmaps, convert_xzy_hmaps


def compute_weighted_center_std(
    hmap_x: np.ndarray, hmap_y: np.ndarray, hmap_z: np.ndarray, hmap_pcn: np.ndarray
):
    """
    Compute the weighted center (mean) and weighted standard deviation for each place cell.

    Args:
        hmap_x: 1D array of x coordinates along the trajectory.
        hmap_y: 1D array of y coordinates along the trajectory.
        hmap_z: 1D array of z coordinates along the trajectory.
        hmap_pcn: 2D array of place cell activations (shape: [N, num_cells]).

    Returns:
        centers: (num_cells x 3) array with weighted [x, y, z] centers for each cell.
        stds: (num_cells x 3) array with weighted standard deviations along [x, y, z] for each cell.
    """
    num_cells = hmap_pcn.shape[1]
    centers = np.zeros((num_cells, 3))
    stds = np.zeros((num_cells, 3))

    for cell in range(num_cells):
        activations = hmap_pcn[:, cell]
        total_activation = np.sum(activations)
        if total_activation > 0:
            # Compute weighted mean for x, y, z
            cx = np.average(hmap_x, weights=activations)
            cy = np.average(hmap_y, weights=activations)
            cz = np.average(hmap_z, weights=activations)
            centers[cell] = [cx, cy, cz]

            # Compute weighted standard deviations
            std_x = np.sqrt(np.average((hmap_x - cx) ** 2, weights=activations))
            std_y = np.sqrt(np.average((hmap_y - cy) ** 2, weights=activations))
            std_z = np.sqrt(np.average((hmap_z - cz) ** 2, weights=activations))
            stds[cell] = [std_x, std_y, std_z]
        else:
            centers[cell] = [np.nan, np.nan, np.nan]
            stds[cell] = [np.nan, np.nan, np.nan]

    return centers, stds


def plot_center_histograms(centers: np.ndarray):
    """
    Plot histograms for the weighted centers (x, y, z) of place cells.

    Args:
        centers: (num_cells x 3) array with the weighted centers.
    """
    valid_x = centers[:, 0][~np.isnan(centers[:, 0])]
    valid_y = centers[:, 1][~np.isnan(centers[:, 1])]
    valid_z = centers[:, 2][~np.isnan(centers[:, 2])]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].hist(valid_x, bins=30, edgecolor="black")
    axes[0].set_title("Histogram of X Centers")
    axes[0].set_xlabel("X Coordinate")
    axes[0].set_ylabel("Frequency")

    axes[1].hist(valid_y, bins=30, edgecolor="black")
    axes[1].set_title("Histogram of Y Centers")
    axes[1].set_xlabel("Y Coordinate")
    axes[1].set_ylabel("Frequency")

    axes[2].hist(valid_z, bins=30, edgecolor="black")
    axes[2].set_title("Histogram of Z Centers")
    axes[2].set_xlabel("Z Coordinate")
    axes[2].set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()


def plot_std_histograms(stds: np.ndarray):
    """
    Plot histograms for the weighted standard deviations (x, y, z) of place cells.

    Args:
        stds: (num_cells x 3) array with the weighted standard deviations.
    """
    valid_std_x = stds[:, 0][~np.isnan(stds[:, 0])]
    valid_std_y = stds[:, 1][~np.isnan(stds[:, 1])]
    valid_std_z = stds[:, 2][~np.isnan(stds[:, 2])]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].hist(valid_std_x, bins=30, edgecolor="black")
    axes[0].set_title("Histogram of Std X")
    axes[0].set_xlabel("Std X")
    axes[0].set_ylabel("Frequency")

    axes[1].hist(valid_std_y, bins=30, edgecolor="black")
    axes[1].set_title("Histogram of Std Y")
    axes[1].set_xlabel("Std Y")
    axes[1].set_ylabel("Frequency")

    axes[2].hist(valid_std_z, bins=30, edgecolor="black")
    axes[2].set_title("Histogram of Std Z")
    axes[2].set_xlabel("Std Z")
    axes[2].set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()


def main():

    if False:
        worlds = os.listdir("webots/controllers/flying_3D_pcn_looping/pkl/")

        for world in worlds:
            CONTROLLER_NAME = "flying_3D_pcn_looping"
            WORLD_NAME = world
            print(f"WORLD: {world}")

            # Load hmap data from hardcoded world name
            hmap_loc, hmap_pcn = load_hmaps(hmap_names=["hmap_loc", "hmap_pcn"])
            hmap_x, hmap_z, hmap_y = convert_xzy_hmaps(hmap_loc)
            centers, stds = compute_weighted_center_std(
                hmap_x, hmap_y, hmap_z, hmap_pcn
            )
            plot_center_histograms(centers)

            # Plot histograms for standard deviations to visualize the spread/skew.
            plot_std_histograms(stds)

    # Load the hmap data using your provided utility functions.
    # hmap_loc: positional data; hmap_pcn: place cell activations.
    hmap_loc, hmap_pcn = load_hmaps(hmap_names=["hmap_loc", "hmap_pcn"])
    hmap_x, hmap_z, hmap_y = convert_xzy_hmaps(hmap_loc)

    # Compute weighted centers and standard deviations.
    centers, stds = compute_weighted_center_std(hmap_x, hmap_y, hmap_z, hmap_pcn)

    # Optionally, print computed values for a few cells.
    num_print = min(10, centers.shape[0])
    print("Weighted Centers and Standard Deviations for the first few place cells:")
    for cell in range(num_print):
        print(f"Cell {cell}: Center = {centers[cell]}, Std = {stds[cell]}")

    # Plot histograms for weighted centers.
    plot_center_histograms(centers)

    # Plot histograms for standard deviations to visualize the spread/skew.
    plot_std_histograms(stds)


if __name__ == "__main__":
    main()
