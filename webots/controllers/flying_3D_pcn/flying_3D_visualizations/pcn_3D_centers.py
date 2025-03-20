# %%
import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import distance_matrix
from sklearn.neighbors import NearestNeighbors
from typing import Optional, List


def weighted_mean_trimmed(data, weights, trim_percent=10):
    if np.sum(weights) == 0:
        return np.nan  # Avoid division by zero

    # Sort data by weights and trim out bottom `trim_percent` percent
    sorted_indices = np.argsort(weights)
    cutoff = int(len(weights) * (trim_percent / 100))
    trimmed_indices = sorted_indices[cutoff:]

    return np.sum(data[trimmed_indices] * weights[trimmed_indices]) / np.sum(
        weights[trimmed_indices]
    )


def check_distribution(valid_means):
    if valid_means.shape[0] < 2:
        print("Not enough valid place fields for analysis.")
        return

    dists = distance_matrix(valid_means, valid_means)
    np.fill_diagonal(dists, np.nan)  # Ignore self-distances
    avg_dist = np.nanmean(dists)
    print(f"Average nearest neighbor distance: {avg_dist:.3f}")


def compute_knn_distribution(valid_means, k=5):
    if valid_means.shape[0] < k:
        print("Not enough valid place fields to compute k-NN distribution.")
        return np.nan, None

    nbrs = NearestNeighbors(n_neighbors=k).fit(valid_means)
    distances, _ = nbrs.kneighbors(valid_means)
    knn_distances = distances[:, 1:].flatten()  # Exclude self-distance (0)
    avg_spread = np.mean(knn_distances)

    print(f"Spread Metric (Avg k-NN Distance for k={k}): {avg_spread:.3f}")
    return avg_spread, knn_distances


def count_cells_in_subsections(
    valid_means, bin_size=1.0, output_dir="place_field_images/"
):
    """Counts place cell centers in discrete spatial subsections (3D grid) and plots the density."""
    x_bins = np.arange(
        np.min(valid_means[:, 0]), np.max(valid_means[:, 0]) + bin_size, bin_size
    )
    y_bins = np.arange(
        np.min(valid_means[:, 1]), np.max(valid_means[:, 1]) + bin_size, bin_size
    )
    z_bins = np.arange(
        np.min(valid_means[:, 2]), np.max(valid_means[:, 2]) + bin_size, bin_size
    )

    hist, edges = np.histogramdd(valid_means, bins=[x_bins, y_bins, z_bins])

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    x_centers = (edges[0][:-1] + edges[0][1:]) / 2
    y_centers = (edges[1][:-1] + edges[1][1:]) / 2
    z_centers = (edges[2][:-1] + edges[2][1:]) / 2

    x, y, z = np.meshgrid(x_centers, y_centers, z_centers, indexing="ij")
    x, y, z, hist_values = x.flatten(), y.flatten(), z.flatten(), hist.flatten()

    nonzero_mask = hist_values > 0
    x, y, z, hist_values = (
        x[nonzero_mask],
        y[nonzero_mask],
        z[nonzero_mask],
        hist_values[nonzero_mask],
    )

    ax.scatter(x, y, z, c=hist_values, cmap="viridis", s=hist_values * 50, alpha=0.7)
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_zlabel("Z Coordinate")
    ax.set_title("Place Cell Density in 3D Grid")

    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, "place_cell_density_3d.jpg")
    plt.savefig(file_path)
    print(f"Saved 3D place cell density plot to {file_path}")

    plt.show()


def plot_place_fields_3d(
    valid_means,
    output_dir="place_field_images/",
    save_plot=True,
    show_plot=False,
):
    """
    Plots the place field centers in a 3D space.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot the place field centers with transparency
    ax.scatter(
        valid_means[:, 0],
        valid_means[:, 1],
        valid_means[:, 2],
        c="red",
        marker="x",
        s=50,
        alpha=0.7,
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
    """
    num_cells = hmap_pcn.shape[-1]
    if cell_indices is None:
        cell_indices = list(
            range(num_cells)
        )  # Use all cells if no specific indices are provided

    means = np.empty([len(cell_indices), 3])

    for i, cell_index in enumerate(cell_indices):
        try:
            x_mean = weighted_mean_trimmed(hmap_x, weights=hmap_pcn[:, cell_index])
            y_mean = weighted_mean_trimmed(hmap_y, weights=hmap_pcn[:, cell_index])
            z_mean = weighted_mean_trimmed(hmap_z, weights=hmap_pcn[:, cell_index])
            means[i] = x_mean, y_mean, z_mean
        except:
            means[i] = np.nan, np.nan, np.nan  # Handle case with no firing

    # Filter out cells with no activation (nan values)
    valid_cells = ~np.isnan(means).any(axis=1)
    valid_means = means[valid_cells]

    # Print the total number of valid place field centers
    print(f"Total number of place field centers plotted: {valid_means.shape[0]}")

    # Check distribution of place field centers
    check_distribution(valid_means)

    # Compute k-NN distribution
    spread_metric, knn_distances = compute_knn_distribution(valid_means)

    # Plot the centers if needed
    plot_place_fields_3d(valid_means, save_plot=save_plot, show_plot=show_plot)

    # Count cells in environment subsections (3D grid) and visualize density
    count_cells_in_subsections(valid_means)

    return valid_means, spread_metric


if __name__ == "__main__":
    from visualizations.vis_utils import load_hmaps, convert_xzy_hmaps

    # Load hmap data
    hmap_loc, hmap_pcn = load_hmaps(hmap_names=["hmap_loc", "hmap_pcn"])
    hmap_x, hmap_z, hmap_y = convert_xzy_hmaps(hmap_loc)

    # Get and plot 3D place field centers
    valid_means, spread_metric = get_place_field_centers_3d(
        hmap_x,
        hmap_y,
        hmap_z,
        hmap_pcn,
        cell_indices=None,
        save_plot=False,
        show_plot=True,
    )

    print(f"Processed 3D place field centers. Spread Metric: {spread_metric:.3f}")
