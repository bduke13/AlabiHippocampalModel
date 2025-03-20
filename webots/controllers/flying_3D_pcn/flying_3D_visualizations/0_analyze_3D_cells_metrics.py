# %%
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import ttest_ind
from scipy.spatial import distance_matrix
from sklearn.neighbors import NearestNeighbors

# Import necessary functions from visualization utilities
from visualizations.vis_utils import load_hmaps, convert_xzy_hmaps


def weighted_mean_trimmed(data, weights, trim_percent=10):
    """Computes a weighted mean after trimming the lowest trim_percent of weights."""
    if np.sum(weights) == 0:
        return np.nan
    sorted_indices = np.argsort(weights)
    cutoff = int(len(weights) * (trim_percent / 100))
    trimmed_indices = sorted_indices[cutoff:]
    return np.sum(data[trimmed_indices] * weights[trimmed_indices]) / np.sum(
        weights[trimmed_indices]
    )


def compute_place_field_centers(hmap_x, hmap_y, hmap_z, hmap_pcn):
    num_cells = hmap_pcn.shape[1]
    centers = np.zeros((num_cells, 3))

    for i in range(num_cells):
        activations = hmap_pcn[:, i]
        if np.sum(activations) > 0:
            cx = weighted_mean_trimmed(hmap_x, activations)
            cy = weighted_mean_trimmed(hmap_y, activations)
            cz = weighted_mean_trimmed(hmap_z, activations)
            centers[i] = [cx, cy, cz]
        else:
            centers[i] = [np.nan, np.nan, np.nan]

    valid_cells = ~np.isnan(centers).any(axis=1)
    return centers[valid_cells]


def compute_spatial_metrics(valid_means):
    if valid_means.shape[0] < 2:
        print("Not enough place fields for analysis.")
        return None, None

    dists = distance_matrix(valid_means, valid_means)
    np.fill_diagonal(dists, np.nan)
    avg_distance = np.nanmean(dists)

    nbrs = NearestNeighbors(n_neighbors=5).fit(valid_means)
    distances, _ = nbrs.kneighbors(valid_means)
    avg_knn_distance = np.mean(distances[:, 1:].flatten())

    return avg_distance, avg_knn_distance


def compute_field_size_metrics(centers):
    stds = np.std(centers, axis=0)
    return stds


def compute_field_volumes(stds):
    return (4 / 3) * np.pi * stds[0] * stds[1] * stds[2]


def plot_3d_centers(centers, output_dir="outputs"):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c="red", marker="o")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Place Field Centers")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "place_field_centers_3d.png"))
    plt.show()


def analyze_place_cells():
    hmap_loc, hmap_pcn = load_hmaps(hmap_names=["hmap_loc", "hmap_pcn"])
    hmap_x, hmap_z, hmap_y = convert_xzy_hmaps(hmap_loc)

    valid_means = compute_place_field_centers(hmap_x, hmap_y, hmap_z, hmap_pcn)
    avg_distance, avg_knn_distance = compute_spatial_metrics(valid_means)
    field_stds = compute_field_size_metrics(valid_means)
    field_volume = compute_field_volumes(field_stds)

    print("\nAnalysis Summary:")
    print(f"Number of Valid Place Fields: {valid_means.shape[0]}")
    print(f"Average Distance Between Place Fields: {avg_distance:.4f}")
    print(f"Average k-NN Distance (k=5): {avg_knn_distance:.4f}")
    print(f"Field Standard Deviations (X, Y, Z): {field_stds}")
    print(f"Estimated Field Volume: {field_volume:.4f}")

    plot_3d_centers(valid_means)


if __name__ == "__main__":
    analyze_place_cells()
