# %%
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.cluster import DBSCAN
from collections import defaultdict
from numpy.linalg import norm


# -------------- Existing functions --------------
def create_hexbin(
    cell_index: int,
    hmap_x: np.ndarray,
    hmap_y: np.ndarray,
    hmap_z: np.ndarray,
    normalize: bool = False,
    filter_bottom_ratio: float = 0.0,
    analyze: bool = False,
):
    """
    Creates a hexbin plot for a given place cell index with optional normalization
    and filtering of the lowest percentage of activation values. Also allows analyzing
    x, y coordinates and their associated activation values.

    Args:
    - cell_index: The index of the place cell to plot.
    - hmap_x: The x coordinates of the grid.
    - hmap_y: The y coordinates of the grid.
    - hmap_z: The activation data for the place cells (z-axis).
    - normalize: Whether to normalize the activation values (default False).
    - filter_bottom_ratio: The fraction of lowest values to set to 0.0 (default 0.0).
    - analyze: Whether to return binned data for analysis (default False).

    Returns:
    - fig: The figure object
    - ax: The axis object
    - hb: The hexbin collection
    - binned_data (optional): A list of tuples containing (x, y, activation) for analysis
    """
    # Get activations for this cell
    activations = hmap_z[:, cell_index]

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(8, 6))

    # Create hexbin
    hb = ax.hexbin(
        hmap_x,
        hmap_y,
        C=activations,
        gridsize=50,
        reduce_C_function=np.mean,
        cmap=None,
        edgecolors="none",
    )

    # Get aggregated activations per bin
    counts = hb.get_array()

    # Get bin centers for analysis
    if analyze:
        verts = hb.get_offsets()
        x_centers = verts[:, 0]
        y_centers = verts[:, 1]

    # Filter bottom percentage of activations
    if filter_bottom_ratio > 0.0:
        threshold = np.percentile(counts, filter_bottom_ratio * 100)
        counts[counts < threshold] = 0.0

    # Normalize counts for alpha values if requested
    if normalize:
        max_count = counts.max()
        if max_count > 0:
            counts /= max_count

    # Create RGBA colors
    rgba_colors = np.zeros((len(counts), 4))  # Initialize with zeros
    rgba_colors[:, 0] = 1.0  # Red channel (example, adjust as needed)
    rgba_colors[:, 3] = counts  # Alpha channel reflects activations

    # Set the facecolors of the hexbin collection
    hb.set_facecolors(rgba_colors)

    if analyze:
        # Prepare binned data for analysis
        binned_data = [
            (x, y, activation) for x, y, activation in zip(x_centers, y_centers, counts)
        ]
        return fig, ax, hb, binned_data

    return fig, ax, hb


def plot_place_cell(
    cell_index,
    hmap_x,
    hmap_y,
    hmap_z,
    output_dir=None,
    show_plot=False,
    return_plot=False,
):
    """
    Plots a hexbin plot for a given place cell index and saves or shows it based on flags.

    Args:
    - cell_index: The index of the place cell to plot.
    - hmap_x: The x coordinates of the grid.
    - hmap_y: The y coordinates of the grid.
    - hmap_z: The activation data for the place cells (z-axis).
    - output_dir: Directory to save the plot (default is None).
    - show_plot: Boolean flag to display the plot on the screen (default is False).
    - return_plot: Boolean flag to return the figure object (default is False).

    Returns:
    - fig (optional): The figure object if return_plot is True
    """

    # Create the hexbin plot
    fig, ax, hb = create_hexbin(cell_index, hmap_x, hmap_y, hmap_z)

    # Add labels and title
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_title(f"Place Cell {cell_index} Activation")

    # Save the plot if output_dir is provided
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, f"place_cell_{cell_index}.jpg")
        plt.savefig(file_path)
        print(f"Saved plot for place cell {cell_index} to {file_path}")

    # Show the plot if show_plot flag is True
    if show_plot:
        plt.show()

    # Return the figure if return_plot flag is True
    if return_plot:
        return fig

    # Close the figure to save memory
    plt.close(fig)


# -------------- Experiment A: DBSCAN --------------
def cluster_hexbin_data(binned_data, eps=0.1, min_samples=5):
    coords = np.array([(x, y) for (x, y, act) in binned_data])
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    labels = db.labels_
    num_clusters = len(set(labels) - {-1})
    return labels, num_clusters


# -------------- Experiment B: Stacking + Cosine --------------
def stack_binned_data(list_of_binned_data):
    stacked = defaultdict(list)
    for bdata in list_of_binned_data:
        for x, y, act in bdata:
            stacked[(x, y)].append(act)
    return stacked


def compute_cosine_similarity(vecA, vecB):
    if norm(vecA) == 0.0 or norm(vecB) == 0.0:
        return 0.0
    return np.dot(vecA, vecB) / (norm(vecA) * norm(vecB))


def compute_cosine_sums(stacked_dict, distance_threshold=1.0):
    coords = list(stacked_dict.keys())
    results = {}
    for i, coord_i in enumerate(coords):
        x_i, y_i = coord_i
        vec_i = np.array(stacked_dict[coord_i])

        cos_sum = 0.0
        for j, coord_j in enumerate(coords):
            if i == j:
                continue
            x_j, y_j = coord_j
            dist = np.sqrt((x_i - x_j) ** 2 + (y_i - y_j) ** 2)
            if dist > distance_threshold:
                vec_j = np.array(stacked_dict[coord_j])
                cos_sum += compute_cosine_similarity(vec_i, vec_j)

        results[coord_i] = cos_sum

    return results


# -------------- Usage Example --------------
if __name__ == "__main__":
    # This is your data loading function
    from controllers.bek_controller.visualizations.analysis_utils import load_hmaps

    data_path = "controllers/bek_controller/IJCNN/3D_1L_1/outside/"
    hmap_x, hmap_y, hmap_z = load_hmaps(data_path)

    # Example: Single place cell for DBSCAN
    cell_index = 0
    fig, ax, hb, binned_data = create_hexbin(
        cell_index,
        hmap_x,
        hmap_y,
        hmap_z,
        normalize=True,
        filter_bottom_ratio=0.1,
        analyze=True,
    )

    # A) DBSCAN on this single binned_data
    labels, num_clusters = cluster_hexbin_data(binned_data, eps=0.2, min_samples=10)
    print("DBSCAN cluster labels:", labels)
    print("Number of clusters found:", num_clusters)

    # B) Stacking + Cosine
    # Let's assume you have multiple cells:
    all_binned_data = []
    for c_idx in [0, 1, 2]:  # just an example
        _, _, _, bd = create_hexbin(
            c_idx, hmap_x, hmap_y, hmap_z, normalize=True, analyze=True
        )
        all_binned_data.append(bd)

    # 1) Stack them
    stacked_dict = stack_binned_data(all_binned_data)

    # 2) Compute cos-sums
    distance_thresh = 1.0
    cos_sums = compute_cosine_sums(stacked_dict, distance_threshold=distance_thresh)

    # Print out the results
    for coord, sum_val in cos_sums.items():
        print(f"Coordinate {coord} -> Sum of cos sims: {sum_val:.4f}")

    # Plot your original place cell, if desired
    plot_place_cell(cell_index, hmap_x, hmap_y, hmap_z, show_plot=True)
