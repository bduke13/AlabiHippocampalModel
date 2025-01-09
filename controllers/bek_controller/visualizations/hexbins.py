# %%
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.sparse import data
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
    close_plot: bool = False,
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
    - close_plot: Whether to close the plot after creating it (default False).

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

    if filter_bottom_ratio > 0.0:
        # Ensure counts is a regular numpy array
        counts_array = np.asarray(counts)  # Convert to a standard ndarray
        threshold = np.percentile(counts_array, filter_bottom_ratio * 100)
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

    if close_plot:
        plt.close(fig)  # Release memory for the figure if requested

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
    coords = np.array([(x, y) for (x, y, act) in binned_data if act > 0])
    if len(coords) == 0:
        return np.array([]), 0
    if len(coords) < min_samples:
        return np.array([-1] * len(coords)), 0
    coords = coords.reshape(-1, 2)  # Ensure 2D array shape
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    labels = db.labels_
    num_clusters = len(set(labels) - {-1})
    return labels, num_clusters


def plot_hexbin_clusters(
    hmap_x,
    hmap_y,
    hmap_z,
    cell_index=0,
    eps=0.2,
    min_samples=5,
    normalize=True,
    filter_bottom_ratio=0.0,
    cmap_name="tab10",
):
    """
    1) Create a hexbin plot for the given cell_index (with optional normalization/filtering).
    2) Run DBSCAN on the bin centers (x, y).
    3) Color each bin by its cluster label.

    Args:
        hmap_x, hmap_y, hmap_z (np.ndarray): data arrays
        cell_index (int): which cell (column in hmap_z) to visualize
        eps (float): DBSCAN neighborhood distance
        min_samples (int): DBSCAN min samples
        normalize (bool): whether to normalize bin activations in create_hexbin
        filter_bottom_ratio (float): if > 0, fraction of lowest activations set to 0
        cmap_name (str): name of a Matplotlib colormap (e.g. "tab10", "Set1", etc.)

    Returns:
        fig, ax, hb, labels: The figure, axis, hexbin object, and cluster labels array
    """
    # Step 1) Create a hexbin with 'analyze=True' to get the bin centers
    fig, ax, hb, binned_data = create_hexbin(
        cell_index=cell_index,
        hmap_x=hmap_x,
        hmap_y=hmap_y,
        hmap_z=hmap_z,
        normalize=normalize,
        filter_bottom_ratio=filter_bottom_ratio,
        analyze=True,
    )

    # Extract bin centers (x, y)
    # coords = np.array([(x, y) for (x, y, _) in binned_data])
    coords = np.array([(x, y) for (x, y, act) in binned_data if act > 0])

    # Step 2) Run DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    labels = db.labels_

    # Step 3) Create a color array for each bin, based on the cluster label
    # We need to map each label to a distinct color. We'll use a colormap that
    # has enough discrete colors, like "tab10" or "tab20". You can also define
    # your own color palette if you have many clusters.

    unique_labels = np.unique(labels)

    # Prepare a color list. For simplicity, use tab10 or tab20 which has multiple discrete colors.
    # If we have more labels than the colormap can handle, we'll wrap around (mod).
    cmap = plt.get_cmap(cmap_name)
    num_colors = cmap.N  # e.g. 10 for "tab10", 20 for "tab20"

    # Initialize RGBA for all bins
    rgba_colors = np.zeros((len(labels), 4))

    for i, label in enumerate(labels):
        if label == -1:
            # Label = -1 is DBSCAN "noise". Let's color it black or gray
            rgba_colors[i] = (0.2, 0.2, 0.2, 1.0)  # dark gray
        else:
            color_idx = label % num_colors
            rgba_colors[i] = cmap(color_idx)

    # Step 4) Assign these colors to the hexbin
    hb.set_facecolors(rgba_colors)

    # OPTIONAL: If you'd like a legend for clusters, you can create patch handles:
    #   But note: DBSCAN label range might be something like [-1, 0, 1, 2, 3, ...].
    #   We'll skip the legend here for brevity, but it can be done.

    # Adjust the plot (labels, title)
    ax.set_title(f"Cell {cell_index} - DBSCAN eps={eps}, min_samples={min_samples}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # Optionally, show or return figure
    return fig, ax, hb, labels


def get_model_hexbin_metrics(hmap_x, hmap_y, hmap_z, verbose=False, show_plot=False):
    # Variables to track metrics and clusters
    total_clusters = 0  # Total number of clusters across all cells
    non_zero_cells = 0  # Number of cells with at least one cluster
    clusters_per_cell = []  # List to track the number of clusters for each cell

    # Loop through all cells and calculate clusters
    for cell_index in range(hmap_z.shape[1]):
        if verbose:
            if cell_index % 100 == 0:
                print(f"Processing cell {cell_index}")

        # Create hexbin and perform DBSCAN
        fig, ax, hb, binned_data = create_hexbin(
            cell_index,
            hmap_x,
            hmap_y,
            hmap_z,
            normalize=True,
            filter_bottom_ratio=0.2,
            analyze=True,
            close_plot=True,
        )

        # Perform DBSCAN clustering
        labels, num_clusters = cluster_hexbin_data(binned_data, eps=1, min_samples=20)

        # Update metrics
        total_clusters += num_clusters
        clusters_per_cell.append(
            num_clusters
        )  # Store clusters for histogram and bar chart
        if num_clusters > 0:
            non_zero_cells += 1

    # Calculate average clusters per non-zero cell
    if non_zero_cells > 0:
        avg_clusters_per_non_zero_cell = total_clusters / non_zero_cells
    else:
        avg_clusters_per_non_zero_cell = 0

    if verbose:
        # Print the results
        print("\nSummary:")
        print(f"Total clusters (including zero-cluster cells): {total_clusters}")
        print(f"Number of cells with non-zero clusters: {non_zero_cells}")
        print(
            f"Average clusters per non-zero cell: {avg_clusters_per_non_zero_cell:.2f}"
        )

    if show_plot:
        # Plot histogram of clusters across cells
        plt.figure(figsize=(8, 6))
        plt.hist(clusters_per_cell, bins=20, edgecolor="black", alpha=0.7)
        plt.xlabel("Number of Clusters")
        plt.ylabel("Frequency")
        plt.title("Histogram of Clusters Across Cells")
        plt.grid(True)
        plt.show()

    return {
        "total_clusters": total_clusters,
        "non_zero_cells": non_zero_cells,
        "avg_clusters_per_cell": avg_clusters_per_non_zero_cell,
    }


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


# -------------- Saving Data --------------


def save_all_hexbins(
    hmap_x, hmap_y, hmap_z, output_dir, normalize=True, filter_bottom_ratio=0.1
):
    """
    Creates and saves hexbin data for all cells in the dataset.

    Args:
    - hmap_x, hmap_y, hmap_z: The coordinate and activation data
    - output_dir: Directory to save the hexbin data
    - normalize: Whether to normalize activations
    - filter_bottom_ratio: Ratio for filtering low activations

    Returns:
    - Dictionary containing all processed hexbin data
    """
    os.makedirs(output_dir, exist_ok=True)

    # Dictionary to store all hexbin data
    all_hexbins = {}

    print(f"Processing {hmap_z.shape[1]} cells...")

    for cell_index in range(hmap_z.shape[1]):
        # Create hexbin and get data
        _, _, hb, binned_data = create_hexbin(
            cell_index,
            hmap_x,
            hmap_y,
            hmap_z,
            normalize=normalize,
            filter_bottom_ratio=filter_bottom_ratio,
            analyze=True,
        )

        # Extract and store the important data
        hexbin_data = {
            "coordinates": np.array([(x, y) for x, y, _ in binned_data]),
            "activations": np.array([act for _, _, act in binned_data]),
            "gridsize": 50,  # This matches the gridsize in create_hexbin
        }

        # Save to numpy file
        save_path = os.path.join(output_dir, f"hexbin_cell_{cell_index}.npz")
        np.savez(
            save_path,
            coordinates=hexbin_data["coordinates"],
            activations=hexbin_data["activations"],
            gridsize=np.array([hexbin_data["gridsize"]]),
        )

        all_hexbins[cell_index] = hexbin_data

        if (cell_index + 1) % 100 == 0:
            print(f"Processed {cell_index + 1} cells...")

    print("Finished processing all cells!")
    return all_hexbins


# -------------- Usage Example --------------
if __name__ == "__main__":
    # This is your data loading function
    from controllers.bek_controller.visualizations.analysis_utils import load_hmaps

    data_path = "controllers/bek_controller/IJCNN/3D_3L_250/inside_3/"
    hmap_x, hmap_y, hmap_z = load_hmaps(data_path)

    # %%
    # Example: Single place cell for DBSCAN

    # Let's pick a cell index to visualize
    cell_index = 1

    # Attempt to see if we can separate it into multiple clusters
    fig, ax, hb, labels = plot_hexbin_clusters(
        hmap_x,
        hmap_y,
        hmap_z,
        cell_index=cell_index,
        eps=1,
        min_samples=20,
        normalize=True,
        filter_bottom_ratio=0.90,
        cmap_name="tab10",
    )

    # Print out how many clusters we found
    num_clusters = len(set(labels) - {-1})
    print(f"Found {num_clusters} clusters (excluding noise).")

    print(data_path)

    plt.show()

    # %%

    # %%
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
