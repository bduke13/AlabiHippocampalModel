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
    hmap_pcn: np.ndarray,
    normalize: bool = False,
    filter_bottom_ratio: float = 0.0,
    analyze: bool = False,
    get_counts: bool = False,
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
    - hmap_pcn: The activation data for the place cells (z-axis).
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
    activations = hmap_pcn[:, cell_index]

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
        counts_array = np.asarray(counts)  # Convert to a standard ndarray
        threshold = np.percentile(counts_array, filter_bottom_ratio * 100)
        counts[counts < threshold] = 0.0

    # Normalize counts for alpha values if requested
    if normalize:
        max_count = counts.max()
        if max_count > 0:
            counts /= max_count

    # Create RGBA colors
    rgba_colors = np.zeros((len(counts), 4))
    rgba_colors[:, 0] = 1.0  # Red channel (example, adjust as needed)
    rgba_colors[:, 3] = counts  # Alpha channel reflects activations

    # Set the facecolors of the hexbin collection
    hb.set_facecolors(rgba_colors)
    counts = hb.get_array()  # Number of points per bin

    if analyze:
        verts = hb.get_offsets()
        x_centers = verts[:, 0]
        y_centers = verts[:, 1]

    if close_plot:
        plt.close(fig)

    if analyze:
        binned_data = [
            (x, y, activation) for x, y, activation in zip(x_centers, y_centers, counts)
        ]
        if get_counts:
            return fig, ax, hb, binned_data, counts
        else:
            return fig, ax, hb, binned_data

    return fig, ax, hb, counts


def plot_place_cell(
    cell_index,
    hmap_x,
    hmap_y,
    hmap_pcn,
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
    - hmap_pcn: The activation data for the place cells (z-axis).
    - output_dir: Directory to save the plot (default is None).
    - show_plot: Boolean flag to display the plot on the screen (default is False).
    - return_plot: Boolean flag to return the figure object (default is False).

    Returns:
    - fig (optional): The figure object if return_plot is True
    """

    # Create the hexbin plot
    fig, ax, hb = create_hexbin(cell_index, hmap_x, hmap_y, hmap_pcn)

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
    hmap_pcn,
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
        hmap_x, hmap_y, hmap_pcn (np.ndarray): data arrays
        cell_index (int): which cell (column in hmap_pcn) to visualize
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
        hmap_pcn=hmap_pcn,
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


def get_model_hexbin_metrics(hmap_x, hmap_y, hmap_pcn, verbose=False, show_plot=False):
    # Variables to track metrics and clusters
    total_clusters = 0
    cells_with_multiple_clusters = 0
    cells_with_non_zero_clusters = 0
    clusters_per_cell = []

    # Loop through all cells and calculate clusters
    for cell_index in range(hmap_pcn.shape[1]):
        if verbose:
            if cell_index % 100 == 0:
                print(f"Processing cell {cell_index}")

        # Create hexbin and perform DBSCAN
        fig, ax, hb, binned_data = create_hexbin(
            cell_index,
            hmap_x,
            hmap_y,
            hmap_pcn,
            normalize=True,
            filter_bottom_ratio=0.2,
            analyze=True,
            close_plot=True,
        )

        # Perform DBSCAN clustering
        labels, num_clusters = cluster_hexbin_data(binned_data, eps=1, min_samples=20)

        # Update metrics
        total_clusters += num_clusters
        clusters_per_cell.append(num_clusters)
        if num_clusters > 0:
            cells_with_non_zero_clusters += 1
        if num_clusters > 1:
            cells_with_multiple_clusters += 1

    # Calculate averages
    avg_clusters_per_non_zero_cell = (
        total_clusters / cells_with_non_zero_clusters
        if cells_with_non_zero_clusters > 0
        else 0
    )
    avg_clusters_per_multi_cluster_cell = (
        total_clusters / cells_with_multiple_clusters
        if cells_with_multiple_clusters > 0
        else 0
    )

    if verbose:
        # Print the results
        print("\nSummary:")
        print(f"Total clusters (including zero-cluster cells): {total_clusters}")
        print(f"Number of cells with non-zero clusters: {cells_with_non_zero_clusters}")
        print(
            f"Number of cells with more than one cluster: {cells_with_multiple_clusters}"
        )
        print(
            f"Average clusters per non-zero cell: {avg_clusters_per_non_zero_cell:.2f}"
        )
        print(
            f"Average clusters per cell with more than one cluster: {avg_clusters_per_multi_cluster_cell:.2f}"
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
        "non_zero_cells": cells_with_non_zero_clusters,
        "cells_with_multiple_clusters": cells_with_multiple_clusters,
        "avg_clusters_per_non_zero_cell": avg_clusters_per_non_zero_cell,
        "avg_clusters_per_multi_cluster_cell": avg_clusters_per_multi_cluster_cell,
    }


# -------------- Experiment B: Stacking + Cosine --------------


def stack_binned_data_by_location(list_of_binned_data):
    """
    Stack binned data from all place cells into a dictionary where bins
    are identified by their center coordinates.

    Args:
    - list_of_binned_data: List of binned data from multiple place cells.

    Returns:
    - stacked_dict: A dictionary with keys as (x, y) coordinates and values as lists of activations.
    """
    stacked_dict = defaultdict(list)
    for bdata in list_of_binned_data:
        for x, y, act in bdata:
            stacked_dict[(x, y)].append(act)
    return stacked_dict


def compute_cosine_similarity_sums(stacked_dict, distance_threshold=2.0):
    """
    Compute the sum of cosine similarities between bins in the environment, considering
    only bins that are more than a specified distance apart.

    Args:
    - stacked_dict: Dictionary of binned data with keys as (x, y) coordinates.
    - distance_threshold: Minimum distance (in meters) to consider for similarity calculation.

    Returns:
    - similarity_sums: A dictionary with keys as (x, y) and values as the sum of cosine similarities.
    """
    coords = np.array(list(stacked_dict.keys()))  # Extract all bin coordinates
    similarity_sums = {}

    # Iterate over each bin
    for i, (x_i, y_i) in enumerate(coords):
        vec_i = np.array(stacked_dict[(x_i, y_i)])
        sum_similarity = 0.0

        # Compare to all other bins
        for j, (x_j, y_j) in enumerate(coords):
            if i == j:
                continue

            # Calculate Euclidean distance
            distance = np.sqrt((x_i - x_j) ** 2 + (y_i - y_j) ** 2)
            if distance > distance_threshold:
                vec_j = np.array(stacked_dict[(x_j, y_j)])

                # Compute cosine similarity
                if norm(vec_i) > 0 and norm(vec_j) > 0:
                    cos_sim = np.dot(vec_i, vec_j) / (norm(vec_i) * norm(vec_j))
                    sum_similarity += cos_sim

        # Store the sum of similarities for this bin
        similarity_sums[(x_i, y_i)] = sum_similarity

    return similarity_sums


def analyze_cosine_similarity(
    hmap_x,
    hmap_y,
    hmap_pcn,
    gridsize=50,
    filter_bottom_ratio=0.1,
    distance_threshold=2.0,
):
    """
    Full analysis pipeline for cosine similarity in discretized activation space.

    Args:
    - hmap_x, hmap_y, hmap_pcn: The coordinate and activation data.
    - gridsize: The size of the hexbin grid.
    - filter_bottom_ratio: Ratio for filtering low activations.
    - distance_threshold: Minimum distance (in meters) to consider for similarity calculation.

    Returns:
    - similarity_sums: Dictionary of cosine similarity sums for each bin.
    """
    all_binned_data = []

    # Create hexbins for all cells
    for cell_index in range(hmap_pcn.shape[1]):
        _, _, _, binned_data = create_hexbin(
            cell_index=cell_index,
            hmap_x=hmap_x,
            hmap_y=hmap_y,
            hmap_pcn=hmap_pcn,
            normalize=True,
            filter_bottom_ratio=filter_bottom_ratio,
            analyze=True,
            close_plot=True,
        )
        all_binned_data.append(binned_data)

    # Stack binned data by location
    stacked_dict = stack_binned_data_by_location(all_binned_data)

    # Compute cosine similarity sums
    similarity_sums = compute_cosine_similarity_sums(stacked_dict, distance_threshold)

    return similarity_sums


def plot_cosine_similarity_heatmap(
    hmap_x,
    hmap_y,
    hmap_pcn,
    stacked_dict,
    similarity_sums,
    cell_index=0,
    gridsize=50,
    cmap="viridis",
):
    """
    Plot a heatmap of cosine similarities for a specific place cell overlaid on the hexbin grid.

    Args:
    - hmap_x, hmap_y, hmap_pcn: The coordinate and activation data.
    - stacked_dict: Dictionary of stacked binned data with keys as (x, y).
    - similarity_sums: Dictionary of cosine similarity sums for each bin.
    - cell_index: The index of the place cell to visualize.
    - gridsize: The hexbin grid size.
    - cmap: The colormap for the heatmap.

    Returns:
    - fig, ax: The figure and axis of the plot.
    """
    # Generate hexbin plot for the place cell
    fig, ax, hb, binned_data = create_hexbin(
        cell_index=cell_index,
        hmap_x=hmap_x,
        hmap_y=hmap_y,
        hmap_pcn=hmap_pcn,
        normalize=True,
        analyze=True,
        close_plot=False,
    )

    # Extract bin centers from hexbin plot
    bin_centers = hb.get_offsets()

    # Map similarity sums to bin centers
    bin_colors = []
    for x, y in bin_centers:
        # Find the similarity sum for this bin, default to 0 if not found
        bin_colors.append(similarity_sums.get((x, y), 0.0))

    # Normalize bin colors for visualization
    bin_colors = np.array(bin_colors)
    bin_colors_normalized = (bin_colors - bin_colors.min()) / (
        bin_colors.max() - bin_colors.min()
    )

    # Apply the normalized colors to the hexbin facecolors
    rgba_colors = plt.cm.get_cmap(cmap)(bin_colors_normalized)
    hb.set_facecolors(rgba_colors)

    # Add colorbar
    cbar = plt.colorbar(hb, ax=ax, orientation="vertical")
    cbar.set_label("Cosine Similarity Sum")

    # Adjust plot title and labels
    ax.set_title(f"Cosine Similarity Heatmap - Cell {cell_index}")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")

    return fig, ax


def analyze_bin_similarities(
    stacked_dict,
    bin_index=None,
    distance_threshold=2.0,
    show_plots=True,
    save_path=None,
):
    """
    Analyzes and visualizes cosine similarities between a selected bin and all other bins.

    Args:
        stacked_dict (dict): Dictionary of binned data with keys as (x, y) coordinates
        bin_index (int, optional): Index of bin to analyze. If None, picks random bin
        distance_threshold (float): Minimum distance to consider for similarity calculation
        show_plots (bool): Whether to display the plots
        save_path (str, optional): Path to save the plots

    Returns:
        tuple: (bin_coords, similarities, fig_spatial, fig_dist)
            - bin_coords: (x,y) coordinates of selected bin
            - similarities: Dictionary mapping (x,y) coords to similarity values
            - fig_spatial: Figure object for spatial similarity plot
            - fig_dist: Figure object for similarity distribution plot
    """
    coords = np.array(list(stacked_dict.keys()))

    # Select bin
    if bin_index is None:
        bin_index = np.random.randint(0, len(coords))
    bin_coords = tuple(coords[bin_index])

    # Get vector for selected bin
    vec_ref = np.array(stacked_dict[bin_coords])

    # Calculate similarities to all other bins
    similarities = {}
    for x, y in coords:
        coord = (x, y)
        if coord == bin_coords:
            similarities[coord] = 1.0  # Self-similarity
            continue

        # Calculate distance
        distance = np.sqrt((bin_coords[0] - x) ** 2 + (bin_coords[1] - y) ** 2)
        if distance <= distance_threshold:
            similarities[coord] = 0.0
            continue

        # Calculate cosine similarity
        vec = np.array(stacked_dict[coord])
        if norm(vec_ref) > 0 and norm(vec) > 0:
            cos_sim = np.dot(vec_ref, vec) / (norm(vec_ref) * norm(vec))
            similarities[coord] = cos_sim
        else:
            similarities[coord] = 0.0

    if show_plots:
        # Create spatial similarity plot
        fig_spatial, ax1 = plt.subplots(figsize=(10, 8))
        scatter_coords = np.array(list(similarities.keys()))
        scatter_values = np.array(list(similarities.values()))

        sc = ax1.scatter(
            scatter_coords[:, 0],
            scatter_coords[:, 1],
            c=scatter_values,
            cmap="viridis",
            vmin=-1,
            vmax=1,
        )
        ax1.scatter(
            bin_coords[0],
            bin_coords[1],
            color="red",
            marker="*",
            s=200,
            label="Selected Bin",
        )
        plt.colorbar(sc, label="Cosine Similarity")
        ax1.set_xlabel("X Coordinate")
        ax1.set_ylabel("Y Coordinate")
        ax1.set_title(
            f"Spatial Distribution of Cosine Similarities\nReference Bin: {bin_coords}"
        )
        ax1.legend()

        # Create similarity distribution plot
        fig_dist, ax2 = plt.subplots(figsize=(8, 6))
        similarity_values = [
            v for v in similarities.values() if v != 1.0
        ]  # Exclude self-similarity
        ax2.hist(similarity_values, bins=50, density=True, alpha=0.7)
        ax2.axvline(
            np.mean(similarity_values),
            color="r",
            linestyle="--",
            label=f"Mean: {np.mean(similarity_values):.3f}",
        )
        ax2.set_xlabel("Cosine Similarity")
        ax2.set_ylabel("Density")
        ax2.set_title("Distribution of Cosine Similarities")
        ax2.legend()

        if save_path:
            fig_spatial.savefig(f"{save_path}_spatial.png")
            fig_dist.savefig(f"{save_path}_distribution.png")

        plt.show()

    return bin_coords, similarities, fig_spatial, fig_dist


def analyze_active_cells_per_bin(
    stacked_dict, activation_threshold=0.0, show_plot=True, save_path=None
):
    """
    Analyzes the number of active cells in each bin and creates a histogram.

    Args:
        stacked_dict (dict): Dictionary mapping (x,y) coordinates to lists of cell activations
        activation_threshold (float): Minimum activation value to consider a cell active
        show_plot (bool): Whether to display the histogram
        save_path (str, optional): Path to save the plot

    Returns:
        tuple: (active_cells_dict, mean_active_cells, std_active_cells, fig)
            - active_cells_dict: Dictionary mapping (x,y) coords to number of active cells
            - mean_active_cells: Mean number of active cells across all bins
            - std_active_cells: Standard deviation of active cells across bins
            - fig: Figure object if show_plot is True, else None
    """
    active_cells_dict = {}
    active_cells_counts = []

    # Count active cells for each bin
    for coord, activations in stacked_dict.items():
        active_count = sum(1 for act in activations if act > activation_threshold)
        active_cells_dict[coord] = active_count
        active_cells_counts.append(active_count)

    # Calculate statistics
    active_cells_counts = np.array(active_cells_counts)
    mean_active_cells = np.mean(active_cells_counts)
    std_active_cells = np.std(active_cells_counts)

    fig = None
    if show_plot:
        fig, ax = plt.subplots(figsize=(10, 6))

        # Create histogram
        counts, bins, _ = ax.hist(
            active_cells_counts,
            bins=50,
            density=False,
            alpha=0.7,
            color="blue",
            edgecolor="black",
        )

        # Add vertical line for mean
        ax.axvline(
            mean_active_cells,
            color="red",
            linestyle="--",
            label=f"Mean: {mean_active_cells:.1f}",
        )

        # Add text with statistics
        stats_text = f"Mean: {mean_active_cells:.1f}\nStd: {std_active_cells:.1f}"
        ax.text(
            0.95,
            0.95,
            stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        ax.set_xlabel("Number of Active Cells")
        ax.set_ylabel("Number of Bins")
        ax.set_title("Distribution of Active Cells per Bin")
        ax.grid(True, alpha=0.3)
        ax.legend()

        if save_path:
            plt.savefig(save_path)

        plt.tight_layout()

    return active_cells_dict, mean_active_cells, std_active_cells, fig


def plot_similarity_sums(
    similarity_sums,
    title="Far-Cosine Similarity Sums",
    output_path=None,
    close_plot=False,
):
    """
    Plots the similarity sums as a scatter plot, where each point (x,y)
    is colored by its cosine similarity sum. Optionally saves the figure
    to disk and can close the figure to free memory.

    Args:
        similarity_sums (dict):
            Keys are (x, y) tuples, values are float similarity sums.
        title (str):
            Title for the plot.
        output_path (str or None):
            File path to save the plot image (e.g., '/path/to/plot.png').
            If None, no image is saved. Default is None.
        close_plot (bool):
            If True, close the figure after saving/showing.
            Default is False.

    Returns:
        fig (matplotlib Figure):
            The figure object.
        ax (matplotlib Axes):
            The axis object.
        total_similarity_sum (float):
            The sum of all the similarity_sums values (sum of all bins).
    """
    # Extract coordinates (x, y) and their sums into arrays
    coords = np.array(list(similarity_sums.keys()))
    sums = np.array(list(similarity_sums.values()))

    # Calculate the total sum of all bins
    total_similarity_sum = float(np.sum(sums))

    fig, ax = plt.subplots(figsize=(8, 6))

    # Scatter the points with sums as colors
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=sums, cmap="viridis")
    cbar = plt.colorbar(sc, ax=ax, label="Cosine Similarity Sum")

    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_title(title)

    # If an output path is provided, save the figure
    if output_path is not None:
        # Make sure the directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        print(f"Plot saved to: {output_path}")

    # Optionally close the figure
    if close_plot:
        plt.close(fig)

    return fig, ax, total_similarity_sum


# -------------- Usage Example --------------
# %%
if __name__ == "__main__":
    from controllers.bek_controller.visualizations.analysis_utils import *

    from vis_utils import (
        load_hmaps,
        convert_xzy_hmaps,
    )

    # Load hmap data from hardcoded world name
    hmap_loc, hmap_pcn = load_hmaps(hmap_names=["hmap_loc", "hmap_pcn"])
    hmap_x, hmap_z, hmap_y = convert_xzy_hmaps(hmap_loc)

    # %%
    metrics = get_model_hexbin_metrics(hmap_x, hmap_y, hmap_pcn, verbose=False)

    # %%
    fig, ax, hb, binned_data = create_hexbin(
        cell_index=0,
        hmap_x=hmap_x,
        hmap_y=hmap_y,
        hmap_pcn=hmap_pcn,
        normalize=True,
        filter_bottom_ratio=0.2,
        analyze=True,
        close_plot=True,
    )
    # %%

    # Perform cosine similarity analysis
    similarity_sums = analyze_cosine_similarity(
        hmap_x,
        hmap_y,
        hmap_pcn,
        gridsize=50,
        filter_bottom_ratio=0.1,
        distance_threshold=2.0,
    )

    # %%
    # Plot cosine similarity heatmap for a specific cell
    fig, ax, total_sum = plot_similarity_sums(
        similarity_sums,
        title="My Cosine Similarity Plot",
        output_path=data_path,
        close_plot=True,
    )
    fig.show()

    # %%

    # Get stacked dictionary first
    all_binned_data = []
    for cell_index in range(hmap_pcn.shape[1]):
        _, _, _, binned_data = create_hexbin(
            cell_index=cell_index,
            hmap_x=hmap_x,
            hmap_y=hmap_y,
            hmap_pcn=hmap_pcn,
            normalize=True,
            filter_bottom_ratio=0.1,
            analyze=True,
            close_plot=True,
        )
        all_binned_data.append(binned_data)

    stacked_dict = stack_binned_data_by_location(all_binned_data)

    # %%
    # Analyze a random bin
    bin_coords, similarities, fig_spatial, fig_dist = analyze_bin_similarities(
        stacked_dict,
        bin_index=None,  # Random bin
        distance_threshold=2.0,
        show_plots=True,
        save_path="",
    )

    # %%
    active_cells_dict, mean_active, std_active, fig = analyze_active_cells_per_bin(
        stacked_dict,
        activation_threshold=0.2,  # Adjust this threshold as needed
        show_plot=True,
        save_path="",  # Optional
    )
    fig.show()

    print(
        f"Average number of active cells per bin: {mean_active:.2f} ± {std_active:.2f}"
    )
