# %%
import numpy as np
import pickle


def get_cell_statistics(hmap_z):
    """
    Calculate and print basic statistics for each cell.

    Args:
        hmap_z (np.array): activation values

    Returns:
        dict: Dictionary containing basic statistics for each cell
    """
    # Calculate basic statistics for each cell
    total_activation_per_cell = np.sum(hmap_z, axis=0)
    mean_activation_per_cell = np.mean(hmap_z, axis=0)
    max_activation_per_cell = np.max(hmap_z, axis=0)
    min_activation_per_cell = np.min(hmap_z, axis=0)

    # Print statistics for each cell
    print("\nPlace Cell Activation Statistics:")
    print(f"{'Cell Index':<10} {'Min':<10} {'Mean':<10} {'Max':<10} {'Total':<10}")
    print("-" * 50)
    for i in range(hmap_z.shape[1]):
        print(
            f"{i:<10} {min_activation_per_cell[i]:<10.3f} {mean_activation_per_cell[i]:<10.3f} {max_activation_per_cell[i]:<10.3f} {total_activation_per_cell[i]:<10.3f}"
        )
    print("\n")

    return {
        "total": total_activation_per_cell,
        "mean": mean_activation_per_cell,
        "max": max_activation_per_cell,
        "min": min_activation_per_cell,
    }


def analyze_far_field_activations(
    hmap_x, hmap_y, hmap_z, distance_threshold=2.0, verbose=False
):
    """
    Analyze far field activations and spatial distributions.

    Args:
        hmap_x (np.array): x coordinates
        hmap_y (np.array): y coordinates
        hmap_z (np.array): activation values
        distance_threshold (float): threshold for considering points as close pairs
        verbose (bool): If True, print detailed analysis information

    Returns:
        list: List of dictionaries containing far field activation statistics
    """
    if verbose:
        print(
            f"\nAnalyzing spatial distribution of activations for each cell (threshold distance = {distance_threshold}):"
        )
        print(
            f"{'Cell':<6} {'#Points':<8} {'Total Far Act':<12} {'#Far Pairs':<12} {'Avg Act':<10}"
        )
        print("-" * 50)

    distance_stats = []
    for cell_idx in range(hmap_z.shape[1]):
        cell_activations = hmap_z[:, cell_idx]
        nonzero_mask = cell_activations > 0

        stats = {
            "cell_idx": cell_idx,
            "num_points": 0,
            "total_far_activation": 0.0,
            "num_far_pairs": 0,
            "avg_activation": 0.0,
        }

        if np.sum(nonzero_mask) > 1:
            nonzero_x = hmap_x[nonzero_mask]
            nonzero_y = hmap_y[nonzero_mask]
            nonzero_activations = cell_activations[nonzero_mask]

            stats["num_points"] = len(nonzero_x)

            for i in range(stats["num_points"]):
                for j in range(i + 1, stats["num_points"]):
                    dist = np.sqrt(
                        (nonzero_x[i] - nonzero_x[j]) ** 2
                        + (nonzero_y[i] - nonzero_y[j]) ** 2
                    )
                    if dist > distance_threshold:
                        # Weight by product of activations for points beyond threshold
                        weight = nonzero_activations[i] * nonzero_activations[j]
                        stats["total_far_activation"] += weight
                        stats["num_far_pairs"] += 1

            stats["avg_activation"] = (
                stats["total_far_activation"] / stats["num_far_pairs"]
                if stats["num_far_pairs"] > 0
                else 0
            )

        if verbose:
            print(
                f"{stats['cell_idx']:<6} {stats['num_points']:<8} "
                f"{stats['total_far_activation']:< 12.2f} {stats['num_far_pairs']:<12} "
                f"{stats['avg_activation']:<10.2f}"
            )

        distance_stats.append(stats)

    # Sort cells by total distance and save indices
    sorted_indices = [
        d["cell_idx"]
        for d in sorted(
            distance_stats, key=lambda x: x["total_far_activation"], reverse=True
        )
    ]

    # Save sorted cell indices to cells.csv
    np.savetxt("cells.csv", sorted_indices, fmt="%d")
    if verbose:
        print("Saved cell indices to cells.csv (ordered by total distance)")

        # Calculate total and average far-field activation across all cells
        total_far_field = sum(d["total_far_activation"] for d in distance_stats)
        avg_far_field = total_far_field / len(distance_stats)
        print(f"\nTotal far-field activation across all cells: {total_far_field:.2f}")
        print(f"Average far-field activation per cell: {avg_far_field:.2f}")

    # Sort distance_stats by total_far_activation
    return sorted(distance_stats, key=lambda x: x["total_far_activation"], reverse=True)


def analyze_place_cells(hmap_x, hmap_y, hmap_z, distance_threshold=2.0, verbose=False):
    """
    Analyze place cell statistics and spatial distributions.

    Args:
        hmap_x (np.array): x coordinates
        hmap_y (np.array): y coordinates
        hmap_z (np.array): activation values
        distance_threshold (float): threshold for considering points as close pairs

    Returns:
        dict: Dictionary containing the analysis results
    """
    basic_stats = get_cell_statistics(hmap_z)
    distance_stats = analyze_far_field_activations(
        hmap_x, hmap_y, hmap_z, distance_threshold, verbose=verbose
    )

    return {
        "basic_stats": basic_stats,
        "distance_stats": distance_stats,
    }


# %%
if __name__ == "__main__":
    prefix = "controllers/bek_controller/paper_data/2D/world_0/trial_0/"
    # Load data and run analysis
    from controllers.bek_controller.visualizations.analysis_utils import (
        load_heatmap_data,
    )

    # Load data
    hmap_x, hmap_y, hmap_z = load_heatmap_data(prefix=prefix)

    # Run analysis
    analysis_results = analyze_place_cells(
        hmap_x, hmap_y, hmap_z, distance_threshold=2.0
    )

    # %%
    # Plot the top 5 cells with highest far field activation
    import matplotlib.pyplot as plt

    # Create a figure with 2x3 subplots
    fig, axs = plt.subplots(3, 3, figsize=(12, 12))
    axs = axs.ravel()

    # Get the top 5 cells from our sorted results
    top_5_cells = [stat["cell_idx"] for stat in analysis_results["distance_stats"][:21]]

    for idx, cell_idx in enumerate(top_5_cells):
        # Get activation data for this cell
        cell_data = hmap_z[:, cell_idx]

        # Create scatter plot
        scatter = axs[idx].scatter(
            hmap_x, hmap_y, c=cell_data, cmap="viridis", s=50, alpha=0.6
        )

        # Add colorbar
        plt.colorbar(scatter, ax=axs[idx])

        # Add title with cell info
        cell_stats = analysis_results["distance_stats"][idx]
        axs[idx].set_title(
            f"Cell {cell_idx}\nFar Field Act: {cell_stats['total_far_activation']:.2f}"
        )

        # Add labels
        axs[idx].set_xlabel("X Position")
        axs[idx].set_ylabel("Y Position")

    # Adjust layout
    plt.tight_layout()
    plt.show()

    # %%
    # Create a new figure for the histogram
    plt.figure(figsize=(10, 6))

    # Get total far field activations for all cells
    total_activations = [
        stat["total_far_activation"] for stat in analysis_results["distance_stats"]
    ]
    # Filter out zeros
    non_zero_activations = [act for act in total_activations if act > 0]

    # Create histogram
    plt.hist(non_zero_activations, bins=30, edgecolor="black")
    plt.title("Distribution of Non-Zero Total Far Field Activations")
    plt.xlabel("Total Far Field Activation")
    plt.ylabel("Count")

    # Add grid for better readability
    plt.grid(True, alpha=0.3)

    # Show the plot
    plt.show()

    # %%
    # Create combined heatmaps of top 10% cells
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Calculate number of cells for top 10%
    num_cells = hmap_z.shape[1]
    top_n = int(np.ceil(num_cells * 0.1))  # top 10%
    top_cells = [
        stat["cell_idx"] for stat in analysis_results["distance_stats"][:top_n]
    ]

    # Create a combined activation map
    combined_activation = np.zeros_like(hmap_x)
    for cell_idx in top_cells:
        # Add each cell's activation to the combined map
        combined_activation += hmap_z[:, cell_idx]

    # Plot raw sum
    scatter1 = ax1.scatter(
        hmap_x, hmap_y, c=combined_activation, cmap="viridis", s=50, alpha=0.6
    )
    plt.colorbar(scatter1, ax=ax1, label="Raw Combined Activation")
    ax1.set_title(f"Raw Sum of Top {top_n} Cells\nBefore Normalization")
    ax1.set_xlabel("X Position")
    ax1.set_ylabel("Y Position")
    ax1.grid(True, alpha=0.3)

    # Plot normalized version
    normalized_activation = combined_activation / combined_activation.max()
    scatter2 = (
        ax2.scatter(
            hmap_x, hmap_y, c=normalized_activation, cmap="viridis", s=50, alpha=0.6
        )
        * 2
    )
    plt.colorbar(scatter2, ax=ax2, label="Normalized Combined Activation")
    ax2.set_title(f"Normalized Sum of Top {top_n} Cells\nDivided by Maximum")
    ax2.set_xlabel("X Position")
    ax2.set_ylabel("Y Position")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # %%
    # Create histogram of all activation values
    plt.figure(figsize=(12, 6))

    # Flatten all activation values into 1D array
    all_activations = hmap_z.flatten()

    # Filter out zeros for better visualization (optional)
    non_zero_activations = all_activations[all_activations > 0]

    # Create histogram
    plt.hist(non_zero_activations, bins=50, edgecolor="black")
    plt.title("Distribution of All Non-Zero Activation Values")
    plt.xlabel("Activation Value")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)

    # Add some statistics as text
    plt.text(
        0.98,
        0.95,
        f"Total points: {len(all_activations):,}\n"
        f"Non-zero points: {len(non_zero_activations):,}\n"
        f"Mean: {np.mean(non_zero_activations):.3f}\n"
        f"Max: {np.max(non_zero_activations):.3f}",
        transform=plt.gca().transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.show()
