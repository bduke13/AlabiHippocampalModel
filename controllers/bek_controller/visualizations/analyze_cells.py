# %%
import numpy as np
import pickle


def analyze_place_cells(hmap_x, hmap_y, hmap_z, distance_threshold=2.0):
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

    # Calculate pairwise distances
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
    print("Saved cell indices to cells.csv (ordered by total distance)")

    # Calculate total and average far-field activation across all cells
    total_far_field = sum(d["total_far_activation"] for d in distance_stats)
    avg_far_field = total_far_field / len(distance_stats)
    print(f"\nTotal far-field activation across all cells: {total_far_field:.2f}")
    print(f"Average far-field activation per cell: {avg_far_field:.2f}")

    return {
        "basic_stats": {
            "total": total_activation_per_cell,
            "mean": mean_activation_per_cell,
            "max": max_activation_per_cell,
            "min": min_activation_per_cell,
        },
        "distance_stats": distance_stats,
    }


if __name__ == "__main__":
    # Load data
    with open("hmap_x.pkl", "rb") as f:
        hmap_x = np.array(pickle.load(f))
    with open("hmap_y.pkl", "rb") as f:
        hmap_y = np.array(pickle.load(f))
    with open("hmap_z.pkl", "rb") as f:
        hmap_z = np.asarray(pickle.load(f))

    # Run analysis
    analysis_results = analyze_place_cells(
        hmap_x, hmap_y, hmap_z, distance_threshold=2.0
    )
