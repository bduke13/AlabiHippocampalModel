# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

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

    # Calculate statistics
    mean_x, mean_y, mean_z = np.mean(valid_x), np.mean(valid_y), np.mean(valid_z)
    std_x, std_y, std_z = np.std(valid_x), np.std(valid_y), np.std(valid_z)

    print("\nPlace Cell Center Distribution Statistics:")
    print(f"X centers - Mean: {mean_x:.4f}, Std: {std_x:.4f}")
    print(f"Y centers - Mean: {mean_y:.4f}, Std: {std_y:.4f}")
    print(f"Z centers - Mean: {mean_z:.4f}, Std: {std_z:.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].hist(valid_x, bins=30, edgecolor="black")
    axes[0].axvline(mean_x, color="r", linestyle="dashed", linewidth=1)
    axes[0].set_title(f"Histogram of X Centers\nMean: {mean_x:.4f}, Std: {std_x:.4f}")
    axes[0].set_xlabel("X Coordinate")
    axes[0].set_ylabel("Frequency")

    axes[1].hist(valid_y, bins=30, edgecolor="black")
    axes[1].axvline(mean_y, color="r", linestyle="dashed", linewidth=1)
    axes[1].set_title(f"Histogram of Y Centers\nMean: {mean_y:.4f}, Std: {std_y:.4f}")
    axes[1].set_xlabel("Y Coordinate")
    axes[1].set_ylabel("Frequency")

    axes[2].hist(valid_z, bins=30, edgecolor="black")
    axes[2].axvline(mean_z, color="r", linestyle="dashed", linewidth=1)
    axes[2].set_title(f"Histogram of Z Centers\nMean: {mean_z:.4f}, Std: {std_z:.4f}")
    axes[2].set_xlabel("Z Coordinate")
    axes[2].set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()


def plot_std_histograms(stds: np.ndarray):
    """
    Plot histograms for the weighted standard deviations (x, y, z) of place cells.
    Also calculates and displays the mean and standard deviation of these standard deviations.

    Args:
        stds: (num_cells x 3) array with the weighted standard deviations.
    """
    valid_std_x = stds[:, 0][~np.isnan(stds[:, 0])]
    valid_std_y = stds[:, 1][~np.isnan(stds[:, 1])]
    valid_std_z = stds[:, 2][~np.isnan(stds[:, 2])]

    # Calculate mean and std of the standard deviations
    mean_std_x = np.mean(valid_std_x)
    mean_std_y = np.mean(valid_std_y)
    mean_std_z = np.mean(valid_std_z)

    std_of_std_x = np.std(valid_std_x)
    std_of_std_y = np.std(valid_std_y)
    std_of_std_z = np.std(valid_std_z)

    # Print summary statistics
    print("\nSummary Statistics for Place Cell Field Sizes:")
    print(f"X dimension - Mean: {mean_std_x:.4f}, Std: {std_of_std_x:.4f}")
    print(f"Y dimension - Mean: {mean_std_y:.4f}, Std: {std_of_std_y:.4f}")
    print(f"Z dimension - Mean: {mean_std_z:.4f}, Std: {std_of_std_z:.4f}")

    # Calculate overall field size statistics
    all_stds = np.concatenate([valid_std_x, valid_std_y, valid_std_z])
    print(f"Overall - Mean: {np.mean(all_stds):.4f}, Std: {np.std(all_stds):.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot histograms with mean and std annotations
    axes[0].hist(valid_std_x, bins=30, edgecolor="black")
    axes[0].axvline(mean_std_x, color="r", linestyle="dashed", linewidth=1)
    axes[0].set_title(
        f"Histogram of Std X\nMean: {mean_std_x:.4f}, Std: {std_of_std_x:.4f}"
    )
    axes[0].set_xlabel("Std X")
    axes[0].set_ylabel("Frequency")

    axes[1].hist(valid_std_y, bins=30, edgecolor="black")
    axes[1].axvline(mean_std_y, color="r", linestyle="dashed", linewidth=1)
    axes[1].set_title(
        f"Histogram of Std Y\nMean: {mean_std_y:.4f}, Std: {std_of_std_y:.4f}"
    )
    axes[1].set_xlabel("Std Y")
    axes[1].set_ylabel("Frequency")

    axes[2].hist(valid_std_z, bins=30, edgecolor="black")
    axes[2].axvline(mean_std_z, color="r", linestyle="dashed", linewidth=1)
    axes[2].set_title(
        f"Histogram of Std Z\nMean: {mean_std_z:.4f}, Std: {std_of_std_z:.4f}"
    )
    axes[2].set_xlabel("Std Z")
    axes[2].set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()

    # Create a combined plot to compare distributions
    plt.figure(figsize=(10, 6))
    plt.hist(valid_std_x, bins=30, alpha=0.5, label="X Std")
    plt.hist(valid_std_y, bins=30, alpha=0.5, label="Y Std")
    plt.hist(valid_std_z, bins=30, alpha=0.5, label="Z Std")
    plt.axvline(
        mean_std_x,
        color="r",
        linestyle="dashed",
        linewidth=1,
        label=f"Mean X: {mean_std_x:.4f}",
    )
    plt.axvline(
        mean_std_y,
        color="g",
        linestyle="dashed",
        linewidth=1,
        label=f"Mean Y: {mean_std_y:.4f}",
    )
    plt.axvline(
        mean_std_z,
        color="b",
        linestyle="dashed",
        linewidth=1,
        label=f"Mean Z: {mean_std_z:.4f}",
    )
    plt.xlabel("Standard Deviation")
    plt.ylabel("Frequency")
    plt.title("Comparison of Place Cell Field Size Distributions")
    plt.legend()
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

    # Calculate number of valid place cells
    valid_cells = np.sum(~np.isnan(centers[:, 0]))
    total_cells = centers.shape[0]
    print(
        f"\nAnalyzing {valid_cells} valid place cells out of {total_cells} total cells"
    )

    # Optionally, print computed values for a few cells.
    num_print = min(10, centers.shape[0])
    print("\nWeighted Centers and Standard Deviations for the first few place cells:")
    for cell in range(num_print):
        print(f"Cell {cell}: Center = {centers[cell]}, Std = {stds[cell]}")

    # Plot histograms for weighted centers.
    plot_center_histograms(centers)

    # Plot histograms for standard deviations to visualize the spread/skew.
    plot_std_histograms(stds)

    # Calculate and print additional statistics about field sizes
    analyze_field_sizes(stds)


def analyze_field_sizes(stds: np.ndarray):
    """
    Perform detailed analysis of place cell field sizes.

    Args:
        stds: (num_cells x 3) array with the weighted standard deviations.
    """
    valid_std_x = stds[:, 0][~np.isnan(stds[:, 0])]
    valid_std_y = stds[:, 1][~np.isnan(stds[:, 1])]
    valid_std_z = stds[:, 2][~np.isnan(stds[:, 2])]

    # Calculate field volumes (assuming ellipsoidal fields)
    # Volume of ellipsoid = 4/3 * pi * a * b * c
    field_volumes = (4 / 3) * np.pi * valid_std_x * valid_std_y * valid_std_z

    # Calculate aspect ratios
    xy_ratio = valid_std_x / valid_std_y
    xz_ratio = valid_std_x / valid_std_z
    yz_ratio = valid_std_y / valid_std_z

    print("\nDetailed Field Size Analysis:")
    print(
        f"Field Volumes - Mean: {np.mean(field_volumes):.4f}, Std: {np.std(field_volumes):.4f}"
    )
    print(
        f"Field Volumes - Min: {np.min(field_volumes):.4f}, Max: {np.max(field_volumes):.4f}"
    )
    print(f"Field Volumes - Median: {np.median(field_volumes):.4f}")

    print("\nAspect Ratios:")
    print(f"X/Y Ratio - Mean: {np.mean(xy_ratio):.4f}, Std: {np.std(xy_ratio):.4f}")
    print(f"X/Z Ratio - Mean: {np.mean(xz_ratio):.4f}, Std: {np.std(xz_ratio):.4f}")
    print(f"Y/Z Ratio - Mean: {np.mean(yz_ratio):.4f}, Std: {np.std(yz_ratio):.4f}")

    # Statistical tests for isotropy (are fields spherical or elongated?)
    print("\nIsotropy Tests (p-values):")
    print(f"X vs Y dimensions: {stats.ttest_ind(valid_std_x, valid_std_y).pvalue:.4f}")
    print(f"X vs Z dimensions: {stats.ttest_ind(valid_std_x, valid_std_z).pvalue:.4f}")
    print(f"Y vs Z dimensions: {stats.ttest_ind(valid_std_y, valid_std_z).pvalue:.4f}")

    # Plot field volumes
    plt.figure(figsize=(10, 6))
    plt.hist(field_volumes, bins=30, edgecolor="black")
    plt.axvline(
        np.mean(field_volumes),
        color="r",
        linestyle="dashed",
        linewidth=1,
        label=f"Mean: {np.mean(field_volumes):.4f}",
    )
    plt.axvline(
        np.median(field_volumes),
        color="g",
        linestyle="dashed",
        linewidth=1,
        label=f"Median: {np.median(field_volumes):.4f}",
    )
    plt.xlabel("Field Volume")
    plt.ylabel("Frequency")
    plt.title("Distribution of Place Cell Field Volumes")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot aspect ratios
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].hist(xy_ratio, bins=30, edgecolor="black")
    axes[0].axvline(
        1.0, color="k", linestyle="dashed", linewidth=1, label="Equal ratio"
    )
    axes[0].axvline(
        np.mean(xy_ratio),
        color="r",
        linestyle="dashed",
        linewidth=1,
        label=f"Mean: {np.mean(xy_ratio):.4f}",
    )
    axes[0].set_title("X/Y Aspect Ratio")
    axes[0].set_xlabel("X/Y Ratio")
    axes[0].set_ylabel("Frequency")
    axes[0].legend()

    axes[1].hist(xz_ratio, bins=30, edgecolor="black")
    axes[1].axvline(
        1.0, color="k", linestyle="dashed", linewidth=1, label="Equal ratio"
    )
    axes[1].axvline(
        np.mean(xz_ratio),
        color="r",
        linestyle="dashed",
        linewidth=1,
        label=f"Mean: {np.mean(xz_ratio):.4f}",
    )
    axes[1].set_title("X/Z Aspect Ratio")
    axes[1].set_xlabel("X/Z Ratio")
    axes[1].set_ylabel("Frequency")
    axes[1].legend()

    axes[2].hist(yz_ratio, bins=30, edgecolor="black")
    axes[2].axvline(
        1.0, color="k", linestyle="dashed", linewidth=1, label="Equal ratio"
    )
    axes[2].axvline(
        np.mean(yz_ratio),
        color="r",
        linestyle="dashed",
        linewidth=1,
        label=f"Mean: {np.mean(yz_ratio):.4f}",
    )
    axes[2].set_title("Y/Z Aspect Ratio")
    axes[2].set_xlabel("Y/Z Ratio")
    axes[2].set_ylabel("Frequency")
    axes[2].legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
