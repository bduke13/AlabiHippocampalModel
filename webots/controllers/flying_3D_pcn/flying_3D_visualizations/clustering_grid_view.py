# %%
import numpy as np
import matplotlib.pyplot as plt
import os

# Use the same hmap loading utilities from your file.
from visualizations.vis_utils import load_hmaps, convert_xzy_hmaps


def weighted_mean_trimmed(data, weights, trim_percent=10):
    """Computes a weighted mean after trimming the lowest trim_percent of weights."""
    if np.sum(weights) == 0:
        return np.nan  # Avoid division by zero
    sorted_indices = np.argsort(weights)
    cutoff = int(len(weights) * (trim_percent / 100))
    trimmed_indices = sorted_indices[cutoff:]
    return np.sum(data[trimmed_indices] * weights[trimmed_indices]) / np.sum(
        weights[trimmed_indices]
    )


def get_valid_means(hmap_x, hmap_y, hmap_z, hmap_pcn, cell_indices=None):
    """
    Computes the place field centers (valid_means) using weighted_mean_trimmed.
    Only cells with non-NaN centers are kept.
    """
    num_cells = hmap_pcn.shape[-1]
    if cell_indices is None:
        cell_indices = list(range(num_cells))
    means = np.empty((len(cell_indices), 3))
    for i, cell_index in enumerate(cell_indices):
        try:
            x_mean = weighted_mean_trimmed(hmap_x, weights=hmap_pcn[:, cell_index])
            y_mean = weighted_mean_trimmed(hmap_y, weights=hmap_pcn[:, cell_index])
            z_mean = weighted_mean_trimmed(hmap_z, weights=hmap_pcn[:, cell_index])
            means[i] = x_mean, y_mean, z_mean
        except Exception as e:
            means[i] = np.nan, np.nan, np.nan
    valid_cells = ~np.isnan(means).any(axis=1)
    valid_means = means[valid_cells]
    return valid_means


def plot_projection_counts(
    points,
    idx1,
    idx2,
    bin_size=1.0,
    title="Projection",
    save_file=None,
    plot_centers=True,
):
    """
    Projects the 3D points onto a 2D plane defined by indices idx1 and idx2,
    subdivides the region into square bins of size bin_size x bin_size, computes
    the count per bin, and plots the grid with the count annotated in each square.

    Parameters:
    -----------
    points : numpy.ndarray
        3D points representing place field centers
    idx1, idx2 : int
        Indices for the dimensions to project onto (0=X, 1=Y, 2=Z)
    bin_size : float
        Size of each square bin
    title : str
        Plot title
    save_file : str, optional
        Path to save the figure
    plot_centers : bool, default=True
        Whether to plot blue x's at the exact location of each place field center
    """
    # Map indices to axis labels.
    axis_labels = {0: "X (m)", 1: "Y (m)", 2: "Z (m)"}

    # Extract the two coordinates.
    data1 = points[:, idx1]
    data2 = points[:, idx2]

    # Define bin edges (use floor and ceil to cover full range).
    bins1 = np.arange(np.floor(data1.min()), np.ceil(data1.max()) + bin_size, bin_size)
    bins2 = np.arange(np.floor(data2.min()), np.ceil(data2.max()) + bin_size, bin_size)

    # Compute the 2D histogram.
    H, xedges, yedges = np.histogram2d(data1, data2, bins=[bins1, bins2])

    # Create the plot.
    fig, ax = plt.subplots(figsize=(8, 6))
    mesh = ax.pcolormesh(xedges, yedges, H.T, shading="auto", cmap="viridis")
    fig.colorbar(mesh, ax=ax)

    # Annotate each square with the count.
    for i in range(len(xedges) - 1):
        for j in range(len(yedges) - 1):
            count = int(H[i, j])
            x_center = (xedges[i] + xedges[i + 1]) / 2
            y_center = (yedges[j] + yedges[j + 1]) / 2
            ax.text(
                x_center,
                y_center,
                str(count),
                color="white",
                ha="center",
                va="center",
                fontweight="bold",
            )

    # Plot blue x's at the exact location of each place field center if requested
    if plot_centers:
        ax.scatter(
            data1,
            data2,
            marker="x",
            color="blue",
            s=50,
            linewidth=1.5,
            label="Place field centers",
            zorder=3,
        )
        ax.legend(loc="upper right")

    ax.set_xlabel(axis_labels[idx1])
    ax.set_ylabel(axis_labels[idx2])
    ax.set_title(title)

    if save_file:
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        plt.savefig(save_file)
        print(f"Saved projection plot to {save_file}")
    plt.show()

    return H, xedges, yedges


def plot_histogram_of_counts(counts_dict, save_file=None):
    """
    Creates a histogram overlaying the distribution of counts per square from each view.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    # Flatten all counts to determine common binning.
    all_counts = np.concatenate([H.flatten() for H in counts_dict.values()])
    min_count = int(all_counts.min())
    max_count = int(all_counts.max())
    bins = (
        np.arange(min_count, max_count + 2) - 0.5
    )  # each integer count gets its own bin

    for view, H in counts_dict.items():
        flattened = H.flatten()
        ax.hist(flattened, bins=bins, alpha=0.5, label=view)

    ax.set_xlabel("Count per square")
    ax.set_ylabel("Number of squares")
    ax.set_title("Histogram of Place Field Counts in Squares")
    ax.legend()

    if save_file:
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        plt.savefig(save_file)
        print(f"Saved histogram plot to {save_file}")
    plt.show()


if __name__ == "__main__":
    # Load hmap data using your provided imports.
    hmap_loc, hmap_pcn = load_hmaps(hmap_names=["hmap_loc", "hmap_pcn"])
    hmap_x, hmap_z, hmap_y = convert_xzy_hmaps(hmap_loc)

    # Compute the 3D place field centers (valid_means).
    valid_means = get_valid_means(hmap_x, hmap_y, hmap_z, hmap_pcn, cell_indices=None)

    # Ensure output directory exists
    os.makedirs("outputs", exist_ok=True)

    # Dictionary to store counts matrices for each projection.
    counts_dict = {}

    # Set whether to plot centers or not (can be changed as needed)
    plot_centers = False

    # XY Projection (Top View): X vs Y.
    H_xy, xedges_xy, yedges_xy = plot_projection_counts(
        valid_means,
        idx1=0,
        idx2=1,
        bin_size=0.25,
        title="Top View (XY Projection)",
        save_file="outputs/projection_xy.png",
        plot_centers=plot_centers,
    )
    counts_dict["XY"] = H_xy

    # XZ Projection (Front View): X vs Z.
    H_xz, xedges_xz, zedges_xz = plot_projection_counts(
        valid_means,
        idx1=0,
        idx2=2,
        bin_size=0.25,
        title="Front View (XZ Projection)",
        save_file="outputs/projection_xz.png",
        plot_centers=plot_centers,
    )
    counts_dict["XZ"] = H_xz

    # YZ Projection (Side View): Y vs Z.
    H_yz, yedges_yz, zedges_yz = plot_projection_counts(
        valid_means,
        idx1=1,
        idx2=2,
        bin_size=0.25,
        title="Side View (YZ Projection)",
        save_file="outputs/projection_yz.png",
        plot_centers=plot_centers,
    )
    counts_dict["YZ"] = H_yz

    # Create a histogram overlaying the counts distribution for all three projections.
    plot_histogram_of_counts(counts_dict, save_file="outputs/histogram_counts.png")
