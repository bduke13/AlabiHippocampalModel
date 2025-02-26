import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity
from vis_utils import load_hmaps, convert_xzy_hmaps, OUTPUT_DIR

def compute_sai_msai(hmap_x, hmap_y, hmap_pcn, gridsize=50, d_th=1.5):
    """
    Compute the Spatial Aliasing Index (SAI) and Mean Spatial Aliasing Index (MSAI).

    Args:
        hmap_x: X coordinates.
        hmap_y: Y coordinates.
        hmap_pcn: Place cell activations (can include grid cell influence).
        gridsize: Number of bins for the grid.
        d_th: Distance threshold for considering bins as distant.

    Returns:
        sai: 2D array of SAI values per bin.
        msai: Mean SAI across all bins.
    """
    xedges = np.linspace(min(hmap_x), max(hmap_x), gridsize + 1)
    yedges = np.linspace(min(hmap_y), max(hmap_y), gridsize + 1)
    activations_per_bin = np.zeros((gridsize, gridsize, hmap_pcn.shape[1]))
    counts_per_bin = np.zeros((gridsize, gridsize))

    for i, (x, y, a) in enumerate(zip(hmap_x, hmap_y, hmap_pcn)):
        ix = np.digitize(x, xedges) - 1
        iy = np.digitize(y, yedges) - 1
        ix = np.clip(ix, 0, gridsize - 1)
        iy = np.clip(iy, 0, gridsize - 1)
        activations_per_bin[ix, iy] += a
        counts_per_bin[ix, iy] += 1

    # Suppress division-by-zero warnings and handle NaN values
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_activations = activations_per_bin / counts_per_bin[..., None]
    mean_activations = np.nan_to_num(mean_activations)

    # Report bins with zero counts for debugging
    zero_count_bins = (counts_per_bin == 0).sum()
    total_bins = counts_per_bin.size
    print(f"Number of bins with zero counts: {zero_count_bins} out of {total_bins}")

    coords = np.array([(x, y) for x in xedges[:-1] for y in yedges[:-1]])
    distances = cdist(coords, coords)
    activations_2d = mean_activations.reshape(-1, hmap_pcn.shape[1])
    similarities = cosine_similarity(activations_2d)

    sai = np.zeros(gridsize * gridsize)
    for i in range(len(sai)):
        distant_mask = distances[i] > d_th
        sai[i] = np.mean(similarities[i, distant_mask])

    sai = sai.reshape(gridsize, gridsize)
    msai = np.mean(sai)

    return sai, msai

def plot_sai(sai, hmap_x, hmap_y, msai, output_dir, label=""):
    """Plot and save the SAI heatmap."""
    plt.figure(figsize=(8, 8))
    plt.imshow(sai.T, origin="lower", extent=[min(hmap_x), max(hmap_x), min(hmap_y), max(hmap_y)])
    plt.colorbar(label="SAI")
    plt.title(f"SAI Map {label} (MSAI = {msai:.3f})")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.savefig(os.path.join(output_dir, f"sai_map_{label}.png"))
    plt.show()

if __name__ == "__main__":
    # Load data
    hmap_loc, hmap_pcn, hmap_grid = load_hmaps(["hmap_loc", "hmap_pcn", "hmap_grid"])
    hmap_x, _, hmap_y = convert_xzy_hmaps(hmap_loc)

    # Compute SAI/MSAI for place cells without grid cells
    sai_pcn, msai_pcn = compute_sai_msai(hmap_x, hmap_y, hmap_pcn)
    plot_sai(sai_pcn, hmap_x, hmap_y, msai_pcn, OUTPUT_DIR, "place_only")

    # Optionally compute SAI/MSAI for place cells influenced by grid cells
    # Assuming hmap_pcn_with_grid exists or can be computed
    # sai_pcn_with_grid, msai_pcn_with_grid = compute_sai_msai(hmap_x, hmap_y, hmap_pcn_with_grid)
    # plot_sai(sai_pcn_with_grid, hmap_x, hmap_y, msai_pcn_with_grid, OUTPUT_DIR, "place_with_grid")