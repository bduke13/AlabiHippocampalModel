# %%
import numpy as np
import matplotlib.pyplot as plt
import pickle
import json
import os
import matplotlib.colors as mcolors
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
from tensorflow.python.ops.gen_math_ops import mul


def is_multimodal(x, y, activations, bandwidth_factor=0.1):
    """
    Check if the cell's activation pattern is multimodal using KDE and peak detection.

    Args:
    - x, y: Coordinates
    - activations: Cell activation values
    - bandwidth_factor: Controls the smoothing of the KDE

    Returns:
    - bool: True if multimodal, False otherwise
    """
    # Filter out zero activations
    mask = activations > 0
    if np.sum(mask) < 10:  # Need minimum points for meaningful KDE
        return False

    points = np.vstack([x[mask], y[mask]])
    values = activations[mask]

    # Weight the KDE by the activation values
    kde = gaussian_kde(points, weights=values, bw_method=bandwidth_factor)

    # Create a grid of points
    grid_points = 100
    xi, yi = np.mgrid[
        x.min() : x.max() : grid_points * 1j, y.min() : y.max() : grid_points * 1j
    ]
    coords = np.vstack([xi.ravel(), yi.ravel()])

    # Evaluate KDE on the grid
    density = kde(coords)
    density = density.reshape(grid_points, grid_points)

    # Find peaks in the density
    peaks, _ = find_peaks(density.max(axis=0), height=0.1 * density.max(), distance=10)

    return len(peaks) > 1


if __name__ == "__main__":
    # Create the directory if it doesn't exist
    output_dir = "visualizations/place_cell_images/"
    os.makedirs(output_dir, exist_ok=True)

    # Load the colors list
    with open("visualizations/colors.json", "r") as f:
        colors = json.load(f)

    # Convert hex colors to RGB format
    colors_rgb = [mcolors.to_rgb(c) for c in colors]

    # Load hmap data
    with open("hmap_x.pkl", "rb") as f:
        hmap_x = np.array(pickle.load(f))
    with open("hmap_y.pkl", "rb") as f:
        hmap_y = np.array(pickle.load(f))
    with open("hmap_z.pkl", "rb") as f:
        hmap_z = np.asarray(pickle.load(f))

    # Loop through selected cells and check for multimodality
    multimodal_cells = []
    for cell_index in range(hmap_z.shape[1]):
        if is_multimodal(hmap_x, hmap_y, hmap_z[:, cell_index]):
            multimodal_cells.append(cell_index)

    print(f"num multimodal cells is {len(multimodal_cells)}")
    print(multimodal_cells)

    # Save multimodal cell indices to cells.csv
    np.savetxt("cells.csv", multimodal_cells, fmt="%d")
    print("Saved multimodal cell indices to cells.csv")
