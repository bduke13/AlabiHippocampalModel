#!/usr/bin/env python3
"""
Grid Cosine Similarity Analysis

Partitions the environment into a grid, computes average Place Cell activation vectors
per cell, then calculates pairwise distances and cosine similarities between cell representations.
"""

import os
import csv
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Define world name and directories for hmap files
WORLD_NAME = "iCreateBotCross_Camera_env"
BASE_DIR = Path(__file__).resolve().parent
HMAP_DIR = BASE_DIR / "pkl" / WORLD_NAME / "hmaps"

# OUTPUT_DIR imported from vis_utils for saving results
from vis_utils import OUTPUT_DIR


def get_grid_indices(x, y, x_min, x_max, y_min, y_max, num_cells=10):
    """
    Determine grid cell indices (i, j) for a position (x, y) in a square grid.
    The grid spans from x_min to x_max and y_min to y_max, divided into num_cells per dimension.
    """
    cell_width = (x_max - x_min) / num_cells
    cell_height = (y_max - y_min) / num_cells

    # Clamp x, y within boundaries
    x_clamped = np.clip(x, x_min, x_max - 1e-6)
    y_clamped = np.clip(y, y_min, y_max - 1e-6)

    i = int((x_clamped - x_min) // cell_width)
    j = int((y_clamped - y_min) // cell_height)
    return i, j


def compute_cell_representations(hmap_loc, hmap_pcn, num_cells=10, x_range=(-5, 5), y_range=(-5, 5)):
    """
    Partition the environment into a num_cells x num_cells grid.
    For each grid cell, compute the average place cell activation vector for all simulation steps
    whose (x, y) positions fall into that cell.

    Returns:
        cell_centers: dict mapping (i,j) to the cell center coordinates.
        cell_vectors: dict mapping (i,j) to the average activation vector (or None if no data).
    """
    x_min, x_max = x_range
    y_min, y_max = y_range

    # Generate center coordinates for each cell
    cell_centers = {}
    for i in range(num_cells):
        for j in range(num_cells):
            center_x = x_min + (i + 0.5) * (x_max - x_min) / num_cells
            center_y = y_min + (j + 0.5) * (y_max - y_min) / num_cells
            cell_centers[(i, j)] = (center_x, center_y)

    # Sum activation vectors per cell
    cell_vectors_sum = {}
    cell_counts = {}
    num_steps = hmap_loc.shape[0]
    for idx in range(num_steps):
        x = hmap_loc[idx, 0]
        y = hmap_loc[idx, 2]
        i, j = get_grid_indices(x, y, x_min, x_max, y_min, y_max, num_cells)
        if (i, j) not in cell_vectors_sum:
            cell_vectors_sum[(i, j)] = np.zeros(hmap_pcn.shape[1])
            cell_counts[(i, j)] = 0
        cell_vectors_sum[(i, j)] += hmap_pcn[idx, :]
        cell_counts[(i, j)] += 1

    # Compute average activation vectors
    cell_vectors = {}
    for key in cell_centers.keys():
        if key in cell_counts and cell_counts[key] > 0:
            cell_vectors[key] = cell_vectors_sum[key] / cell_counts[key]
        else:
            cell_vectors[key] = None
    return cell_centers, cell_vectors


def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors."""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


def compute_pairwise_metrics(cell_centers, cell_vectors):
    """
    For all pairs of grid cells with valid activation vectors,
    compute the Euclidean distance between their centers and the cosine similarity between their vectors.

    Returns:
        distances: np.ndarray of distances.
        cos_sims: np.ndarray of cosine similarities.
    """
    distances = []
    cos_sims = []
    keys = list(cell_centers.keys())
    for idx1 in range(len(keys)):
        for idx2 in range(idx1 + 1, len(keys)):
            key1 = keys[idx1]
            key2 = keys[idx2]
            vec1 = cell_vectors[key1]
            vec2 = cell_vectors[key2]
            if vec1 is None or vec2 is None:
                continue
            center1 = np.array(cell_centers[key1])
            center2 = np.array(cell_centers[key2])
            dist = np.linalg.norm(center1 - center2)
            sim = cosine_similarity(vec1, vec2)
            distances.append(dist)
            cos_sims.append(sim)
    return np.array(distances), np.array(cos_sims)


def plot_scatter(distances, cos_sims, output_dir):
    """
    Plot scatter of cosine similarity vs distance and save to output_dir.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(distances, cos_sims, alpha=0.6)
    plt.xlabel("Distance between grid centers (m)")
    plt.ylabel("Cosine similarity")
    plt.title("Scatter Plot: Cosine Similarity vs Distance")
    plt.grid(True)

    os.makedirs(output_dir, exist_ok=True)
    scatter_path = os.path.join(output_dir, "scatter_plot.png")
    plt.savefig(scatter_path, dpi=300, bbox_inches="tight")
    print(f"Scatter plot saved to {scatter_path}")
    plt.show()


def main():
    # Load history maps directly from local pkl files
    with open(HMAP_DIR / "hmap_loc.pkl", "rb") as f:
        hmap_loc = pickle.load(f)
    with open(HMAP_DIR / "hmap_pcn.pkl", "rb") as f:
        hmap_pcn = pickle.load(f)

    # Compute grid-based representations
    cell_centers, cell_vectors = compute_cell_representations(
        hmap_loc, hmap_pcn, num_cells=10, x_range=(-5, 5), y_range=(-5, 5)
    )

    # Compute pairwise distances and cosine similarities
    distances, cos_sims = compute_pairwise_metrics(cell_centers, cell_vectors)

    # Save metrics to CSV
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    metrics_csv_path = os.path.join(OUTPUT_DIR, "grid_pairwise_metrics.csv")
    with open(metrics_csv_path, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Distance", "Cosine Similarity"])
        for d, cs in zip(distances, cos_sims):
            writer.writerow([d, cs])
    print(f"Pairwise metrics saved to {metrics_csv_path}")

    # Plot scatter
    plot_scatter(distances, cos_sims, OUTPUT_DIR)


if __name__ == "__main__":
    main()