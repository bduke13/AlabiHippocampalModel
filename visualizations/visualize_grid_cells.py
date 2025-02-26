# visualize_grid_cells.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate2d
import os
from vis_utils import load_hmaps, OUTPUT_DIR

def compute_firing_rate_map(hmap_loc, hmap_grid, neuron_idx, bins=50):
    """
    Compute the firing rate map for a specific grid neuron.

    Args:
        hmap_loc (np.ndarray): Robot positions, shape (num_steps, 3), [x, y, z].
        hmap_grid (np.ndarray): Grid neuron activations, shape (num_steps, num_neurons).
        neuron_idx (int): Index of the grid neuron to visualize.
        bins (int): Number of bins for the spatial grid.

    Returns:
        tuple: (firing_rate_map, x_edges, y_edges)
            - firing_rate_map: 2D array of average firing rates per bin.
            - x_edges: Bin edges along x-axis.
            - y_edges: Bin edges along z-axis (y is vertical in Webots).
    """
    activations = hmap_grid[:, neuron_idx]
    x_positions = hmap_loc[:, 0]  # x coordinates
    z_positions = hmap_loc[:, 2]  # z coordinates (y is vertical)

    x_edges = np.linspace(np.min(x_positions), np.max(x_positions), bins + 1)
    y_edges = np.linspace(np.min(z_positions), np.max(z_positions), bins + 1)

    # Occupancy map (time spent in each bin)
    hist, x_bins, y_bins = np.histogram2d(x_positions, z_positions, bins=[x_edges, y_edges])

    # Sum of activations per bin
    activation_sum, _, _ = np.histogram2d(x_positions, z_positions, bins=[x_edges, y_edges], weights=activations)

    # Firing rate map (average activation per bin)
    firing_rate_map = activation_sum / (hist + 1e-5)  # Avoid division by zero
    return firing_rate_map, x_edges, y_edges

def plot_firing_rate_map(firing_rate_map, x_edges, y_edges, neuron_idx):
    """
    Plot and save the firing rate map for a grid neuron.

    Args:
        firing_rate_map (np.ndarray): 2D array of firing rates.
        x_edges (np.ndarray): Bin edges along x-axis.
        y_edges (np.ndarray): Bin edges along z-axis.
        neuron_idx (int): Index of the neuron being visualized.
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(firing_rate_map.T, origin='lower', extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]])
    plt.colorbar(label='Firing Rate')
    plt.title(f'Grid Neuron {neuron_idx} Firing Rate Map')
    plt.xlabel('X Position')
    plt.ylabel('Z Position')
    plt.savefig(os.path.join(OUTPUT_DIR, f'firing_rate_map_neuron_{neuron_idx}.png'))
    plt.show()

def compute_autocorrelogram(firing_rate_map):
    """
    Compute the autocorrelogram of the firing rate map.

    Args:
        firing_rate_map (np.ndarray): 2D array of firing rates.

    Returns:
        np.ndarray: 2D autocorrelation of the firing rate map.
    """
    autocorr = correlate2d(firing_rate_map, firing_rate_map, mode='full')
    return autocorr

def plot_autocorrelogram(autocorr, neuron_idx):
    """
    Plot and save the autocorrelogram of the firing rate map.

    Args:
        autocorr (np.ndarray): 2D autocorrelation array.
        neuron_idx (int): Index of the neuron being visualized.
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(autocorr, origin='lower')
    plt.colorbar(label='Autocorrelation')
    plt.title(f'Autocorrelogram of Grid Neuron {neuron_idx}')
    plt.savefig(os.path.join(OUTPUT_DIR, f'autocorrelogram_neuron_{neuron_idx}.png'))
    plt.show()

def visualize_grid_cell(neuron_idx, bins=50):
    """
    Load data and visualize firing rate map and autocorrelogram for a grid neuron.

    Args:
        neuron_idx (int): Index of the grid neuron to visualize.
        bins (int): Number of bins for the spatial grid (default: 50).
    """
    # Load history maps using vis_utils
    hmap_loc, hmap_grid = load_hmaps(["hmap_loc", "hmap_gc"])
    
    # Compute and plot firing rate map
    firing_rate_map, x_edges, y_edges = compute_firing_rate_map(hmap_loc, hmap_grid, neuron_idx, bins)
    plot_firing_rate_map(firing_rate_map, x_edges, y_edges, neuron_idx)
    
    # Compute and plot autocorrelogram
    autocorr = compute_autocorrelogram(firing_rate_map)
    plot_autocorrelogram(autocorr, neuron_idx)

if __name__ == "__main__":
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Example: Visualize neurons 0, 10, 20
    neurons_to_visualize = [0, 10, 20]
    for neuron_idx in neurons_to_visualize:
        print(f"Visualizing grid neuron {neuron_idx}...")
        visualize_grid_cell(neuron_idx)