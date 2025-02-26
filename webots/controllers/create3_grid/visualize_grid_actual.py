import os
from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate2d
import sys
import torch

# Set project root to the current working directory (Alex-Dev-Updated)
project_root = Path(os.getcwd())
sys.path.append(str(project_root))

# --- Configuration ---
CONTROLLER_PATH_PREFIX = "webots/controllers/"
CONTROLLER_NAME = "create3_grid"
WORLD_NAME = "A_iCreateBotEasy_10"

HMAP_DIR = os.path.join(CONTROLLER_PATH_PREFIX, CONTROLLER_NAME, "pkl", WORLD_NAME, "hmaps")
NETWORKS_DIR = os.path.join(CONTROLLER_PATH_PREFIX, CONTROLLER_NAME, "pkl", WORLD_NAME, "networks")

# --- Load Data ---
# Load grid cell layer parameters
with open(os.path.join(NETWORKS_DIR, "gcn.pkl"), "rb") as f:
    gcn = pickle.load(f)

# Load actual grid cell activations
with open(os.path.join(HMAP_DIR, "hmap_gcn.pkl"), "rb") as f:
    hmap_gcn = pickle.load(f)

# Load robot position history
with open(os.path.join(HMAP_DIR, "hmap_loc.pkl"), "rb") as f:
    hmap_loc = pickle.load(f)

# Convert to NumPy arrays if necessary (e.g., from torch tensors)
if isinstance(hmap_gcn, torch.Tensor):
    hmap_gcn = hmap_gcn.cpu().numpy()
if isinstance(hmap_loc, torch.Tensor):
    hmap_loc = hmap_loc.numpy()
else:
    hmap_loc = np.array(hmap_loc)

# --- Define Environment Range ---
x_min = hmap_loc[:, 0].min() - 0.5
x_max = hmap_loc[:, 0].max() + 0.5
y_min = hmap_loc[:, 2].min() - 0.5  # Z-axis is the second horizontal dimension in Webots
y_max = hmap_loc[:, 2].max() + 0.5

# --- Set Up Bins ---
num_bins = 150
x_edges = np.linspace(x_min, x_max, num_bins + 1)
y_edges = np.linspace(y_min, y_max, num_bins + 1)

# --- Compute Dwell Time Histogram ---
dwell_hist, _, _ = np.histogram2d(
    hmap_loc[:, 0], hmap_loc[:, 2], bins=[x_edges, y_edges]
)

# --- Visualize Firing Rate Maps and Autocorrelation ---
for m in range(gcn.num_modules):
    cell_idx = m * gcn.num_cells_per_module  # First cell of each module
    activations = hmap_gcn[:, cell_idx]

    # Compute activation histogram
    activation_hist, _, _ = np.histogram2d(
        hmap_loc[:, 0], hmap_loc[:, 2], bins=[x_edges, y_edges], weights=activations
    )

    # Compute firing rate (avoid division by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        firing_rate = activation_hist / dwell_hist
        firing_rate[dwell_hist == 0] = np.nan

    # --- Plot Firing Rate Map ---
    plt.figure(figsize=(6, 5))
    plt.imshow(
        firing_rate.T,  # Transpose to match (x, y) orientation
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        origin='lower',
        cmap='jet',
    )
    plt.title(f'Actual Firing Rate Map - Module {m}, Cell {cell_idx}')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.colorbar(label='Firing Rate')
    plt.tight_layout()

    # --- Compute and Plot Autocorrelation ---
    # Replace NaNs with 0 for autocorrelation computation
    firing_rate_no_nan = np.nan_to_num(firing_rate, nan=0.0)

    # Compute 2D autocorrelation
    autocorr = correlate2d(firing_rate_no_nan, firing_rate_no_nan, mode='same')

    # Calculate spatial step sizes for lag extent
    dx = (x_max - x_min) / num_bins
    dy = (y_max - y_min) / num_bins

    # Define extent for autocorrelation plot (centered at zero lag)
    half_bins = num_bins // 2
    extent = [-half_bins * dx, half_bins * dx, -half_bins * dy, half_bins * dy]

    # Plot autocorrelation
    plt.figure(figsize=(6, 5))
    plt.imshow(
        autocorr.T,  # Transpose to match (x, y) orientation
        extent=extent,
        origin='lower',
        cmap='RdBu',
    )
    plt.title(f'Autocorrelation - Module {m}, Cell {cell_idx}')
    plt.xlabel('Lag X (m)')
    plt.ylabel('Lag Y (m)')
    plt.colorbar(label='Autocorrelation')
    plt.tight_layout()

# --- Plot Robot Path ---
plt.figure(figsize=(6, 5))
plt.plot(hmap_loc[:, 0], hmap_loc[:, 2], 'k-', linewidth=0.5)
plt.title('Robot Path')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.grid(True)
plt.tight_layout()

plt.show()