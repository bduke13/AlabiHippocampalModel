import sys
import os
from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate2d

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
# Load grid cell layer (gcn) from saved file
with open(os.path.join(NETWORKS_DIR, "gcn.pkl"), "rb") as f:
    gcn = pickle.load(f)

# Load robot position history (hmap_loc)
with open(os.path.join(HMAP_DIR, "hmap_loc.pkl"), "rb") as f:
    hmap_loc = pickle.load(f)

# --- Define Environment Range ---
# Determine the range of the environment from hmap_loc with padding
x_min = hmap_loc[:, 0].min() - 0.5
x_max = hmap_loc[:, 0].max() + 0.5
y_min = hmap_loc[:, 2].min() - 0.5  # Z-axis is the second horizontal dimension in Webots
y_max = hmap_loc[:, 2].max() + 0.5

# --- Create Position Grid ---
num_points = 100
x = np.linspace(x_min, x_max, num_points)
y = np.linspace(y_min, y_max, num_points)
X, Y = np.meshgrid(x, y, indexing='ij')

# --- Grid Cell Parameters ---
# Preferred directions for oscillators (0°, 60°, 120°)
theta_i = np.deg2rad([0, 60, 120])

# --- Visualize Grid Cells ---
num_modules = gcn.num_modules
for m in range(num_modules):
    lambda_m = gcn.grid_spacings[m].item()  # Grid spacing for module m
    k = 0  # Select the first cell in the module
    phi = gcn.phase_offsets[m, k, :].cpu().numpy()  # Phase offsets for the cell

    # Compute activation over the grid
    activation = np.ones_like(X)
    for i in range(3):
        proj = X * np.cos(theta_i[i]) + Y * np.sin(theta_i[i])
        activation *= np.cos((2 * np.pi / lambda_m) * proj + phi[i])

    # Plot firing rate map
    plt.figure(figsize=(6, 5))
    plt.pcolormesh(x, y, activation, cmap='jet', shading='auto')
    plt.title(f'Firing Rate Map - Module {m}, λ={lambda_m:.2f} m')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.colorbar(label='Activation')
    plt.tight_layout()

    # Compute and plot autocorrelation
    autocorr = correlate2d(activation, activation, mode='full')
    center = autocorr.shape[0] // 2
    autocorr_central = autocorr[
        center - num_points // 2 : center + num_points // 2 + 1,
        center - num_points // 2 : center + num_points // 2 + 1
    ]

    # Calculate spatial step sizes for lag extent
    dx = (x_max - x_min) / (num_points - 1)
    dy = (y_max - y_min) / (num_points - 1)
    lag_extent = [-50 * dx, 50 * dx, -50 * dy, 50 * dy]

    # Plot autocorrelation
    plt.figure(figsize=(6, 5))
    plt.imshow(autocorr_central, cmap='RdBu', extent=lag_extent, origin='lower')
    plt.title(f'Autocorrelation - Module {m}')
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

# --- Display All Plots ---
plt.show()