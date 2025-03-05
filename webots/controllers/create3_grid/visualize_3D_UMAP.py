import sys
import os
from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from umap import UMAP
import torch  # For potential tensor conversion

# Set project root to the current working directory (Alex-Dev-Updated)
project_root = Path(os.getcwd())
sys.path.append(str(project_root))

# --- Configuration ---
CONTROLLER_PATH_PREFIX = "webots/controllers/"
CONTROLLER_NAME = "create3_grid_v3"
WORLD_NAME = "A_iCreateBotEasy_10"

HMAP_DIR = os.path.join(CONTROLLER_PATH_PREFIX, CONTROLLER_NAME, "pkl", WORLD_NAME, "hmaps")
NETWORKS_DIR = os.path.join(CONTROLLER_PATH_PREFIX, CONTROLLER_NAME, "pkl", WORLD_NAME, "networks")

# --- Load Data ---
# Load grid cell layer (gcn) from saved file
with open(os.path.join(NETWORKS_DIR, "gcn.pkl"), "rb") as f:
    gcn = pickle.load(f)

# Load grid cell activations (hmap_gcn)
with open(os.path.join(HMAP_DIR, "hmap_gcn.pkl"), "rb") as f:
    hmap_gcn = pickle.load(f)

# Load robot position history (hmap_loc)
with open(os.path.join(HMAP_DIR, "hmap_loc.pkl"), "rb") as f:
    hmap_loc = pickle.load(f)

# --- Preprocess Data ---
# Convert to NumPy arrays if they are PyTorch tensors or lists
if isinstance(hmap_gcn, torch.Tensor):
    hmap_gcn = hmap_gcn.cpu().numpy()
if isinstance(hmap_loc, torch.Tensor):
    hmap_loc = hmap_loc.cpu().numpy()
else:
    hmap_loc = np.array(hmap_loc)

# Ensure hmap_loc has the expected shape [num_steps, 3]
if hmap_loc.ndim == 1:
    hmap_loc = np.array(hmap_loc).reshape(-1, 3)

# --- Dimensionality Reduction with UMAP ---
reducer = UMAP(n_components=3, random_state=42)
embedding = reducer.fit_transform(hmap_gcn)

# --- Visualization Function ---
def plot_embedding(embedding, hmap_loc, start_step=0, end_step=None, show_path=True):
    """
    Plots the 3D UMAP projection of grid cell activity with an optional sub-section of the agent's path.

    Parameters:
    - embedding: 3D UMAP projection of grid cell activations [num_steps, 3].
    - hmap_loc: Robot position history [num_steps, 3].
    - start_step: Starting index of the steps to visualize (default: 0).
    - end_step: Ending index of the steps to visualize (default: last step).
    - show_path: Whether to show the agent's path (default: True).
    """
    num_steps = embedding.shape[0]

    # Clamp start_step and end_step to valid range
    start_step = max(0, min(start_step, num_steps - 1))
    if end_step is None:
        end_step = num_steps
    else:
        end_step = max(start_step, min(end_step, num_steps))

    # Slice the data for the selected steps
    embedding_subset = embedding[start_step:end_step]
    hmap_loc_subset = hmap_loc[start_step:end_step]

    # Create the 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot of the 3D embedding, colored by x-position
    scatter = ax.scatter(embedding_subset[:, 0], embedding_subset[:, 1], embedding_subset[:, 2],
                         c=hmap_loc_subset[:, 0], cmap='viridis', s=10)

    # Add colorbar
    plt.colorbar(scatter, ax=ax, label='X Position (m)', pad=0.1)

    # Plot the agent's path in red if show_path is True
    if show_path:
        ax.plot(embedding_subset[:, 0], embedding_subset[:, 1], embedding_subset[:, 2],
                'r-', linewidth=0.5)

    # Set labels and title
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_zlabel('Dimension 3')
    ax.set_title(f'3D UMAP Projection (Steps {start_step} to {end_step})')

    plt.show()

# Example usage: Visualize steps 1000 to 2000 with path
#plot_embedding(embedding, hmap_loc, start_step=1000, end_step=1500, show_path=True)
plot_embedding(embedding, hmap_loc, show_path=False)

# To visualize without the path:
plot_embedding(embedding, hmap_loc, start_step=1000, end_step=2000, show_path=True)

# To visualize the entire path:
# plot_embedding(embedding, hmap_loc, show_path=True)