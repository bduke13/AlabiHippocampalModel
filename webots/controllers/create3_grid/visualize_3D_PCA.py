import sys
from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA  # Replaced UMAP with PCA
import torch

# Set project root to the current working directory (Alex-Dev-Updated)
project_root = Path.cwd()
sys.path.append(str(project_root))

# --- Configuration using pathlib for better cross-platform compatibility ---
CONTROLLER_PATH_PREFIX = Path("webots/controllers")
CONTROLLER_NAME = "create3_grid_v3"
WORLD_NAME = "A_iCreateBotEasy_10"
HMAP_DIR = CONTROLLER_PATH_PREFIX / CONTROLLER_NAME / "pkl" / WORLD_NAME / "hmaps"
NETWORKS_DIR = CONTROLLER_PATH_PREFIX / CONTROLLER_NAME / "pkl" / WORLD_NAME / "networks"

# --- Load Data ---
# Load grid cell layer (gcn) from saved file
with open(NETWORKS_DIR / "gcn.pkl", "rb") as f:
    gcn = pickle.load(f)

# Load grid cell activations (hmap_gcn)
with open(HMAP_DIR / "hmap_gcn.pkl", "rb") as f:
    hmap_gcn = pickle.load(f)

# Load robot position history (hmap_loc)
with open(HMAP_DIR / "hmap_loc.pkl", "rb") as f:
    hmap_loc = pickle.load(f)

# --- Preprocess Data ---
# Convert to NumPy arrays if they are PyTorch tensors or lists
hmap_gcn_np = hmap_gcn.cpu().numpy() if isinstance(hmap_gcn, torch.Tensor) else np.asarray(hmap_gcn)
hmap_loc_np = hmap_loc.cpu().numpy() if isinstance(hmap_loc, torch.Tensor) else np.asarray(hmap_loc)

# Ensure hmap_loc has the expected shape [num_steps, 3]
if hmap_loc_np.ndim == 1:
    hmap_loc_np = hmap_loc_np.reshape(-1, 3)

# --- Dimensionality Reduction with PCA ---
pca = PCA(n_components=3, random_state=42)
embedding = pca.fit_transform(hmap_gcn_np)

# Calculate and display explained variance
explained_variance = pca.explained_variance_ratio_
print(f"Explained variance ratio: {explained_variance}")
print(f"Total explained variance: {sum(explained_variance):.4f}")

# --- Optimized Visualization Function ---
def plot_embedding(embedding, hmap_loc, start_step=0, end_step=None, show_path=True):
    """
    Plots the 3D PCA projection of grid cell activity with an optional sub-section of the agent's path.
    Parameters:
    - embedding: 3D PCA projection of grid cell activations [num_steps, 3].
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
    if show_path and embedding_subset.shape[0] > 1:
        ax.plot(embedding_subset[:, 0], embedding_subset[:, 1], embedding_subset[:, 2],
                'r-', linewidth=0.5)
    
    # Set labels and title
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_zlabel('PCA Component 3')
    
    step_info = f"Steps {start_step} to {end_step-1}" if end_step < num_steps else f"All {num_steps} Steps"
    ax.set_title(f'3D PCA Projection ({step_info})')
    
    plt.tight_layout()
    return fig, ax

# Visualize the entire dataset without showing path
fig, ax = plot_embedding(embedding, hmap_loc_np, show_path=False)
plt.show()

# Example usage (commented out):
# Visualize steps 1000 to 1500 with path
fig, ax = plot_embedding(embedding, hmap_loc_np, start_step=1000, end_step=1500, show_path=True)
plt.show()
# Visualize the entire path
# plot_embedding(embedding, hmap_loc_np, show_path=True)