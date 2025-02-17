import matplotlib.pyplot as plt
import torch
import numpy as np
from matplotlib import cm, rcParams
from typing import List, Optional

# Import utility functions from vis_utils.py
from vis_utils import load_hmaps, convert_xzy_hmaps, load_layer_pkl

def plot_rcn_activation(
    rcn,
    hmap_pcn: np.ndarray,
    hmap_x: np.ndarray,
    hmap_y: np.ndarray,
    scale_idx: int = 0,
):
    """
    Visualizes the reward cell activations across the environment.

    Parameters:
      rcn: RewardCellLayer object with attribute 'w_in_effective'
      hmap_pcn: 2D numpy array of place cell activations (num_steps, num_pc)
      hmap_x: 1D numpy array for x coordinates
      hmap_y: 1D numpy array for y coordinates
      scale_idx: Index of the place cell scale to visualize
    """

    # Ensure correct tensor shapes
    w_in_float32 = torch.tensor(rcn.w_in_effective, dtype=torch.float32).clone().detach().cpu()
    hmap_pcn_float32 = torch.tensor(hmap_pcn.T, dtype=torch.float32)  # Shape (num_pc, num_steps)

    print(f"[INFO] w_in_float32 shape: {w_in_float32.shape}")
    print(f"[INFO] hmap_pcn_float32 shape: {hmap_pcn_float32.shape}")

    # Ensure dimensions match
    if w_in_float32.shape[1] != hmap_pcn_float32.shape[0]:
        raise ValueError(
            f"Dimension mismatch: w_in_effective has {w_in_float32.shape[1]} place cells, "
            f"but hmap_pcn has {hmap_pcn_float32.shape[0]}."
        )

    # Compute reward function using dot product
    sum_activations = torch.sum(hmap_pcn_float32, dim=0)
    safe_denom = torch.where(sum_activations > 0, sum_activations, torch.ones_like(sum_activations))
    reward_function = torch.tensordot(w_in_float32, hmap_pcn_float32, dims=1) / safe_denom
    reward_function = torch.squeeze(reward_function)

    # Ensure dimensions match for plotting
    if reward_function.shape != hmap_x.shape:
        raise ValueError(
            f"Shape mismatch: reward_function has shape {reward_function.shape}, "
            f"but expected {hmap_x.shape}."
        )

    # Determine axis limits
    x_min, x_max = np.min(hmap_x), np.max(hmap_x)
    y_min, y_max = np.min(hmap_y), np.max(hmap_y)

    # Plot the reward function
    fig, ax = plt.subplots(dpi=150)
    ax.set_aspect("equal")

    cntr = ax.hexbin(
        hmap_x,
        hmap_y,
        reward_function.detach().cpu().numpy(),
        gridsize=100,
        cmap=cmap,
        alpha=0.6,
    )

    # Adjust the view using data-based limits
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_axis_off()
    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

    # Add a colorbar and title, then show the plot
    fig.colorbar(cntr)
    plt.title(f"RCN Reward Map for Scale {scale_idx}")
    plt.show()


if __name__ == "__main__":
    # Set up the colormap and default settings
    cmap = cm.get_cmap("plasma")
    rcParams.update({"font.size": 12})

    # Select which scale to visualize
    scale_idx = 2  # Change this to visualize a different scale

    # Load hmap data and the rcn layer using utility functions
    hmap_loc, hmap_pcn = load_hmaps(["hmap_loc", f"hmap_pcn_scale_{scale_idx}"])
    hmap_x, hmap_z, hmap_y = convert_xzy_hmaps(hmap_loc)
    rcn = load_layer_pkl(layer_name=f"rcn_scale_{scale_idx}")

    # Plot the reward map activation
    plot_rcn_activation(
        rcn=rcn,
        hmap_pcn=hmap_pcn,
        hmap_x=hmap_x,
        hmap_y=hmap_y,
        scale_idx=scale_idx,
    )
