# %%
import matplotlib.pyplot as plt
import torch
import numpy as np
from matplotlib import cm, rcParams
import matplotlib.gridspec as gridspec
from typing import List, Optional
import os
import sys
from pathlib import Path

# Get the project root directory
project_root = Path(__file__).resolve().parent.parent  # Adjust if needed
sys.path.append(str(project_root))

# Import utility functions from vis_utils.py
from vis_utils import (
    load_hmaps, 
    convert_xzy_hmaps, 
    load_layer_pkl,
    CONTROLLER_PATH_PREFIX,
    CONTROLLER_NAME,
    WORLD_NAME,
    OUTPUT_DIR
)

def load_rcn_data(scale_idx):
    """
    Load RCN and place cell data for a specific scale.
    
    Args:
        scale_idx: The scale index (0, 1, or 2)
        
    Returns:
        tuple: (rcn, hmap_loc, hmap_pcn) or (None, None, None) if loading fails
    """
    try:
        # Load the RCN layer
        rcn = load_layer_pkl(layer_name=f"rcn_scale_{scale_idx}")
        
        # Load the hmap data
        hmap_loc, hmap_pcn = load_hmaps(["hmap_loc", f"hmap_pcn_scale_{scale_idx}"])
        
        print(f"Successfully loaded RCN data for scale {scale_idx}")
        return rcn, hmap_loc, hmap_pcn
    except Exception as e:
        print(f"Error loading data for scale {scale_idx}: {e}")
        return None, None, None

def compute_reward_function(rcn, hmap_pcn):
    """
    Compute the reward function across the environment.
    
    Args:
        rcn: RewardCellLayer object with attribute 'w_in_effective'
        hmap_pcn: Place cell activations (time steps × num_place_cells)
        
    Returns:
        torch.Tensor: Reward function values
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
    
    return reward_function

def plot_rcn_activation_single(
    rcn,
    hmap_pcn,
    hmap_x,
    hmap_y,
    scale_idx,
    ax=None,
    cmap_name="plasma",
    show_colorbar=True,
):
    """
    Visualizes the reward cell activations across the environment.

    Parameters:
      rcn: RewardCellLayer object with attribute 'w_in_effective'
      hmap_pcn: 2D numpy array of place cell activations (num_steps, num_pc)
      hmap_x: 1D numpy array for x coordinates
      hmap_y: 1D numpy array for y coordinates
      scale_idx: Index of the place cell scale to visualize
      ax: Matplotlib axis to plot on (optional)
      cmap_name: Name of the colormap to use
      show_colorbar: Whether to show the colorbar
    """
    # Create figure if not provided
    if ax is None:
        fig, ax = plt.subplots(dpi=150)
    
    # Compute the reward function
    reward_function = compute_reward_function(rcn, hmap_pcn)
    
    # Ensure dimensions match for plotting
    if reward_function.shape != hmap_x.shape:
        raise ValueError(
            f"Shape mismatch: reward_function has shape {reward_function.shape}, "
            f"but expected {hmap_x.shape}."
        )

    # Determine axis limits
    x_min, x_max = np.min(hmap_x), np.max(hmap_x)
    y_min, y_max = np.min(hmap_y), np.max(hmap_y)

    # Use the specified colormap
    cmap = cm.get_cmap(cmap_name)
    
    # Plot the reward function
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
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(f"Scale {scale_idx}")
    
    # Add colorbar if requested
    if show_colorbar:
        plt.colorbar(cntr, ax=ax)
    
    return ax, cntr

def plot_all_scales(
    scales=None,
    cmap_name="plasma",
    save_path=None,
    show_plot=True,
):
    """
    Plot reward cell activations for multiple scales side by side.
    
    Args:
        scales: List of scales to plot
        cmap_name: Name of the colormap to use
        save_path: Path to save the figure
        show_plot: Whether to show the plot
    """
    if scales is None:
        scales = [0, 1, 2]
    
    # Set up the figure
    fig = plt.figure(figsize=(15, 5))
    fig.suptitle("Reward Cell Network Activation Across Scales", fontsize=16)
    gs = gridspec.GridSpec(1, len(scales))
    
    # Set font size
    rcParams.update({"font.size": 12})
    
    # Load location data once (shared across scales)
    hmap_loc = load_hmaps(["hmap_loc"])
    if isinstance(hmap_loc, list):
        hmap_loc = hmap_loc[0]
    hmap_x, hmap_z, hmap_y = convert_xzy_hmaps(hmap_loc)
    
    # Create a list to store plots for colorbar normalization
    plots = []
    
    # Plot each scale
    for i, scale in enumerate(scales):
        # Load data for this scale
        rcn, _, hmap_pcn = load_rcn_data(scale)
        
        # Skip if data loading failed
        if rcn is None:
            continue
        
        # Create the subplot
        ax = plt.subplot(gs[0, i])
        
        # Plot the reward function
        _, cntr = plot_rcn_activation_single(
            rcn=rcn,
            hmap_pcn=hmap_pcn,
            hmap_x=hmap_x,
            hmap_y=hmap_y,
            scale_idx=scale,
            ax=ax,
            cmap_name=cmap_name,
            show_colorbar=True,
        )
        
        plots.append(cntr)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure if requested
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    
    # Show the plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

def save_rcn_plots(
    scales=None,
    cmap_name="plasma",
):
    """
    Generate and save RCN plots.
    
    Args:
        scales: List of scales to plot
        cmap_name: Name of the colormap to use
    """
    if scales is None:
        scales = [0, 1, 2]
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(OUTPUT_DIR, "rcn_plots")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the combined plot
    save_path = os.path.join(output_dir, "rcn_all_scales.png")
    plot_all_scales(
        scales=scales,
        cmap_name=cmap_name,
        save_path=save_path,
        show_plot=False,
    )
    
    # Also save individual plots for each scale
    for scale in scales:
        # Load data for this scale
        rcn, hmap_loc, hmap_pcn = load_rcn_data(scale)
        
        # Skip if data loading failed
        if rcn is None:
            continue
        
        # Convert coordinates
        hmap_x, hmap_z, hmap_y = convert_xzy_hmaps(hmap_loc)
        
        # Create figure
        fig, ax = plt.subplots(dpi=150, figsize=(8, 6))
        
        # Plot the reward function
        plot_rcn_activation_single(
            rcn=rcn,
            hmap_pcn=hmap_pcn,
            hmap_x=hmap_x,
            hmap_y=hmap_y,
            scale_idx=scale,
            ax=ax,
            cmap_name=cmap_name,
            show_colorbar=True,
        )
        
        # Set title
        ax.set_title(f"RCN Reward Map - Scale {scale}", fontsize=14)
        
        # Save the figure
        save_path = os.path.join(output_dir, f"rcn_scale_{scale}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved individual plot for scale {scale} to {save_path}")
        
        # Close the figure
        plt.close(fig)


# %%
if __name__ == "__main__":
    print("Starting multi-scale RCN visualization...")
    
    # Plot all scales together
    plot_all_scales(
        scales=[0, 1, 2],
        cmap_name="plasma",
        save_path=os.path.join(OUTPUT_DIR, "rcn_all_scales.png"),
        show_plot=True,
    )
    
    # Alternatively, save all plots (combined and individual)
    # save_rcn_plots(scales=[0, 1, 2], cmap_name="plasma")
    
    print("RCN visualization complete!")