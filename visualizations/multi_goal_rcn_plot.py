# %%
import matplotlib.pyplot as plt
import torch
import numpy as np
from matplotlib import cm, rcParams
import matplotlib.gridspec as gridspec
from typing import List, Optional, Dict
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
    load_multi_goal_rcn_data,
    load_goal_associations,
    get_available_multi_goal_combinations,
    load_multi_goal_hmaps,
    CONTROLLER_PATH_PREFIX,
    CONTROLLER_NAME,
    WORLD_NAME,
    OUTPUT_DIR
)

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

    print(f"[INFO] Computing reward for w_in shape: {w_in_float32.shape}, hmap_pcn shape: {hmap_pcn_float32.shape}")

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

def plot_multi_goal_reward_single(
    goal_name,
    scale_idx,
    rcn,
    hmap_x,
    hmap_y,
    hmap_pcn,
    ax=None,
    cmap_name="plasma",
    show_colorbar=True,
    show_goal_location=True,
    show_legend=False,  # New parameter to control legend display
):
    """
    Visualize the reward cell activations for a specific goal-scale combination.

    Parameters:
      goal_name: Name of the goal
      scale_idx: Index of the scale
      rcn: RewardCellLayer object for this goal-scale combination
      hmap_x: 1D numpy array for x coordinates
      hmap_y: 1D numpy array for y coordinates  
      hmap_pcn: 2D numpy array of place cell activations
      ax: Matplotlib axis to plot on (optional)
      cmap_name: Name of the colormap to use
      show_colorbar: Whether to show the colorbar
      show_goal_location: Whether to mark the goal location
      show_legend: Whether to show the legend
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

    # Use the specified colormap
    cmap = cm.get_cmap(cmap_name)
    
    # Plot the reward function
    cntr = ax.hexbin(
        hmap_x,
        hmap_y,
        reward_function.detach().cpu().numpy(),
        gridsize=100,
        cmap=cmap,
        alpha=0.7,
    )

    # Set axis limits
    x_min, x_max = np.min(hmap_x), np.max(hmap_x)
    y_min, y_max = np.min(hmap_y), np.max(hmap_y)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    # Set equal aspect ratio to prevent distortion
    ax.set_aspect('equal', adjustable='box')
    
    # Add goal location marker if requested
    if show_goal_location:
        try:
            associations = load_goal_associations()
            goal_info = None
            for goal in associations['goals']:
                if goal['name'] == goal_name:
                    goal_info = goal
                    break
            
            if goal_info:
                goal_x, goal_y = goal_info['location']
                ax.scatter(goal_x, goal_y, c='white', s=200, marker='*', 
                          edgecolors='black', linewidth=2, zorder=10)
                
                # Only show legend if explicitly requested
                if show_legend:
                    ax.legend([f'{goal_name.title()} Goal'], loc='upper right')
        except Exception as e:
            print(f"Warning: Could not load goal location for {goal_name}: {e}")
    
    # Labels and title
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(f"{goal_name.title()} - Scale {scale_idx}")
    
    # Add colorbar if requested
    if show_colorbar:
        plt.colorbar(cntr, ax=ax, label='Reward Value')
    
    return ax, cntr

def plot_all_multi_goal_rewards(
    goals=None,
    scales=None,
    cmap_name="plasma",
    save_path=None,
    show_plot=True,
    figsize=None,
):
    """
    Plot reward maps for all goal-scale combinations in a grid layout.
    
    Args:
        goals: List of goal names to plot (None for all available)
        scales: List of scale indices to plot (None for all available)
        cmap_name: Name of the colormap to use
        save_path: Path to save the figure
        show_plot: Whether to show the plot
        figsize: Figure size tuple (width, height)
    """
    # Discover available combinations if not specified
    available_goals, available_scales, combinations = get_available_multi_goal_combinations()
    
    if not available_goals:
        raise ValueError("No multi-goal reward data found")
    
    # Use available data if not specified
    if goals is None:
        goals = available_goals
    if scales is None:
        scales = available_scales
    
    # Validate requested goals and scales exist
    for goal in goals:
        if goal not in available_goals:
            raise ValueError(f"Goal '{goal}' not found in available data: {available_goals}")
    for scale in scales:
        if scale not in available_scales:
            raise ValueError(f"Scale {scale} not found in available data: {available_scales}")
    
    # Set up figure size - more compact
    if figsize is None:
        figsize = (3.5 * len(scales), 3.5 * len(goals))
    
    # Create the figure
    fig = plt.figure(figsize=figsize)
    
    # Create grid: goals as rows, scales as columns
    # Minimal spacing to remove whitespace
    gs = gridspec.GridSpec(len(goals), len(scales), 
                          hspace=0.3, wspace=0.1,
                          left=0.08, right=0.98, 
                          bottom=0.08, top=0.92)
    
    # Set font size
    rcParams.update({"font.size": 9})
    
    # Load shared location data
    hmap_loc = load_hmaps(["hmap_loc"])
    if isinstance(hmap_loc, list):
        hmap_loc = hmap_loc[0]
    hmap_x, hmap_z, hmap_y = convert_xzy_hmaps(hmap_loc)
    
    # Track successful plots
    successful_plots = 0
    total_expected = len(goals) * len(scales)
    
    # Plot each combination
    for goal_idx, goal_name in enumerate(goals):
        for scale_idx, scale in enumerate(scales):
            try:
                # Load RCN and hmap data for this combination
                rcn = load_multi_goal_rcn_data(goal_name=goal_name, scale_idx=scale)
                _, hmap_pcn = load_multi_goal_hmaps(goal_name=goal_name, scale_idx=scale)
                
                # Create subplot
                ax = plt.subplot(gs[goal_idx, scale_idx])
                
                # Set equal aspect ratio to prevent distortion
                ax.set_aspect('equal')
                
                # Plot the reward map
                plot_multi_goal_reward_single(
                    goal_name=goal_name,
                    scale_idx=scale,
                    rcn=rcn,
                    hmap_x=hmap_x,
                    hmap_y=hmap_y,
                    hmap_pcn=hmap_pcn,
                    ax=ax,
                    cmap_name=cmap_name,
                    show_colorbar=False,  # Skip individual colorbars for cleaner look
                    show_goal_location=True,
                    show_legend=False,  # Remove legends for cleaner look
                )
                
                successful_plots += 1
                
            except Exception as e:
                print(f"Warning: Failed to plot {goal_name} scale {scale}: {e}")
                # Create empty subplot with error message
                ax = plt.subplot(gs[goal_idx, scale_idx])
                ax.set_aspect('equal')
                ax.text(0.5, 0.5, f"Error loading\n{goal_name}\nScale {scale}", 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_title(f"{goal_name.title()} - Scale {scale}")
    """
    # Add column labels (scales) - positioned to fit in the top margin
    for scale_idx, scale in enumerate(scales):
        ax = plt.subplot(gs[0, scale_idx])
        ax.text(0.5, 1.03, f'Scale {scale}', ha='center', va='bottom', 
               transform=ax.transAxes, fontsize=11, fontweight='bold')
    
    # Add row labels (goals) - positioned to fit in the left margin
    for goal_idx, goal_name in enumerate(goals):
        ax = plt.subplot(gs[goal_idx, 0])
        ax.text(-0.08, 0.5, f'{goal_name.title()}', ha='right', va='center', 
               transform=ax.transAxes, fontsize=11, fontweight='bold', rotation=90)
    """
    print(f"Successfully plotted {successful_plots}/{total_expected} combinations")
    
    # Save the figure if requested
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
        print(f"Figure saved to {save_path}")
    
    # Show the plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

def plot_single_goal_all_scales(
    goal_name,
    scales=None,
    cmap_name="plasma",
    save_path=None,
    show_plot=True,
):
    """
    Plot all scales for a single goal.
    
    Args:
        goal_name: Name of the goal to plot
        scales: List of scale indices (None for all available)
        cmap_name: Colormap name
        save_path: Path to save figure
        show_plot: Whether to show the plot
    """
    # Get available data
    available_goals, available_scales, combinations = get_available_multi_goal_combinations()
    
    if goal_name not in available_goals:
        raise ValueError(f"Goal '{goal_name}' not found in available data: {available_goals}")
    
    if scales is None:
        scales = combinations[goal_name]
    
    # Create figure - more compact
    fig = plt.figure(figsize=(4 * len(scales), 4))
    
    # Load shared location data
    hmap_loc = load_hmaps(["hmap_loc"])
    if isinstance(hmap_loc, list):
        hmap_loc = hmap_loc[0]
    hmap_x, hmap_z, hmap_y = convert_xzy_hmaps(hmap_loc)
    
    # Plot each scale
    for i, scale in enumerate(scales):
        try:
            # Load data
            rcn = load_multi_goal_rcn_data(goal_name=goal_name, scale_idx=scale)
            _, hmap_pcn = load_multi_goal_hmaps(goal_name=goal_name, scale_idx=scale)
            
            # Create subplot
            ax = plt.subplot(1, len(scales), i + 1)
            
            # Set equal aspect ratio to prevent distortion
            ax.set_aspect('equal')
            
            # Plot
            plot_multi_goal_reward_single(
                goal_name=goal_name,
                scale_idx=scale,
                rcn=rcn,
                hmap_x=hmap_x,
                hmap_y=hmap_y,
                hmap_pcn=hmap_pcn,
                ax=ax,
                cmap_name=cmap_name,
                show_colorbar=True,
                show_goal_location=True,
                show_legend=False,  # Remove legend for cleaner look
            )
            
        except Exception as e:
            print(f"Warning: Failed to plot {goal_name} scale {scale}: {e}")
    
    # Use tight layout with minimal padding
    plt.tight_layout(pad=0.5)
    
    # Save if requested
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
        print(f"Figure saved to {save_path}")
    
    # Show if requested
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

def save_all_multi_goal_plots(output_dir=None):
    """
    Generate and save all multi-goal reward plots.
    """
    if output_dir is None:
        output_dir = os.path.join(OUTPUT_DIR, "multi_goal_rcn_plots")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get available data
    goals, scales, combinations = get_available_multi_goal_combinations()
    
    if not goals:
        print("No multi-goal data found to plot")
        return
    
    print(f"Generating plots for {len(goals)} goals and {len(scales)} scales")
    
    # Save the main grid plot
    main_save_path = os.path.join(output_dir, "all_goals_all_scales.png")
    plot_all_multi_goal_rewards(
        goals=goals,
        scales=scales,
        save_path=main_save_path,
        show_plot=False,
    )
    
    # Save individual goal plots
    for goal in goals:
        goal_save_path = os.path.join(output_dir, f"{goal}_all_scales.png")
        plot_single_goal_all_scales(
            goal_name=goal,
            scales=combinations[goal],
            save_path=goal_save_path,
            show_plot=False,
        )
    
    print(f"All plots saved to {output_dir}")

# %%
if __name__ == "__main__":
    print("Starting multi-goal RCN visualization...")
    
    try:
        # Check what data is available
        goals, scales, combinations = get_available_multi_goal_combinations()
        print(f"Available combinations: {combinations}")
        
        if not goals:
            print("No multi-goal reward data found. Make sure you've run LEARN_LOCATIONS mode first.")
        else:
            # Plot all combinations in a grid
            plot_all_multi_goal_rewards(
                goals=goals,
                scales=scales,
                cmap_name="plasma",
                save_path=os.path.join(OUTPUT_DIR, "multi_goal_all_rewards.png"),
                show_plot=True,
            )
            
            # Optionally save all individual plots too
            # save_all_multi_goal_plots()
            
    except Exception as e:
        print(f"Error during visualization: {e}")
        import traceback
        traceback.print_exc()
    
    print("Multi-goal RCN visualization complete!")