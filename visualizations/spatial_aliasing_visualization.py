# %%
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
import matplotlib.gridspec as gridspec
import pandas as pd
import seaborn as sns

# Get the project root directory
project_root = Path(__file__).resolve().parent.parent  # Adjust if needed
sys.path.append(str(project_root))

# Import from vis_utils
from vis_utils import (
    OUTPUT_DIR
)

# ===================== CONFIGURABLE PARAMETERS =====================
# Visualization parameters
SCALES_TO_VISUALIZE = [0, 1, 2]  # Scales to visualize (0=small, 1=medium, 2=large)
SCALE_NAMES = ["Small", "Medium", "Large"]  # Names for the scales
CMAP = 'viridis'  # Colormap for aliasing heatmaps

# Colormap ranges
VMIN = None  # Set to a value to standardize colormap min (e.g., 0.0)
VMAX = None  # Set to a value to standardize colormap max (e.g., 0.1)
USE_GLOBAL_COLORMAP = True  # Whether to use the same color range for all plots

# Output directory
OUTPUT_DIR_ALIASING = os.path.join(OUTPUT_DIR, "spatial_aliasing")
# =====================================================================

def plot_spatial_aliasing_heatmap(
    coords,
    sai_values,
    ax=None,
    title=None,
    cmap=CMAP,
    vmin=None,
    vmax=None,
    scatter_size=20,
    show_colorbar=True
):
    """
    Plot a spatial aliasing heatmap.
    
    Args:
        coords: Array of coordinates [(x1, y1), (x2, y2), ...].
        sai_values: Array of SAI values.
        ax: Matplotlib axis to plot on (optional).
        title: Title for the plot.
        cmap: Colormap to use.
        vmin: Minimum value for colormap.
        vmax: Maximum value for colormap.
        scatter_size: Size of scatter points.
        show_colorbar: Whether to show the colorbar.
        
    Returns:
        sc: Scatter plot object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create scatter plot
    sc = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=sai_values,
        cmap=cmap,
        s=scatter_size,
        alpha=0.7,
        vmin=vmin,
        vmax=vmax
    )
    
    # Add colorbar if requested
    if show_colorbar:
        plt.colorbar(sc, ax=ax, label="Spatial Aliasing Index")
    
    # Set labels and title
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    if title:
        ax.set_title(title)
    
    # Set aspect ratio to equal
    ax.set_aspect('equal')
    
    return sc

def create_comparison_plot(save_path=None, show_plot=True):
    """
    Create a figure comparing spatial aliasing across scales.
    
    Args:
        save_path: Path to save the figure.
        show_plot: Whether to show the plot.
        
    Returns:
        fig: The figure object.
    """
    # Check if results exist
    if not os.path.exists(OUTPUT_DIR_ALIASING):
        print(f"Error: {OUTPUT_DIR_ALIASING} not found. Run spatial_aliasing_analysis.py first.")
        return None
    
    # Load MSAI results
    msai_path = os.path.join(OUTPUT_DIR_ALIASING, "msai_results.csv")
    if not os.path.exists(msai_path):
        print(f"Error: {msai_path} not found. Run spatial_aliasing_analysis.py first.")
        return None
    
    results_df = pd.read_csv(msai_path)
    available_scales = results_df["scale_idx"].values
    
    # Determine which scales to plot
    scales_to_plot = [s for s in SCALES_TO_VISUALIZE if s in available_scales]
    
    if not scales_to_plot:
        print("No scales available to plot.")
        return None
    
    # Determine global colormap range if requested
    if USE_GLOBAL_COLORMAP:
        all_sai_values = []
        for scale_idx in scales_to_plot:
            sai_path = os.path.join(OUTPUT_DIR_ALIASING, f"scale_{scale_idx}", "sai_values.npy")
            if os.path.exists(sai_path):
                sai_values = np.load(sai_path)
                all_sai_values.extend(sai_values)
        
        if all_sai_values:
            global_vmin = VMIN if VMIN is not None else np.min(all_sai_values)
            global_vmax = VMAX if VMAX is not None else np.max(all_sai_values)
        else:
            global_vmin, global_vmax = VMIN, VMAX
    else:
        global_vmin, global_vmax = VMIN, VMAX
    
    # Create figure
    fig = plt.figure(figsize=(5 * len(scales_to_plot), 6))
    gs = gridspec.GridSpec(1, len(scales_to_plot))
    
    # Plot each scale
    scatter_plots = []
    
    for i, scale_idx in enumerate(scales_to_plot):
        # Create subplot
        ax = plt.subplot(gs[0, i])
        
        # Load data
        coords_path = os.path.join(OUTPUT_DIR_ALIASING, f"scale_{scale_idx}", "coords.npy")
        sai_path = os.path.join(OUTPUT_DIR_ALIASING, f"scale_{scale_idx}", "sai_values.npy")
        
        if not os.path.exists(coords_path) or not os.path.exists(sai_path):
            ax.text(0.5, 0.5, f"No data for Scale {scale_idx}", 
                    ha='center', va='center', transform=ax.transAxes)
            continue
        
        coords = np.load(coords_path)
        sai_values = np.load(sai_path)
        
        # Get scale info
        scale_row = results_df[results_df["scale_idx"] == scale_idx].iloc[0]
        scale_name = scale_row["scale_name"]
        msai = scale_row["msai"]
        
        # Plot heatmap
        sc = plot_spatial_aliasing_heatmap(
            coords=coords,
            sai_values=sai_values,
            ax=ax,
            title=f"{scale_name} (MSAI: {msai:.4f})",
            cmap=CMAP,
            vmin=global_vmin,
            vmax=global_vmax,
            show_colorbar=False  # We'll add a single colorbar for all plots
        )
        
        scatter_plots.append(sc)
    
    # Add a single colorbar for all plots
    if scatter_plots:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = plt.colorbar(scatter_plots[0], cax=cbar_ax)
        cbar.set_label("Spatial Aliasing Index (SAI)")
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    
    # Add figure title
    plt.suptitle("Spatial Aliasing Comparison Across Scales", fontsize=16, y=0.98)
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return fig

def create_msai_bar_plot(save_path=None, show_plot=True):
    """
    Create a bar plot of MSAI values across scales.
    
    Args:
        save_path: Path to save the figure.
        show_plot: Whether to show the plot.
        
    Returns:
        fig: The figure object.
    """
    # Load MSAI results
    msai_path = os.path.join(OUTPUT_DIR_ALIASING, "msai_results.csv")
    if not os.path.exists(msai_path):
        print(f"Error: {msai_path} not found. Run spatial_aliasing_analysis.py first.")
        return None
    
    results_df = pd.read_csv(msai_path)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Create bar plot
    sns.barplot(
        data=results_df,
        x="scale_name",
        y="msai",
        ax=ax
    )
    
    # Add labels and title
    ax.set_xlabel("Scale")
    ax.set_ylabel("Mean Spatial Aliasing Index (MSAI)")
    ax.set_title("Comparison of Mean Spatial Aliasing Index Across Scales")
    
    # Add grid lines
    ax.grid(axis='y', alpha=0.3)
    
    # Add text labels
    for i, msai in enumerate(results_df["msai"]):
        ax.text(i, msai + 0.001, f"{msai:.4f}", ha='center', va='bottom')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return fig

def create_all_visualizations():
    """
    Create all visualizations.
    """
    os.makedirs(OUTPUT_DIR_ALIASING, exist_ok=True)
    
    # Create comparison plot
    print("Creating spatial aliasing comparison plot...")
    comparison_path = os.path.join(OUTPUT_DIR_ALIASING, "spatial_aliasing_comparison.png")
    create_comparison_plot(save_path=comparison_path, show_plot=False)
    
    # Create MSAI bar plot
    print("Creating MSAI bar plot...")
    bar_path = os.path.join(OUTPUT_DIR_ALIASING, "msai_bar_plot.png")
    create_msai_bar_plot(save_path=bar_path, show_plot=False)
    
    print("Visualizations created!")

# Execute when run as a script
if __name__ == "__main__":
    print("Starting Spatial Aliasing Visualization...")
    create_all_visualizations()
    print("Visualization complete!")
