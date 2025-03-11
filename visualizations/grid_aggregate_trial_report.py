#!/usr/bin/env python3
"""
enhanced_trial_report.py

This script scans trial data directories and generates an HTML report with multiple visualizations:
1. Overlayed place cell activations
2. Top five most active place cells
3. First grid cell activation pattern
4. Average grid cell activation pattern

The script generates these visualizations for each trial and compiles them into an HTML report.
"""

import os
import sys
import json
import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# ===== Configurable Parameters =====
# Set these to control the visualization and file paths
GRID_SIZE = 120  # Resolution for all plots (higher = more detailed but slower)
PLOT_DPI = 300   # DPI for saved figures
FIGURE_SIZE = 8  # Size of square figures in inches

# ===== Configurable Paths =====
# Set this to the root folder of your project
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Trial data location
TRIAL_DATA_ROOT = PROJECT_ROOT / "webots" / "controllers" / "create3_grid_v1" / "trial_data"
# Output directory for reports
OUTPUT_DIR = PROJECT_ROOT / "webots" / "controllers" / "create3_grid_v1" / "3_grid_trial_report"

sys.path.append(str(PROJECT_ROOT))

# -------------
# Helper functions
# -------------

def load_pickle(file_path):
    """Load and return the pickle data from the given file path."""
    with open(file_path, "rb") as f:
        return pickle.load(f)

def load_trial_hmaps(trial_dir, include_grid=True):
    """
    Given a trial directory, load the hmap pickle files from the "hmaps" subfolder.
    
    Args:
        trial_dir: Path to the trial directory
        include_grid: Whether to load grid cell data too
        
    Returns:
        Tuple containing arrays: hmap_loc, hmap_pcn, hmap_gcn (if include_grid=True)
    """
    hmap_dir = os.path.join(trial_dir, "hmaps")
    hmap_loc_path = os.path.join(hmap_dir, "hmap_loc.pkl")
    hmap_pcn_path = os.path.join(hmap_dir, "hmap_pcn.pkl")
    
    if not os.path.exists(hmap_loc_path) or not os.path.exists(hmap_pcn_path):
        raise FileNotFoundError(f"Missing hmap files in {hmap_dir}")
    
    hmap_loc = np.array(load_pickle(hmap_loc_path))
    hmap_pcn = np.array(load_pickle(hmap_pcn_path))
    
    # As per legacy usage, remove the first element of hmap_loc
    hmap_loc = hmap_loc[1:] if len(hmap_loc) > 1 else hmap_loc
    
    if include_grid:
        hmap_gcn_path = os.path.join(hmap_dir, "hmap_gcn.pkl")
        if not os.path.exists(hmap_gcn_path):
            raise FileNotFoundError(f"Missing grid cell data: {hmap_gcn_path}")
        hmap_gcn = np.array(load_pickle(hmap_gcn_path))
        return hmap_loc, hmap_pcn, hmap_gcn
    
    return hmap_loc, hmap_pcn

def convert_xzy_hmaps(hmap_loc: np.ndarray):
    """
    Convert hmap location array to separate components.
    New model: getSFVec3f() returns [x, y, z] (with y vertical).
    For horizontal plotting we use x and z.
    Returns: hmap_x, hmap_z, hmap_y.
    """
    hmap_x = hmap_loc[:, 0]
    hmap_z = hmap_loc[:, 2]
    hmap_y = hmap_loc[:, 1]
    return hmap_x, hmap_z, hmap_y

def generate_random_colors(num_colors):
    """Generate an array of random vibrant colors for num_colors cells."""
    colors = np.zeros((num_colors, 3))
    for i in range(num_colors):
        while True:
            color = np.random.random(3)
            # Ensure at least one channel is vibrant (>0.8)
            color[np.random.randint(3)] = np.random.uniform(0.8, 1.0)
            if np.sum(color) > 1.2:
                colors[i] = color
                break
    return colors

def plot_overlayed_cells(
    hmap_pcn: np.ndarray,
    hmap_x: np.ndarray,
    hmap_z: np.ndarray,
    gridsize: int = GRID_SIZE,
    num_cells_to_sample: int = None,
    save_path: str = None
):
    """
    Generates an overlay image of place cell activations.
    This function is adapted from pcn_overlayed.py.
    Returns the save_path.
    """
    # Sum activations per cell over time.
    total_activation_per_cell = np.sum(hmap_pcn, axis=0)
    # Find indices with non-zero activation.
    nonzero_indices = np.where(total_activation_per_cell > 0)[0]
    if num_cells_to_sample is None:
        num_cells = len(nonzero_indices)
    else:
        num_cells = min(num_cells_to_sample, len(nonzero_indices))
    print(f"Plotting {num_cells} cells out of {len(nonzero_indices)} active cells.")
    
    # Randomly choose cells.
    selected_indices = np.random.choice(nonzero_indices, size=num_cells, replace=False)
    
    # Determine grid boundaries using x and z.
    xmin, xmax = np.min(hmap_x), np.max(hmap_x)
    zmin, zmax = np.min(hmap_z), np.max(hmap_z)
    xedges = np.linspace(xmin, xmax, gridsize + 1)
    zedges = np.linspace(zmin, zmax, gridsize + 1)
    
    total_activations_per_bin = np.zeros((gridsize, gridsize, num_cells))
    counts_per_bin = np.zeros((gridsize, gridsize, num_cells))
    
    for idx, cell in enumerate(selected_indices):
        activations = hmap_pcn[:, cell]
        mask = activations > 0
        if not np.any(mask):
            continue
        x_vals = hmap_x[mask]
        z_vals = hmap_z[mask]
        a_vals = activations[mask]
        ix = np.digitize(x_vals, xedges) - 1
        iz = np.digitize(z_vals, zedges) - 1
        ix = np.clip(ix, 0, gridsize - 1)
        iz = np.clip(iz, 0, gridsize - 1)
        for i_val, j_val, act in zip(ix, iz, a_vals):
            total_activations_per_bin[i_val, j_val, idx] += act
            counts_per_bin[i_val, j_val, idx] += 1
    
    mean_activation_per_bin = np.zeros_like(total_activations_per_bin)
    nonzero_counts = counts_per_bin > 0
    mean_activation_per_bin[nonzero_counts] = total_activations_per_bin[nonzero_counts] / counts_per_bin[nonzero_counts]
    
    max_mean_activation_per_bin = np.max(mean_activation_per_bin, axis=2)
    cell_with_max_activation = np.argmax(mean_activation_per_bin, axis=2)
    
    max_activation = np.max(max_mean_activation_per_bin)
    max_activation = max_activation if max_activation > 0 else 1  # Avoid division by zero
    normalized_activation = max_mean_activation_per_bin / max_activation
    
    colors_rgb = generate_random_colors(num_cells)
    
    image = np.zeros((gridsize, gridsize, 3))
    for i in range(gridsize):
        for j in range(gridsize):
            activation_level = normalized_activation[i, j]
            if activation_level > 0:
                idx = cell_with_max_activation[i, j]
                adjusted_color = activation_level * np.array(colors_rgb[idx])
                image[i, j, :] = adjusted_color
    
    # Transpose image so that the first axis corresponds to the vertical axis.
    image = np.transpose(image, (1, 0, 2))
    
    plt.figure(figsize=(FIGURE_SIZE, FIGURE_SIZE))
    extent = [xmin, xmax, zmin, zmax]
    plt.imshow(image, extent=extent, origin="lower")
    plt.xlabel("X Coordinate")
    plt.ylabel("Z Coordinate")
    plt.title("Overlay of Place Cell Activations")
    
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
        print(f"Overlay image saved to: {save_path}")
    plt.close()
    return save_path

def plot_top_place_cells(
    hmap_pcn: np.ndarray,
    hmap_x: np.ndarray,
    hmap_z: np.ndarray,
    gridsize: int = GRID_SIZE,
    num_top_cells: int = 5,
    save_path: str = None
):
    """
    Generates an overlay image of the top N most active place cells.
    Uses the same format as plot_overlayed_cells but limited to the most active cells.
    
    Args:
        hmap_pcn: Place cell activation history map
        hmap_x: X coordinates
        hmap_z: Z coordinates
        gridsize: Resolution of the plot
        num_top_cells: Number of top cells to include
        save_path: Path to save the visualization
        
    Returns:
        The save path of the generated image
    """
    # Sum activations per cell over time to find the most active cells
    total_activation_per_cell = np.sum(hmap_pcn, axis=0)
    
    # Get the indices of the top N most active cells
    top_cell_indices = np.argsort(total_activation_per_cell)[-num_top_cells:]
    
    print(f"Plotting top {num_top_cells} most active cells")
    
    # Determine grid boundaries using x and z
    xmin, xmax = np.min(hmap_x), np.max(hmap_x)
    zmin, zmax = np.min(hmap_z), np.max(hmap_z)
    xedges = np.linspace(xmin, xmax, gridsize + 1)
    zedges = np.linspace(zmin, zmax, gridsize + 1)
    
    total_activations_per_bin = np.zeros((gridsize, gridsize, num_top_cells))
    counts_per_bin = np.zeros((gridsize, gridsize, num_top_cells))
    
    for idx, cell in enumerate(top_cell_indices):
        activations = hmap_pcn[:, cell]
        mask = activations > 0
        if not np.any(mask):
            continue
        x_vals = hmap_x[mask]
        z_vals = hmap_z[mask]
        a_vals = activations[mask]
        ix = np.digitize(x_vals, xedges) - 1
        iz = np.digitize(z_vals, zedges) - 1
        ix = np.clip(ix, 0, gridsize - 1)
        iz = np.clip(iz, 0, gridsize - 1)
        for i_val, j_val, act in zip(ix, iz, a_vals):
            total_activations_per_bin[i_val, j_val, idx] += act
            counts_per_bin[i_val, j_val, idx] += 1
    
    mean_activation_per_bin = np.zeros_like(total_activations_per_bin)
    nonzero_counts = counts_per_bin > 0
    mean_activation_per_bin[nonzero_counts] = total_activations_per_bin[nonzero_counts] / counts_per_bin[nonzero_counts]
    
    max_mean_activation_per_bin = np.max(mean_activation_per_bin, axis=2)
    cell_with_max_activation = np.argmax(mean_activation_per_bin, axis=2)
    
    max_activation = np.max(max_mean_activation_per_bin)
    max_activation = max_activation if max_activation > 0 else 1  # Avoid division by zero
    normalized_activation = max_mean_activation_per_bin / max_activation
    
    # Generate distinct colors for the top cells
    colors_rgb = generate_random_colors(num_top_cells)
    
    image = np.zeros((gridsize, gridsize, 3))
    for i in range(gridsize):
        for j in range(gridsize):
            activation_level = normalized_activation[i, j]
            if activation_level > 0:
                idx = cell_with_max_activation[i, j]
                adjusted_color = activation_level * np.array(colors_rgb[idx])
                image[i, j, :] = adjusted_color
    
    # Transpose image so that the first axis corresponds to the vertical axis
    image = np.transpose(image, (1, 0, 2))
    
    plt.figure(figsize=(FIGURE_SIZE, FIGURE_SIZE))
    extent = [xmin, xmax, zmin, zmax]
    plt.imshow(image, extent=extent, origin="lower")
    plt.xlabel("X Coordinate")
    plt.ylabel("Z Coordinate")
    plt.title(f"Top {num_top_cells} Most Active Place Cells")
    
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
        print(f"Top place cells image saved to: {save_path}")
    plt.close()
    return save_path

def plot_first_grid_cell(
    hmap_gcn: np.ndarray,
    hmap_x: np.ndarray,
    hmap_z: np.ndarray,
    gridsize: int = GRID_SIZE,
    save_path: str = None
):
    """
    Generates a visualization of the first grid cell's activation pattern.
    
    Args:
        hmap_gcn: Grid cell activation history map
        hmap_x: X coordinates
        hmap_z: Z coordinates
        gridsize: Resolution of the plot
        save_path: Path to save the visualization
        
    Returns:
        The save path of the generated image
    """
    # Get the first grid cell's activations
    cell_activations = hmap_gcn[:, 0]
    
    # Determine grid boundaries
    xmin, xmax = np.min(hmap_x), np.max(hmap_x)
    zmin, zmax = np.min(hmap_z), np.max(hmap_z)
    
    # Create the plot
    plt.figure(figsize=(FIGURE_SIZE, FIGURE_SIZE))
    
    # Create a hexbin plot
    hb = plt.hexbin(
        hmap_x, hmap_z, 
        C=cell_activations, 
        gridsize=gridsize // 2,  # Coarser grid for hexbin
        cmap='viridis',
        reduce_C_function=np.mean,
        edgecolor='none',
        extent=[xmin, xmax, zmin, zmax]
    )
    
    plt.colorbar(hb, label='Activation')
    plt.title("First Grid Cell Activation Pattern")
    plt.xlabel("X Coordinate")
    plt.ylabel("Z Coordinate")
    
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
        print(f"First grid cell image saved to: {save_path}")
    plt.close()
    return save_path

def plot_average_grid_activation(
    hmap_gcn: np.ndarray,
    hmap_x: np.ndarray,
    hmap_z: np.ndarray,
    gridsize: int = GRID_SIZE,
    save_path: str = None
):
    """
    Generates a visualization of the average grid cell activation pattern.
    
    Args:
        hmap_gcn: Grid cell activation history map
        hmap_x: X coordinates
        hmap_z: Z coordinates
        gridsize: Resolution of the plot
        save_path: Path to save the visualization
        
    Returns:
        The save path of the generated image
    """
    # Calculate the average activation across all grid cells
    average_activations = np.mean(hmap_gcn, axis=1)
    
    # Determine grid boundaries
    xmin, xmax = np.min(hmap_x), np.max(hmap_x)
    zmin, zmax = np.min(hmap_z), np.max(hmap_z)
    
    # Create the plot
    plt.figure(figsize=(FIGURE_SIZE, FIGURE_SIZE))
    
    # Create a hexbin plot
    hb = plt.hexbin(
        hmap_x, hmap_z, 
        C=average_activations, 
        gridsize=gridsize // 2,  # Coarser grid for hexbin
        cmap='plasma',
        reduce_C_function=np.mean,
        edgecolor='none',
        extent=[xmin, xmax, zmin, zmax]
    )
    
    plt.colorbar(hb, label='Average Activation')
    plt.title("Average Grid Cell Activation")
    plt.xlabel("X Coordinate")
    plt.ylabel("Z Coordinate")
    
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
        print(f"Average grid activation image saved to: {save_path}")
    plt.close()
    return save_path

def generate_html_report(entries):
    """
    Generate an HTML report displaying each trial's visualizations and key parameters.
    """
    html = []
    html.append("<html>")
    html.append("<head>")
    html.append("<title>Enhanced Trial Report</title>")
    html.append("<style>")
    html.append("body { font-family: Arial, sans-serif; margin: 20px; }")
    html.append("table { border-collapse: collapse; width: 100%; }")
    html.append("th, td { padding: 10px; text-align: center; border: 1px solid #ddd; }")
    html.append("th { background-color: #f2f2f2; }")
    html.append("h1 { color: #333; }")
    html.append("img { max-width: 100%; height: auto; }")
    html.append(".trial-row { background-color: #f9f9f9; }")
    html.append(".trial-row:nth-child(even) { background-color: #ffffff; }")
    html.append(".parameter { font-weight: bold; }")
    html.append("</style>")
    html.append("</head>")
    html.append("<body>")
    html.append("<h1>Enhanced Trial Report</h1>")
    html.append("<table>")
    html.append("<tr>")
    html.append("<th>Trial Parameters</th>")
    html.append("<th>Overlayed Place Cells</th>")
    html.append("<th>Top 5 Most Active Place Cells</th>")
    html.append("<th>First Grid Cell</th>")
    html.append("<th>Average Grid Activation</th>")
    html.append("</tr>")
    
    for entry in entries:
        html.append("<tr class='trial-row'>")
        
        # Trial parameters cell
        html.append("<td>")
        html.append(f"<div class='parameter'>Trial Name:</div> {entry['trial_name']}<br>")
        html.append(f"<div class='parameter'>Sigma_ang:</div> {entry['sigma_ang']}<br>")
        html.append(f"<div class='parameter'>Sigma_d:</div> {entry['sigma_d']}<br>")
        html.append(f"<div class='parameter'>Num Place Cells:</div> {entry.get('num_place_cells', 'N/A')}<br>")
        html.append(f"<div class='parameter'>Num Grid Cells:</div> {entry.get('num_grid_cells', 'N/A')}<br>")
        html.append(f"<div class='parameter'>Scale Multiplier:</div> {entry.get('scale_multiplier', 'N/A')}<br>")
        html.append(f"<div class='parameter'>Runtime:</div> {entry['max_runtime_hours']} hours<br>")
        html.append("</td>")
        
        # Visualization cells
        html.append(f"<td><img src='{entry['overlayed_image']}' alt='Overlayed Place Cells'></td>")
        html.append(f"<td><img src='{entry['top_place_cells_image']}' alt='Top Place Cells'></td>")
        html.append(f"<td><img src='{entry['first_grid_cell_image']}' alt='First Grid Cell'></td>")
        html.append(f"<td><img src='{entry['avg_grid_image']}' alt='Average Grid Activation'></td>")
        
        html.append("</tr>")
    
    html.append("</table>")
    html.append("</body></html>")
    return "\n".join(html)

def aggregate_trials(trial_data_root, output_dir):
    """
    Aggregate trial data from all subdirectories in trial_data_root.
    For each trial, generate multiple visualizations and collect trial parameters.
    Then produce an HTML report summarizing the trials.
    """
    trial_data_root = Path(trial_data_root)
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    trial_dirs = sorted([d for d in trial_data_root.iterdir() if d.is_dir()])
    
    report_entries = []
    
    for trial_dir in trial_dirs:
        try:
            print(f"\nProcessing trial: {trial_dir.name}")
            json_path = trial_dir / "trial_params.json"
            if not json_path.exists():
                print(f"Warning: {json_path} not found. Skipping {trial_dir}")
                continue
            with open(json_path, "r") as f:
                trial_params = json.load(f)
            
            try:
                hmap_loc, hmap_pcn, hmap_gcn = load_trial_hmaps(trial_dir, include_grid=True)
            except Exception as e:
                print(f"Error loading hmaps for {trial_dir}: {e}")
                continue
            
            # Trim arrays to the minimum length
            min_len = min(hmap_loc.shape[0], hmap_pcn.shape[0], hmap_gcn.shape[0])
            hmap_loc = hmap_loc[:min_len]
            hmap_pcn = hmap_pcn[:min_len]
            hmap_gcn = hmap_gcn[:min_len]
            
            # Convert coordinates
            hmap_x, hmap_z, _ = convert_xzy_hmaps(hmap_loc)
            
            # Generate image filenames
            trial_name = trial_params['trial_name']
            overlayed_filename = f"{trial_name}_overlayed.png"
            top_place_cells_filename = f"{trial_name}_top_place_cells.png"
            first_grid_cell_filename = f"{trial_name}_first_grid_cell.png"
            avg_grid_filename = f"{trial_name}_avg_grid.png"
            
            # Generate visualizations
            overlayed_path = output_dir / overlayed_filename
            plot_overlayed_cells(
                hmap_pcn=hmap_pcn,
                hmap_x=hmap_x,
                hmap_z=hmap_z,
                gridsize=GRID_SIZE,
                save_path=str(overlayed_path)
            )
            
            top_place_cells_path = output_dir / top_place_cells_filename
            plot_top_place_cells(
                hmap_pcn=hmap_pcn,
                hmap_x=hmap_x,
                hmap_z=hmap_z,
                gridsize=GRID_SIZE,
                num_top_cells=5,
                save_path=str(top_place_cells_path)
            )
            
            first_grid_cell_path = output_dir / first_grid_cell_filename
            plot_first_grid_cell(
                hmap_gcn=hmap_gcn,
                hmap_x=hmap_x,
                hmap_z=hmap_z,
                gridsize=GRID_SIZE,
                save_path=str(first_grid_cell_path)
            )
            
            avg_grid_path = output_dir / avg_grid_filename
            plot_average_grid_activation(
                hmap_gcn=hmap_gcn,
                hmap_x=hmap_x,
                hmap_z=hmap_z,
                gridsize=GRID_SIZE,
                save_path=str(avg_grid_path)
            )
            
            # Create report entry
            entry = {
                "trial_name": trial_params.get("trial_name", "Unknown"),
                "sigma_ang": trial_params.get("sigma_ang", "N/A"),
                "sigma_d": trial_params.get("sigma_d", "N/A"),
                "num_place_cells": trial_params.get("num_place_cells", "N/A"),
                "num_grid_cells": trial_params.get("num_grid_cells", "N/A"),
                "scale_multiplier": trial_params.get("scale_multiplier", "N/A"),
                "max_runtime_hours": trial_params.get("max_runtime_hours", "N/A"),
                
                # Image paths (using basenames for HTML)
                "overlayed_image": overlayed_filename,
                "top_place_cells_image": top_place_cells_filename,
                "first_grid_cell_image": first_grid_cell_filename,
                "avg_grid_image": avg_grid_filename,
                
                # Full parameter set for reference
                "other_params": trial_params
            }
            
            report_entries.append(entry)
            print(f"Successfully processed trial {entry['trial_name']}")
            
        except Exception as e:
            print(f"Error processing trial {trial_dir}: {e}")
            import traceback
            traceback.print_exc()
    
    # Generate HTML report
    html_output = generate_html_report(report_entries)
    report_path = output_dir / "enhanced_trial_report.html"
    with open(report_path, "w") as f:
        f.write(html_output)
    print(f"\nEnhanced HTML report generated at: {report_path}")

if __name__ == "__main__":
    # Define target directories using the configurable variables
    aggregate_trials(TRIAL_DATA_ROOT, OUTPUT_DIR)