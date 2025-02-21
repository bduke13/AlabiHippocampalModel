#!/usr/bin/env python3
"""
aggregate_trial_report.py

This script scans a given trial data directory (each subfolder represents a trial),
loads the hmap data (hmap_loc and hmap_pcn) and the trial parameters (from a JSON file),
trims the arrays so that their first dimensions match,
generates an overlay image for each trial (using an updated version of pcn_overlayed logic),
and produces an HTML report displaying each trial’s key parameters along with its overlay image.

The target directory (for both reading trial data and saving outputs) is configurable.
"""

import os
import sys
import json
import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ===== Configurable Paths =====
# Set this to the root folder of your project.
PROJECT_ROOT = Path(__file__).resolve().parent.parent  
# You can change the following paths if needed.
TRIAL_DATA_ROOT = PROJECT_ROOT / "webots" / "controllers" / "alex_controller" / "trial_data"
OUTPUT_DIR = PROJECT_ROOT / "webots" / "controllers" / "alex_controller" / "trial_report"

sys.path.append(str(PROJECT_ROOT))

# -------------
# Helper functions
# -------------

def load_pickle(file_path):
    """Load and return the pickle data from the given file path."""
    with open(file_path, "rb") as f:
        return pickle.load(f)

def load_trial_hmaps(trial_dir):
    """
    Given a trial directory, load the hmap_loc and hmap_pcn pickle files
    from the "hmaps" subfolder.
    """
    hmap_dir = os.path.join(trial_dir, "hmaps")
    hmap_loc_path = os.path.join(hmap_dir, "hmap_loc.pkl")
    hmap_pcn_path = os.path.join(hmap_dir, "hmap_pcn.pkl")
    if not os.path.exists(hmap_loc_path) or not os.path.exists(hmap_pcn_path):
        raise FileNotFoundError(f"Missing hmap files in {hmap_dir}")
    hmap_loc = np.array(load_pickle(hmap_loc_path))
    hmap_pcn = np.array(load_pickle(hmap_pcn_path))
    # As per legacy usage, remove the first element of hmap_loc
    hmap_loc = hmap_loc[1:]
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
    hmap_z: np.ndarray,  # Use z for the horizontal coordinate
    gridsize: int = 500,
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
    
    plt.figure(figsize=(8, 8))
    extent = [xmin, xmax, zmin, zmax]
    plt.imshow(image, extent=extent, origin="lower")
    plt.xlabel("X Coordinate")
    plt.ylabel("Z Coordinate")
    plt.title("Overlay of Place Cell Activations")
    
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=1000, bbox_inches="tight")
        print(f"Overlay image saved to: {save_path}")
    plt.close()
    return save_path

def generate_html_report(entries):
    """
    Generate a simple HTML report displaying each trial's overlay image and key parameters.
    """
    html = []
    html.append("<html>")
    html.append("<head><title>Aggregate Trial Report</title></head>")
    html.append("<body>")
    html.append("<h1>Aggregate Trial Report</h1>")
    html.append("<table border='1' cellspacing='0' cellpadding='5'>")
    html.append("<tr><th>Trial Name</th><th>Sigma_ang</th><th>Sigma_d</th><th>Max Runtime (hrs)</th><th>Overlay Image</th></tr>")
    for entry in entries:
        row = "<tr>"
        row += f"<td>{entry['trial_name']}</td>"
        row += f"<td>{entry['sigma_ang']}</td>"
        row += f"<td>{entry['sigma_d']}</td>"
        row += f"<td>{entry['max_runtime_hours']}</td>"
        # Assume the HTML file will be in the same folder as the images.
        row += f"<td><img src='{entry['image']}' alt='{entry['trial_name']}' width='300'></td>"
        row += "</tr>"
        html.append(row)
    html.append("</table>")
    html.append("</body></html>")
    return "\n".join(html)

def aggregate_trials(trial_data_root, output_dir):
    """
    Aggregate trial data from all subdirectories in trial_data_root.
    For each trial, generate an overlay image and collect trial parameters.
    Then produce an HTML report summarizing the trials.
    """
    trial_data_root = Path(trial_data_root)
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    trial_dirs = sorted([d for d in trial_data_root.iterdir() if d.is_dir()])
    
    report_entries = []
    
    for trial_dir in trial_dirs:
        try:
            json_path = trial_dir / "trial_params.json"
            if not json_path.exists():
                print(f"Warning: {json_path} not found. Skipping {trial_dir}")
                continue
            with open(json_path, "r") as f:
                trial_params = json.load(f)
            
            try:
                hmap_loc, hmap_pcn = load_trial_hmaps(trial_dir)
            except Exception as e:
                print(f"Error loading hmaps for {trial_dir}: {e}")
                continue
            
            # Trim both arrays to the minimum length.
            min_len = min(hmap_loc.shape[0], hmap_pcn.shape[0])
            hmap_loc = hmap_loc[:min_len]
            hmap_pcn = hmap_pcn[:min_len]
            
            # Use the new conversion: this returns (hmap_x, hmap_z, hmap_y)
            hmap_x, hmap_z, _ = convert_xzy_hmaps(hmap_loc)
            
            image_filename = f"{trial_params['trial_name']}_overlay.png"
            image_path = output_dir / image_filename
            plot_overlayed_cells(
                hmap_pcn=hmap_pcn,
                hmap_x=hmap_x,
                hmap_z=hmap_z,
                gridsize=80,
                num_cells_to_sample=None,
                save_path=str(image_path)
            )
            
            entry = {
                "trial_name": trial_params.get("trial_name", "Unknown"),
                "sigma_ang": trial_params.get("sigma_ang", "N/A"),
                "sigma_d": trial_params.get("sigma_d", "N/A"),
                "max_runtime_hours": trial_params.get("max_runtime_hours", "N/A"),
                # Use basename for the image path (assuming HTML and images are in the same folder)
                "image": os.path.basename(str(image_path)),
                "other_params": trial_params
            }
            report_entries.append(entry)
            print(f"Processed trial {entry['trial_name']}")
        except Exception as e:
            print(f"Error processing trial {trial_dir}: {e}")
    
    html_output = generate_html_report(report_entries)
    report_path = output_dir / "trial_report.html"
    with open(report_path, "w") as f:
        f.write(html_output)
    print(f"HTML report generated at: {report_path}")

if __name__ == "__main__":
    # Define target directories using the configurable variables.
    trial_data_root = os.path.join("webots", "controllers", "alex_controller", "trial_data")
    output_dir = os.path.join("webots", "controllers", "alex_controller", "trial_report")
    aggregate_trials(trial_data_root, output_dir)
