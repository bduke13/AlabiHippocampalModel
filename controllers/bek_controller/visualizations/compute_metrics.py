# compute_metrics.py

import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from math import pi

#############################################################
# ENVIRONMENT DICTIONARY
#############################################################
# For each environment, define:
#   - min_x, max_x, min_y, max_y: the bounding box
#   - grid_resolution: the resolution for the environment grid
#
# Extend this dictionary if you add more environments.
#############################################################

ENVIRONMENTS = {
    "10_10_Square": {
        "min_x": -5.0,
        "max_x":  5.0,
        "min_y": -5.0,
        "max_y":  5.0,
        "grid_resolution": 0.5
    },
    "10_10_Cross": {
        "min_x": -5.0,
        "max_x":  5.0,
        "min_y": -5.0,
        "max_y":  5.0,
        "grid_resolution": 0.5
    },
    "10_10_Warp": {
        "min_x": -5.0,
        "max_x":  5.0,
        "min_y": -5.0,
        "max_y":  5.0,
        "grid_resolution": 0.5
    },
    "15_15_Maze": {
        "min_x": -7.5,
        "max_x":  7.5,
        "min_y": -7.5,
        "max_y":  7.5,
        "grid_resolution": 0.5
    }
}

#############################################################
# METRIC COMPUTATION FUNCTIONS
#############################################################

def compute_environment_coverage(visitation_map, total_cells):
    """Compute fraction of environment grid cells visited."""
    visited_cells = len(visitation_map)
    coverage = visited_cells / total_cells if total_cells > 0 else 0
    return coverage

def compute_redundancy(visitation_map):
    """Total revisits = sum of visits - number of distinct visited cells."""
    total_visits = sum(visitation_map.values())
    unique_cells = len(visitation_map)
    redundancy = total_visits - unique_cells
    return redundancy

def compute_path_length(hmap_x, hmap_y):
    """Sums up stepwise Euclidean distance to get total path length."""
    path_length = np.sum(np.sqrt(np.diff(hmap_x)**2 + np.diff(hmap_y)**2))
    return path_length

def compute_exploration_efficiency(hmap_x, hmap_y, visitation_map, coverage):
    """Coverage per unit distance traveled."""
    total_distance = compute_path_length(hmap_x, hmap_y)
    efficiency = coverage / total_distance if total_distance > 0 else 0
    return efficiency

def compute_path_diversity(hmap_h):
    """
    Calculates path diversity as:
      1 - |mean(exp(i * heading))|
    Typically hmap_h might store either a single heading per timestep or
    head-direction cell activations. Adjust as needed.
    """
    # Convert headings to radians
    headings_rad = np.deg2rad(hmap_h)
    # Compute circular variance measure:
    #   path_diversity = 1 - magnitude of average heading vector
    path_diversity = 1 - np.abs(np.mean(np.exp(1j * headings_rad)))
    return path_diversity

#############################################################
# PLOTTING FUNCTIONS (OPTIONAL)
#############################################################

def plot_coverage(visitation_map, grid_resolution, mode_name):
    """Plot the visited cells in a scatter, color-coded by count."""
    cells = np.array(list(visitation_map.keys()))
    counts = np.array(list(visitation_map.values()))
    if len(cells) == 0:
        print("No cells visited; skipping coverage plot.")
        return

    counts_normalized = counts / np.max(counts) if np.max(counts) > 0 else counts

    plt.figure(figsize=(8, 6))
    plt.scatter(cells[:, 0] * grid_resolution, cells[:, 1] * grid_resolution,
                c=counts_normalized, cmap='viridis', marker='s', s=100)
    plt.colorbar(label='Normalized Visit Count')
    plt.title(f'Environment Coverage - {mode_name}')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.axis('equal')
    plt.show()

def plot_redundancy(visitation_map, mode_name):
    """Histogram of the visit counts to show how many times cells were revisited."""
    counts = np.array(list(visitation_map.values()))
    if len(counts) == 0:
        print("No visits to plot.")
        return
    plt.figure(figsize=(8, 6))
    plt.hist(counts, bins=range(1, int(np.max(counts))+2),
             align='left', rwidth=0.8)
    plt.title(f'Redundancy - {mode_name}')
    plt.xlabel('Visit Count per Cell')
    plt.ylabel('Number of Cells')
    plt.show()

def plot_efficiency(efficiency, mode_name):
    """Simple bar plot of exploration efficiency."""
    plt.figure(figsize=(8, 6))
    plt.bar([mode_name], [efficiency], color='orange')
    plt.title('Exploration Efficiency')
    plt.ylabel('Coverage per Unit Distance')
    plt.show()

def plot_path_diversity(hmap_h, mode_name):
    """Histogram of headings (0..360) to visualize distribution."""
    if len(hmap_h) == 0:
        print("No heading data to plot.")
        return
    headings_deg = hmap_h % 360  # ensure headings are in [0, 360)
    plt.figure(figsize=(8, 6))
    plt.hist(headings_deg, bins=36, range=(0, 360), density=True)
    plt.title(f'Path Diversity - {mode_name}')
    plt.xlabel('Heading (Degrees)')
    plt.ylabel('Frequency')
    plt.show()

#############################################################
# MAIN EXECUTION
#############################################################

def main():
    # 1. Ask user for the exploration mode (e.g. "Random Walk", "Curiosity", etc.)
    mode_name = input("Enter the exploration mode (e.g. 'Random Walk', 'Curiosity', 'Hybrid'): ").strip()
    if not mode_name:
        print("No exploration mode entered. Exiting.")
        return

    # 2. Ask user for environment selection
    print("\nAvailable environments:")
    for env_name in ENVIRONMENTS.keys():
        print(f"  - {env_name}")
    selected_env = input("Select environment name from the above: ").strip()
    if selected_env not in ENVIRONMENTS:
        print(f"Error: '{selected_env}' is not recognized. Exiting.")
        return

    env_params = ENVIRONMENTS[selected_env]
    min_x = env_params["min_x"]
    max_x = env_params["max_x"]
    min_y = env_params["min_y"]
    max_y = env_params["max_y"]
    grid_resolution = env_params["grid_resolution"]

    # Compute total cells from bounding box and resolution
    total_cells_x = int((max_x - min_x) / grid_resolution)
    total_cells_y = int((max_y - min_y) / grid_resolution)
    total_cells = total_cells_x * total_cells_y

    print(f"\nUsing environment '{selected_env}' with bounding box:")
    print(f"  X: {min_x} to {max_x}, Y: {min_y} to {max_y}")
    print(f"  Grid resolution: {grid_resolution}, Total cells: {total_cells}")

    # 3. Load data from pickled files
    # Adjust data_dir as needed
    current_dir = os.path.dirname(os.path.abspath(__file__))
    bek_controller_dir = os.path.abspath(os.path.join(current_dir, '..'))
    data_dir = bek_controller_dir  # or wherever your .pkl files are stored

    try:
        with open(os.path.join(data_dir, 'hmap_x.pkl'), 'rb') as f:
            hmap_x = np.array(pickle.load(f))
        with open(os.path.join(data_dir, 'hmap_y.pkl'), 'rb') as f:
            hmap_y = np.array(pickle.load(f))
        with open(os.path.join(data_dir, 'hmap_h.pkl'), 'rb') as f:
            hmap_h_array = np.array(pickle.load(f))

            # If you store the raw head direction cell activations (2D array), convert to headings
            if hmap_h_array.ndim == 2:
                n_hd = hmap_h_array.shape[1]
                hd_angles = np.linspace(0, 360, n_hd, endpoint=False)
                # Weighted sum of angles by activations
                headings = np.degrees(np.arctan2(
                    np.sum(np.sin(np.deg2rad(hd_angles)) * hmap_h_array, axis=1),
                    np.sum(np.cos(np.deg2rad(hd_angles)) * hmap_h_array, axis=1)
                )) % 360
                hmap_h = headings
            else:
                # If hmap_h_array is already a 1D array of headings
                hmap_h = hmap_h_array

        # Load the integer-based visitation map for metrics
        with open(os.path.join(data_dir, 'visitation_map_metrics.pkl'), 'rb') as f:
            visitation_map = pickle.load(f)
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return

    # 4. Compute metrics
    coverage = compute_environment_coverage(visitation_map, total_cells)
    redundancy = compute_redundancy(visitation_map)
    efficiency = compute_exploration_efficiency(hmap_x, hmap_y, visitation_map, coverage)
    diversity = compute_path_diversity(hmap_h)

    # 5. Print metrics
    print(f"\nMetrics for {mode_name} in {selected_env}:")
    print(f"Environment Coverage: {coverage:.2%}")
    print(f"Redundancy (total revisits): {redundancy}")
    print(f"Exploration Efficiency (coverage per unit distance): {efficiency:.5f}")
    print(f"Path Diversity: {diversity:.5f}\n")

    # 6. Generate visualizations
    plot_coverage(visitation_map, grid_resolution, mode_name)
    plot_redundancy(visitation_map, mode_name)
    plot_efficiency(efficiency, mode_name)
    plot_path_diversity(hmap_h, mode_name)

    # 7. Save metrics to a CSV
    metrics_file = os.path.join(current_dir, 'metrics.csv')
    metrics_data = {
        'Mode': mode_name,
        'Environment': selected_env,
        'Coverage': coverage,
        'Redundancy': redundancy,
        'Efficiency': efficiency,
        'Path_Diversity': diversity
    }

    # If the file doesn't exist, create it with a header
    if not os.path.isfile(metrics_file):
        with open(metrics_file, 'w') as f:
            f.write('Mode,Environment,Coverage,Redundancy,Efficiency,Path_Diversity\n')

    # Append the row
    with open(metrics_file, 'a') as f:
        f.write(f"{mode_name},{selected_env},{coverage},{redundancy},{efficiency},{diversity}\n")

    print(f"Metrics saved to {metrics_file}")

#############################################################
# ENTRY POINT
#############################################################

if __name__ == '__main__':
    main()
