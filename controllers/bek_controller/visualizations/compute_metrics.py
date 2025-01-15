# compute_metrics.py

import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from math import pi
import pandas as pd

def compute_environment_coverage(visitation_map, total_cells):
    visited_cells = len(visitation_map)
    coverage = visited_cells / total_cells
    return coverage

def compute_redundancy(visitation_map):
    total_visits = sum(visitation_map.values())
    unique_cells = len(visitation_map)
    redundancy = total_visits - unique_cells
    return redundancy

def compute_exploration_efficiency(hmap_x, hmap_y, visitation_map):
    total_distance = compute_path_length(hmap_x, hmap_y)
    coverage = len(visitation_map)
    efficiency = coverage / total_distance if total_distance > 0 else 0
    return efficiency

def compute_path_length(hmap_x, hmap_y):
    path_length = np.sum(np.sqrt(np.diff(hmap_x)**2 + np.diff(hmap_y)**2))
    return path_length

def compute_path_diversity(hmap_h):
    # Convert headings to radians
    headings_rad = np.deg2rad(hmap_h)
    # Compute the circular variance
    path_diversity = 1 - np.abs(np.mean(np.exp(1j * headings_rad)))
    return path_diversity

def plot_coverage(visitation_map, grid_resolution, mode_name):
    cells = np.array(list(visitation_map.keys()))
    counts = np.array(list(visitation_map.values()))
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
    counts = np.array(list(visitation_map.values()))
    plt.figure(figsize=(8, 6))
    plt.hist(counts, bins=range(1, int(np.max(counts))+2), align='left', rwidth=0.8)
    plt.title(f'Redundancy - {mode_name}')
    plt.xlabel('Visit Count per Cell')
    plt.ylabel('Number of Cells')
    plt.show()

def plot_efficiency(efficiency, mode_name):
    plt.figure(figsize=(8, 6))
    plt.bar([mode_name], [efficiency])
    plt.title('Exploration Efficiency')
    plt.ylabel('Coverage per Unit Distance')
    plt.show()

def plot_path_diversity(hmap_h, mode_name):
    headings_deg = hmap_h % 360  # Ensure headings are in [0, 360)
    plt.figure(figsize=(8, 6))
    plt.hist(headings_deg, bins=36, range=(0, 360), density=True)
    plt.title(f'Path Diversity - {mode_name}')
    plt.xlabel('Heading (Degrees)')
    plt.ylabel('Frequency')
    plt.show()

def main():
    # Ask the user to input the exploration mode
    mode_name = input("Enter the exploration mode (e.g., 'Random Walk', 'Curiosity', 'Hybrid'): ").strip()
    if not mode_name:
        print("No exploration mode entered. Exiting.")
        return

    # Set up paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    bek_controller_dir = os.path.abspath(os.path.join(current_dir, '..'))
    data_dir = bek_controller_dir  # Data files are in the bek_controller directory

    # Define the environment boundaries and grid resolution
    grid_resolution = 0.5  # Should match the one used in driver.py
    min_x, max_x = -5, 5  # Update based on your environment
    min_y, max_y = -5, 5
    total_cells_x = int((max_x - min_x) / grid_resolution)
    total_cells_y = int((max_y - min_y) / grid_resolution)
    total_cells = total_cells_x * total_cells_y

    # Load data
    try:
        with open(os.path.join(data_dir, 'hmap_x.pkl'), 'rb') as f:
            hmap_x = np.array(pickle.load(f))
        with open(os.path.join(data_dir, 'hmap_y.pkl'), 'rb') as f:
            hmap_y = np.array(pickle.load(f))
        with open(os.path.join(data_dir, 'hmap_h.pkl'), 'rb') as f:
            hmap_h_array = np.array(pickle.load(f))
            # Process hmap_h_array to get headings
            # If hmap_h_array is (steps, n_hd), compute headings accordingly
            if hmap_h_array.ndim == 2:
                # Assuming head direction cells are evenly distributed over 360 degrees
                n_hd = hmap_h_array.shape[1]
                hd_angles = np.linspace(0, 360, n_hd, endpoint=False)
                headings = np.degrees(np.arctan2(
                    np.sum(np.sin(np.deg2rad(hd_angles)) * hmap_h_array, axis=1),
                    np.sum(np.cos(np.deg2rad(hd_angles)) * hmap_h_array, axis=1)
                )) % 360
                hmap_h = headings
            else:
                hmap_h = hmap_h_array
        with open(os.path.join(data_dir, 'visitation_map.pkl'), 'rb') as f:
            visitation_map = pickle.load(f)
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return

    # Compute metrics
    coverage = compute_environment_coverage(visitation_map, total_cells)
    redundancy = compute_redundancy(visitation_map)
    efficiency = compute_exploration_efficiency(hmap_x, hmap_y, visitation_map)
    diversity = compute_path_diversity(hmap_h)

    # Print metrics
    print(f"\nMetrics for {mode_name}:")
    print(f"Environment Coverage: {coverage:.2%}")
    print(f"Redundancy (total revisits): {redundancy}")
    print(f"Exploration Efficiency (coverage per unit distance): {efficiency:.5f}")
    print(f"Path Diversity: {diversity:.5f}")

    # Generate visualizations
    plot_coverage(visitation_map, grid_resolution, mode_name)
    plot_redundancy(visitation_map, mode_name)
    plot_efficiency(efficiency, mode_name)
    plot_path_diversity(hmap_h, mode_name)

    # Save metrics to a CSV file
    metrics_file = os.path.join(current_dir, 'metrics.csv')
    metrics_data = {
        'Mode': mode_name,
        'Environment Coverage': coverage,
        'Redundancy': redundancy,
        'Exploration Efficiency': efficiency,
        'Path Diversity': diversity,
    }

    # Check if the file exists
    if not os.path.isfile(metrics_file):
        # Create the file and write the header
        with open(metrics_file, 'w') as f:
            f.write('Mode,Environment Coverage,Redundancy,Exploration Efficiency,Path Diversity\n')

    # Append the metrics
    with open(metrics_file, 'a') as f:
        f.write(f"{metrics_data['Mode']},{metrics_data['Environment Coverage']},{metrics_data['Redundancy']},{metrics_data['Exploration Efficiency']},{metrics_data['Path Diversity']}\n")

    print(f"Metrics saved to {metrics_file}")


if __name__ == '__main__':
    main()