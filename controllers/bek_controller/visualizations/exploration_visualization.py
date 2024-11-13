# exploration_visualization.py

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

def load_data():
    data_dir = 'C:/Users/alexm/Documents/senior_design/AlabiHippocampalModel-1/controllers/bek_controller/'

    # Load the robot's positions
    with open(os.path.join(data_dir, 'hmap_x.pkl'), 'rb') as f:
        hmap_x = pickle.load(f)
    with open(os.path.join(data_dir, 'hmap_y.pkl'), 'rb') as f:
        hmap_y = pickle.load(f)

    # Try to load the visitation map
    visitation_map_path = os.path.join(data_dir, 'visitation_map.pkl')
    if os.path.exists(visitation_map_path):
        with open(visitation_map_path, 'rb') as f:
            visitation_map = pickle.load(f)
    else:
        visitation_map = None

    return hmap_x, hmap_y, visitation_map

def plot_exploration(hmap_x, hmap_y, visitation_map, grid_resolution):
    plt.figure(figsize=(10, 8))
    plt.plot(hmap_x, hmap_y, color='blue', linewidth=0.5, label='Robot Path')
    plt.scatter(hmap_x, hmap_y, c='blue', s=1)

    if visitation_map:
        # Extract visitation map data
        cells = np.array(list(visitation_map.keys()))
        counts = np.array(list(visitation_map.values()))
        counts_normalized = counts / np.max(counts)

        # Convert cell coordinates to positions
        cell_positions_x = cells[:, 0] * grid_resolution + grid_resolution / 2
        cell_positions_y = cells[:, 1] * grid_resolution + grid_resolution / 2

        # Create a scatter plot of the visitation map
        plt.scatter(cell_positions_x, cell_positions_y, c=counts_normalized, cmap='hot', s=100, marker='s', alpha=0.6, label='Visitation Map')
        plt.colorbar(label='Normalized Visitation Count')

    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('Robot Exploration and Visitation Map')
    plt.legend()
    plt.axis('equal')
    plt.show()

def main():
    hmap_x, hmap_y, visitation_map = load_data()
    grid_resolution = 0.5  # Same as in driver.py
    plot_exploration(hmap_x, hmap_y, visitation_map, grid_resolution)

if __name__ == '__main__':
    main()
