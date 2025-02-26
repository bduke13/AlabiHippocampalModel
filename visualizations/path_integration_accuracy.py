import numpy as np
import os
import matplotlib.pyplot as plt
from vis_utils import load_hmaps, convert_xzy_hmaps, OUTPUT_DIR

def compute_path_integration_drift(hmap_x, hmap_y, hmap_grid, output_dir=OUTPUT_DIR):
    """Compute and plot the drift between actual and estimated positions."""
    # Placeholder for estimated positions (replace with your grid cell path integration logic)
    estimated_positions = np.cumsum(hmap_grid.mean(axis=1) * 0.1)  # Example drift
    actual_positions = np.sqrt(hmap_x**2 + hmap_y**2)  # Euclidean distance from origin

    drift = np.abs(estimated_positions - actual_positions)

    plt.figure(figsize=(10, 5))
    plt.plot(drift, label="Position Drift")
    plt.xlabel("Time Step")
    plt.ylabel("Drift (meters)")
    plt.title("Path Integration Drift")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "path_integration_drift.png"))
    plt.show()

if __name__ == "__main__":
    hmap_loc, hmap_grid = load_hmaps(["hmap_loc", "hmap_grid"])
    hmap_x, _, hmap_y = convert_xzy_hmaps(hmap_loc)
    compute_path_integration_drift(hmap_x, hmap_y, hmap_grid)