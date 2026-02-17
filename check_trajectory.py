"""
Check the robot's trajectory to see if it explored the environment properly.
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
CONTROLLER_NAME = "multiscale_controller"

def check_trajectory(env_name):
    """Check the robot's trajectory."""

    hmap_loc_file = PROJECT_ROOT / "webots" / "controllers" / CONTROLLER_NAME / "pkl" / env_name / "hmaps" / "hmap_loc.pkl"

    if not hmap_loc_file.exists():
        print(f"Location file not found: {hmap_loc_file}")
        return

    with open(hmap_loc_file, "rb") as f:
        hmap_loc = np.array(pickle.load(f))
        hmap_loc = hmap_loc[1:]  # Remove first element

    print(f"\n{env_name} Trajectory:")
    print(f"  Total timesteps: {len(hmap_loc)}")
    print(f"  Location shape: {hmap_loc.shape}")

    x = hmap_loc[:, 0]
    y = hmap_loc[:, 1]  # This is vertical (height)
    z = hmap_loc[:, 2]  # This is the other horizontal dimension

    print(f"\n  X coordinate (horizontal 1):")
    print(f"    Min: {np.min(x):.3f}")
    print(f"    Max: {np.max(x):.3f}")
    print(f"    Range: {np.max(x) - np.min(x):.3f}")
    print(f"    Std: {np.std(x):.3f}")

    print(f"\n  Y coordinate (vertical/height):")
    print(f"    Min: {np.min(y):.3f}")
    print(f"    Max: {np.max(y):.3f}")
    print(f"    Range: {np.max(y) - np.min(y):.3f}")
    print(f"    Std: {np.std(y):.3f}")

    print(f"\n  Z coordinate (horizontal 2):")
    print(f"    Min: {np.min(z):.3f}")
    print(f"    Max: {np.max(z):.3f}")
    print(f"    Range: {np.max(z) - np.min(z):.3f}")
    print(f"    Std: {np.std(z):.3f}")

    # Check if robot is stuck at one location
    unique_x = len(np.unique(np.round(x, 1)))
    unique_z = len(np.unique(np.round(z, 1)))
    print(f"\n  Unique positions (rounded to 0.1m):")
    print(f"    X: {unique_x} unique positions")
    print(f"    Z: {unique_z} unique positions")

    # Create trajectory plot
    plt.figure(figsize=(10, 10))
    plt.scatter(x, z, c=range(len(x)), cmap='viridis', s=1, alpha=0.5)
    plt.colorbar(label='Timestep')
    plt.xlabel('X position (m)')
    plt.ylabel('Z position (m)')
    plt.title(f'{env_name} - Robot Trajectory')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)

    # Add arena bounds
    plt.plot([-10, 10, 10, -10, -10], [-10, -10, 10, 10, -10], 'r--', label='Arena bounds', linewidth=2)
    plt.legend()

    output_file = PROJECT_ROOT / "place_field_plots" / f"{env_name}_trajectory.png"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n  Saved trajectory plot to: {output_file}")
    plt.close()


def main():
    print("="*70)
    print("Checking Robot Trajectory")
    print("="*70)

    for env_name in ["environment_1", "environment_2"]:
        check_trajectory(env_name)


if __name__ == "__main__":
    main()
