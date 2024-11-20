import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_velodyne_point_cloud(file_path="velodyne_point_cloud.npy"):
    """Plots the Velodyne Puck point cloud in 3D.

    Args:
        file_path (str): Path to the saved Velodyne point cloud file.
    """
    try:
        # Load the point cloud data
        point_cloud = np.load(file_path)
        print(f"point cloud shape: {point_cloud.shape}")
    except FileNotFoundError:
        print(f"File not found: {file_path}. Please ensure the data is saved.")
        return

    if point_cloud.ndim != 2 or point_cloud.shape[1] != 3:
        print(f"Unexpected point cloud shape: {point_cloud.shape}")
        return

    # Extract coordinates
    x = point_cloud[:, 0]  # X-axis
    z = point_cloud[:, 1]  # Swap Y to Z for vertical alignment
    y = point_cloud[:, 2]  # Swap Z to Y

    # Create 3D plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    scatter = ax.scatter(x, y, z, c=z, cmap="viridis", s=1, alpha=0.5)
    plt.colorbar(scatter, label="Height (Z)")

    # Customize the plot
    ax.set_title("Velodyne Point Cloud (Corrected Axes)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.grid(True)

    plt.show()

if __name__ == "__main__":
    # Plot the Velodyne Puck data
    plot_velodyne_point_cloud(r"controllers\test_controller\velodyne_point_cloud.npy")
