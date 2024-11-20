import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_point_cloud(file_path):
    """
    Load and plot the Velodyne point cloud data from a NumPy file.

    Args:
        file_path (str): Path to the NumPy file containing the point cloud data.
    """
    # Load the point cloud data
    try:
        point_cloud = np.load(file_path)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return

    # Check if the point cloud is empty
    if point_cloud.size == 0:
        print("Error: The point cloud data is empty.")
        return

    # Extract x, y, z coordinates
    x = point_cloud[:, 0]  # X-axis remains the same
    z = point_cloud[:, 1]  # Swap Y with Z
    y = point_cloud[:, 2]  # Use original Z for Y

    # Plot the point cloud
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z, c=z, cmap='viridis', s=1, alpha=0.8)
    ax.set_title("Velodyne Point Cloud")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z (Height)")
    ax.grid(True)

    plt.show()


if __name__ == "__main__":
    # Path to the saved point cloud file
    file_path = r"controllers\test_controller\velodyne_point_cloud.npy"

    # Plot the point cloud
    plot_point_cloud(file_path)
