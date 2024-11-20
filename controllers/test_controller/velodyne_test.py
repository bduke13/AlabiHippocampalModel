import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_velodyne_layer(file_path="velodyne_point_cloud.npy", layer_index=0):
    """Plots a specific layer of the Velodyne Puck point cloud in 3D.

    Args:
        file_path (str): Path to the saved Velodyne point cloud file.
        layer_index (int): Index of the layer to plot (0 for the first layer).
    """
    try:
        # Load the point cloud data
        point_cloud = np.load(file_path)
    except FileNotFoundError:
        print(f"File not found: {file_path}. Please ensure the data is saved.")
        return

    if point_cloud.ndim != 2 or point_cloud.shape[1] != 3:
        print(f"Unexpected point cloud shape: {point_cloud.shape}")
        return
    
    for i in range(3):
        start_idx = i * (point_cloud.shape[0] // 3)
        end_idx = start_idx + (point_cloud.shape[0] // 3)
        print(f"Layer {i} shape: {point_cloud[start_idx:end_idx].shape}")

    if layer_index == 3:
        print(point_cloud[start_idx:end_idx])

    # Calculate points per layer
    num_points = point_cloud.shape[0]
    num_layers = 1  # Velodyne Puck has 16 layers
    points_per_layer = num_points // num_layers

    # Validate layer index
    if layer_index < 0 or layer_index >= num_layers:
        print(f"Invalid layer index: {layer_index}. Must be between 0 and {num_layers - 1}.")
        return

    # Extract points for the specified layer
    start_idx = layer_index * points_per_layer
    end_idx = start_idx + points_per_layer
    layer_points = point_cloud[start_idx:end_idx]

    # Correct axis alignment (swap Y and Z)
    x = layer_points[:, 0]  # X remains the same
    z = layer_points[:, 1]  # Use Y as Z
    y = layer_points[:, 2]  # Use Z as Y

    # Create 3D plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    scatter = ax.scatter(x, y, z, c=z, cmap="viridis", s=5, alpha=0.8)
    plt.colorbar(scatter, label="Height (Z)")

    # Customize the plot
    ax.set_title(f"Velodyne Layer {layer_index}")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.grid(True)

    plt.show()


if __name__ == "__main__":
    # Example usage: Specify the layer index to plot (e.g., 0 for the first layer)
    plot_velodyne_layer(
        file_path="controllers/test_controller/velodyne_point_cloud.npy",
        layer_index=0  # Change this to plot different layers (0-15)
    )
