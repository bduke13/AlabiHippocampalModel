# %%
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf  # Import TensorFlow

# Precompute all the static tensors we'll need
num_rows, num_cols = 90, 180
pi = tf.constant(np.pi, dtype=tf.float32)
lon_angles = tf.linspace(0.0, 2.0 * pi, num_cols)
lat_angles = tf.linspace(pi / 2.0, -pi / 2.0, num_rows)

# Precompute the meshgrid once
lon_mesh, lat_mesh = tf.meshgrid(lon_angles, lat_angles, indexing="xy")
# Precompute the flattened angle tensors
LON_FLAT = tf.reshape(lon_mesh, [-1])
LAT_FLAT = tf.reshape(lat_mesh, [-1])


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=(90, 180), dtype=tf.float64),
        tf.TensorSpec(shape=(), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32),
    ]
)
def get_scan_points(
    scan_data: tf.Tensor,
    top_cutoff_percentage: float = 0.0,
    bottom_cutoff_percentage: float = 0.5,
) -> tf.Tensor:
    """
    Convert scan data into structured point information, excluding points outside
    specified vertical percentage ranges.

    Args:
        scan_data: Raw scan data of shape (num_rows, num_cols)
        top_cutoff_percentage: The percentage (0 to 1) from the top to start processing
        bottom_cutoff_percentage: The percentage (0 to 1) from the top to stop processing

    Returns:
        tf.Tensor: Tensor of shape (N, 3) containing:
            [:, 0]: latitude angles in radians (float32)
            [:, 1]: longitude angles in radians (float32)
            [:, 2]: distance values (float32)
    """
    # Ensure scan_data is float32
    scan_data = tf.cast(scan_data, tf.float32)

    # Just flatten the distances - we'll use the precomputed angle tensors
    r_flat = tf.reshape(scan_data, [-1])

    # Stack tensors into a single tensor using precomputed angles
    points = tf.stack([LAT_FLAT, LON_FLAT, r_flat], axis=1)

    if top_cutoff_percentage > 0.0 or bottom_cutoff_percentage < 1.0:
        # Only apply cutoffs if needed
        top_idx = tf.cast(num_rows * top_cutoff_percentage * num_cols, tf.int32)
        bottom_idx = tf.cast(num_rows * bottom_cutoff_percentage * num_cols, tf.int32)
        points = points[top_idx:bottom_idx]

    return points


def convert_to_3D(points: tf.Tensor) -> tf.Tensor:
    """
    Convert points from spherical coordinates (latitude angles, longitude angles, distances)
    to Cartesian coordinates (x, y, z).

    Args:
        points: tf.Tensor of shape (N, 3), containing:
            [:, 0]: latitude angles in radians (float32)
            [:, 1]: longitude angles in radians (float32)
            [:, 2]: distance values (float32)

    Returns:
        tf.Tensor: Tensor of shape (N, 3), containing:
            [:, 0]: x coordinates (float32)
            [:, 1]: y coordinates (float32)
            [:, 2]: z coordinates (float32)
    """
    # Ensure points are float32
    points = tf.cast(points, tf.float32)

    # Extract variables from points tensor
    lat_angles = points[:, 0]  # (N,)
    lon_angles = points[:, 1]  # (N,)
    distances = points[:, 2]  # (N,)

    # Compute x, y, z coordinates using TensorFlow operations
    x_coords = distances * tf.cos(lat_angles) * tf.cos(lon_angles)
    y_coords = distances * tf.cos(lat_angles) * tf.sin(lon_angles)
    z_coords = distances * tf.sin(lat_angles)

    # Stack the coordinates into a tensor
    xyz_coords = tf.stack([x_coords, y_coords, z_coords], axis=1)

    return xyz_coords


def plot_3d_environment_with_reference_line(points: tf.Tensor, env_size=10):
    """
    Plot 3D environment with specified size, 1m increments, and a reference line.

    Args:
        points: tf.Tensor of shape (N, 3), containing:
            [:, 0]: latitude angles in radians (float32)
            [:, 1]: longitude angles in radians (float32)
            [:, 2]: distance values (float32)

        env_size: Size of the environment in meters (creates a cube of ±env_size)

    Returns:
        None
    """
    # Convert points to x, y, z coordinates
    xyz_coords = convert_to_3D(points)
    x_coords = xyz_coords[:, 0].numpy()
    y_coords = xyz_coords[:, 1].numpy()
    z_coords = xyz_coords[:, 2].numpy()
    distances = points[:, 2].numpy()  # Distance values for coloring

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection="3d")

    # Plot the points
    scatter = ax.scatter(
        x_coords,
        y_coords,
        z_coords,
        c=distances,
        cmap="viridis",
        alpha=0.3,
        s=1,
    )
    plt.colorbar(scatter, label="Distance (meters)")

    # Plot the red dot at the origin
    ax.scatter(
        0,
        0,
        0,
        color="red",
        s=100,
        marker="o",
        label="Scanner Position",
        zorder=5,
    )
    ax.legend()

    # Set axis limits
    ax.set_xlim(-env_size, env_size)
    ax.set_ylim(-env_size, env_size)
    ax.set_zlim(-env_size, env_size)

    # Set ticks every 1m
    ticks = np.arange(-env_size, env_size + 1, 1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_zticks(ticks)

    # Customize the plot
    ax.set_title(
        f"3D Scan Projection (Above -45° Latitude) with Reference Line (Env Size: ±{env_size}m)"
    )
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.set_zlabel("Z (meters)")

    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])

    # Add grid
    ax.grid(True)

    plt.show()


if __name__ == "__main__":
    # Load the data
    vertical_boundaries = np.load("first_vertical_scan.npy")

    # Reshape the data from (259200,) to (360, 720)
    reshaped_data = vertical_boundaries.reshape(360, 720)

    # Create the plot for the reshaped data
    plt.figure(figsize=(12, 6))
    plt.imshow(reshaped_data, cmap="viridis")
    plt.colorbar(label="Value")
    plt.title("Vertical Boundaries Data")
    plt.xlabel("Longitude (720 points)")
    plt.ylabel("Latitude (360 points)")
    plt.show()

    # Get the middle scan line (180th row)
    middle_scan = reshaped_data[180, :]
    center_point = middle_scan[720 // 2]

    # Add intermediate visualization of the 180th scan line as a horizontal bar
    plt.figure(figsize=(12, 2))
    plt.imshow(middle_scan.reshape(1, -1), cmap="viridis", aspect="auto")
    plt.colorbar(label="Value")
    plt.title("180th Scan Line (Horizontal Bar)")
    plt.xlabel("Longitude (720 points)")
    plt.axvline(x=360, color="r", linestyle="--", label="Center (360th point)")
    plt.legend()
    plt.yticks([])  # Remove y-axis ticks since it's just one line
    plt.show()

    # Use the updated get_scan_points function
    points = get_scan_points(reshaped_data)

    # Call the plotting function
    plot_3d_environment_with_reference_line(points)
