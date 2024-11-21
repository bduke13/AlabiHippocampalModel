# %%
import numpy as np
import matplotlib.pyplot as plt


def get_scan_points(scan_data: np.ndarray, z_cutoff: float = 2.0) -> np.ndarray:
    """
    Convert scan data into structured point information, excluding points below
    45 degrees south of horizontal and above specified z-cutoff.

    Args:
        scan_data: Raw scan data of shape (360, 720)
        z_cutoff: Maximum z-coordinate value to include (in meters)

    Returns:
        np.ndarray: Array of shape (N, 6) containing:
            [:, 0]: x coordinates
            [:, 1]: y coordinates
            [:, 2]: z coordinates
            [:, 3]: latitude angles in radians
            [:, 4]: longitude angles in radians
            [:, 5]: distance values
    """
    # Create angles for longitude and latitude
    lon_angles = np.linspace(0, 2 * np.pi, 720)
    lat_angles = np.linspace(np.pi / 2, -np.pi / 2, 360)

    # Only process points above 20 degrees south of horizontal
    # 180 is horizontal, +45 gives us the cutoff index
    cutoff_idx = 200  # 180 + 20
    filtered_data = scan_data[:cutoff_idx, :]
    filtered_lat_angles = lat_angles[:cutoff_idx]

    # Create meshgrid for angles
    lon_mesh, lat_mesh = np.meshgrid(lon_angles, filtered_lat_angles)

    # Calculate 3D coordinates using broadcasting
    r_values = filtered_data
    x = r_values * np.cos(lat_mesh) * np.cos(lon_mesh)
    y = r_values * np.cos(lat_mesh) * np.sin(lon_mesh)
    z = r_values * np.sin(lat_mesh)

    # Flatten arrays
    x_flat = x.flatten()
    y_flat = y.flatten()
    z_flat = z.flatten()
    lat_flat = lat_mesh.flatten()
    lon_flat = lon_mesh.flatten()
    r_flat = r_values.flatten()

    # Filter points based on z-coordinate
    z_mask = z_flat <= z_cutoff

    # Stack all arrays into a single array with z-filtering
    points = np.column_stack(
        (
            x_flat[z_mask],
            y_flat[z_mask],
            z_flat[z_mask],
            lat_flat[z_mask],
            lon_flat[z_mask],
            r_flat[z_mask],
        )
    )

    return points


def plot_3d_environment_with_reference_line(points: np.ndarray, env_size=10):
    """
    Plot 3D environment with specified size, 1m increments, and a reference line.

    Args:
        points: np.ndarray of shape (N, 6), containing:
            [:, 0]: x coordinates
            [:, 1]: y coordinates
            [:, 2]: z coordinates
            [:, 3]: latitude angles in radians
            [:, 4]: longitude angles in radians
            [:, 5]: distance values
        env_size: Size of the environment in meters (creates a cube of ±env_size)

    Returns:
        None
    """
    # Extract variables from points array
    x_coords = points[:, 0]
    y_coords = points[:, 1]
    z_coords = points[:, 2]
    values = points[:, 5]

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection="3d")

    # Plot the filtered points
    scatter = ax.scatter(
        x_coords,
        y_coords,
        z_coords,
        c=values,
        cmap="viridis",
        alpha=0.3,
        s=1,
    )
    plt.colorbar(scatter, label="Value")

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

    # Create the plot
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

    # Use the updated function
    points = get_scan_points(reshaped_data)

    # Call the plotting function
    plot_3d_environment_with_reference_line(points)
