import numpy as np
import matplotlib.pyplot as plt

def plot_lidar_data(file_path="first_vertical_scan.npy", heading_deg=None):
    """Plots saved LiDAR data in 3D.

    Args:
        file_path (str): Path to the saved LiDAR .npy file.
        heading_deg (int, optional): Current heading in degrees for reference. Defaults to None.
    """
    try:
        # Load the saved LiDAR data
        lidar_data = np.load(file_path)
    except FileNotFoundError:
        print(f"File not found: {file_path}. Please ensure the data is saved.")
        return

    # Check if data is valid
    if lidar_data.ndim != 2:
        print(f"Unexpected LiDAR data shape: {lidar_data.shape}")
        return

    height, width = lidar_data.shape

    # Create angles for longitude (width points around the circle)
    lon_angles = np.linspace(0, 2 * np.pi, width, endpoint=False)
    # Create angles for latitude (height points from -90 to +90 degrees)
    lat_angles = np.linspace(-np.pi / 2, np.pi / 2, height, endpoint=False)

    # Initialize arrays for 3D coordinates
    x_coords = []
    y_coords = []
    z_coords = []
    values = []

    # Calculate 3D coordinates for each point
    for lat_idx, lat in enumerate(lat_angles):
        r = lidar_data[lat_idx, :]

        # Calculate 3D coordinates
        x = r * np.cos(lat) * np.cos(lon_angles)
        y = r * np.cos(lat) * np.sin(lon_angles)
        z = r * np.sin(lat)

        x_coords.extend(x)
        y_coords.extend(y)
        z_coords.extend(z)
        values.extend(r)

    # Convert to numpy arrays
    x_coords = np.array(x_coords)
    y_coords = np.array(y_coords)
    z_coords = -1 * np.array(z_coords)  # Invert Z-axis for visualization
    values = np.array(values)

    # Create 3D plot
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection="3d")

    # Plot points with color based on distance values
    scatter = ax.scatter(
        x_coords,
        y_coords,
        z_coords,
        c=values,
        cmap="viridis",
        alpha=0.5,
        s=1,
    )
    plt.colorbar(scatter, label="Distance (m)")

    # Customize the plot
    ax.set_title("3D LiDAR Scan Visualization")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_box_aspect([1, 1, 1])
    ax.grid(True)

    # Add a reference line for the current heading, if provided
    if heading_deg is not None:
        heading_rad = np.deg2rad(heading_deg)
        ref_x = [0, np.cos(heading_rad) * 5]  # Length of 5 for visibility
        ref_y = [0, np.sin(heading_rad) * 5]
        ref_z = [0, 0]
        ax.plot(ref_x, ref_y, ref_z, "r-", linewidth=2, label="Current Heading")
        ax.legend()

    # Show the plot
    plt.show()


if __name__ == "__main__":
    # Example usage: Change file_path if the file is saved elsewhere
    plot_lidar_data(file_path="controllers\\bek_controller\\first_vertical_scan.npy")
