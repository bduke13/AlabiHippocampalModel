# %%
from typing import Union
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


class BoundaryVectorCellLayer:
    def __init__(
        self,
        max_dist: float,
        input_dim: int,
        n_hd: int,
        sigma_ang: float,
        sigma_d: float,
        vertical_angles: list[float] = [0, 15, 30, 45],  # degrees up from horizontal
        layer_indices: list[int] = None,  # indices to sample from vertical scan
    ) -> None:
        """Initialize the boundary vector cell (BVC) layer.

        This layer models neurons that respond to obstacles at specific distances and angles, including
        vertical projections. It creates BVC cells for multiple vertical angles, each having n_hd head
        directions with multiple neurons per head direction.

        Args:
            max_dist: Max distance that the BVCs respond to. Units depend on the context of the environment.
            input_dim: Size of horizontal input vector to the BVC layer (e.g., 720 for RPLidar).
            n_hd: Number of head direction cells per vertical angle.
            sigma_ang: Standard deviation (tuning width) for the Gaussian function modeling angular tuning of BVCs (in degrees).
            sigma_d: Standard deviation (tuning width) for the Gaussian function modeling distance tuning of BVCs.
            vertical_angles: List of vertical angles (in degrees) to project BVCs at.
            layer_indices: List of indices to sample from the vertical scan. If None, uses [0, 119, 239, 359].
        """
        # Store vertical configuration
        self.vertical_angles = np.array(vertical_angles)
        self.layer_indices = layer_indices or [
            0,
            119,
            239,
            359,
        ]  # Default indices if None

        # Compute the number of preferred distances per head direction
        N_dist = len(np.arange(0, max_dist, sigma_d / 2))

        # Create arrays for each vertical angle
        d_i_single = np.arange(0, max_dist, sigma_d / 2)
        self.d_i = np.tile(d_i_single, n_hd * len(vertical_angles))[np.newaxis, :]

        # Total number of BVC neurons = vertical_angles * head directions * preferred distances
        self.num_bvc = self.d_i.size

        # Create horizontal angle indices for each vertical angle
        horiz_indices = np.repeat(
            np.linspace(0, input_dim, n_hd, endpoint=False, dtype=int),
            N_dist,
        )
        self.input_indices = np.tile(horiz_indices, len(vertical_angles))[np.newaxis, :]

        # Create vertical angle preferences (repeated for each horizontal direction and distance)
        vert_angles = np.repeat(np.deg2rad(vertical_angles), n_hd * N_dist)
        self.theta_i = vert_angles[np.newaxis, :]

        # Preferred horizontal angles for each BVC
        self.phi_i = np.tile(
            np.linspace(0, 2 * np.pi, input_dim)[horiz_indices], len(vertical_angles)
        )[np.newaxis, :]

        # Angular standard deviation for the Gaussian function (converted to radians).
        self.sigma_ang = tf.constant(np.deg2rad(sigma_ang), dtype=tf.float32)

        # Distance standard deviation for the Gaussian function.
        self.sigma_d = tf.constant(sigma_d, dtype=tf.float32)

    def get_bvc_activation(
        self, distances: np.ndarray, angles: np.ndarray, vertical_scan: np.ndarray
    ) -> tf.Tensor:
        """Calculate the activation of BVCs based on input distances and angles.

        Args:
            distances: Array of distance readings for horizontal scan.
            angles: Array of horizontal angles corresponding to the distance readings.
            vertical_scan: 2D array of shape (720, 360) containing vertical scan data.

        Returns:
            Activations of the BVC neurons, including vertical projections.
        """
        # Convert inputs to tensors
        distances = tf.convert_to_tensor(distances, dtype=tf.float32)
        angles = tf.convert_to_tensor(angles, dtype=tf.float32)
        vertical_scan = tf.convert_to_tensor(vertical_scan, dtype=tf.float32)

        # Initialize list to store activations for each vertical angle
        all_activations = []

        # Process each vertical angle
        for i, v_angle in enumerate(self.vertical_angles):
            # Get the corresponding layer from vertical scan
            layer_idx = self.layer_indices[i]
            vert_distances = vertical_scan[:, layer_idx]

            # Calculate start and end indices for this vertical angle's neurons
            start_idx = i * (self.num_bvc // len(self.vertical_angles))
            end_idx = (i + 1) * (self.num_bvc // len(self.vertical_angles))

            # Get input distances and angles for this set of cells
            input_distances = tf.gather(
                vert_distances, self.input_indices[:, start_idx:end_idx]
            )
            input_angles = tf.gather(angles, self.input_indices[:, start_idx:end_idx])

            # Calculate distance tuning
            distance_gaussian = tf.exp(
                -(tf.square(input_distances - self.d_i[:, start_idx:end_idx]))
                / (2 * tf.square(self.sigma_d))
            )

            # Calculate horizontal angle tuning
            angle_diff = input_angles - self.phi_i[:, start_idx:end_idx]
            horiz_gaussian = tf.exp(
                -(tf.square(angle_diff)) / (2 * tf.square(self.sigma_ang))
            )

            # Calculate vertical angle tuning
            v_angle_rad = tf.constant(np.deg2rad(v_angle), dtype=tf.float32)
            vert_diff = v_angle_rad - self.theta_i[:, start_idx:end_idx]
            vert_gaussian = tf.exp(
                -(tf.square(vert_diff)) / (2 * tf.square(self.sigma_ang))
            )

            # Combine all tuning factors
            activation = tf.reduce_sum(
                distance_gaussian * horiz_gaussian * vert_gaussian, axis=0
            )
            all_activations.append(activation)

        # Combine activations from all vertical angles
        return tf.concat(all_activations, 0)

    def plot_activation(
        self,
        distances: np.ndarray,
        angles: np.ndarray,
        vertical_scan: np.ndarray,
        return_plot: bool = False,
    ) -> Union[None, plt.Figure]:
        """Plot the BVC activation in 3D showing vertical projections.

        Args:
            distances: Input distances to the BVC layer (e.g., from a LiDAR).
            angles: Input angles corresponding to the distance measurements.
            vertical_scan: 2D array of vertical scan data.
            return_plot: If True, returns the plot object instead of showing it.

        Returns:
            The matplotlib Figure object if return_plot is True, otherwise None.
        """
        # Get BVC activations based on distances, angles and vertical scan
        activations = self.get_bvc_activation(distances, angles, vertical_scan).numpy()

        # Create 3D plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection="3d")

        # Calculate points per vertical angle
        points_per_angle = self.num_bvc // len(self.vertical_angles)

        # Plot BVC activations for each vertical angle
        for v_idx, v_angle in enumerate(self.vertical_angles):
            start_idx = v_idx * points_per_angle
            end_idx = (v_idx + 1) * points_per_angle

            # Get activations for this vertical angle
            v_activations = activations[start_idx:end_idx]

            # Convert polar to cartesian coordinates
            r = self.d_i[0, start_idx:end_idx]
            theta = self.phi_i[0, start_idx:end_idx]
            v_angle_rad = np.deg2rad(v_angle)

            # Calculate 3D coordinates
            x = r * np.cos(theta) * np.cos(v_angle_rad)
            y = r * np.sin(theta) * np.cos(v_angle_rad)
            z = r * np.sin(v_angle_rad)

            # Normalize activations for color mapping with epsilon to show inactive points
            epsilon = 0.01  # Small value to make inactive points visible
            norm_activations = v_activations / np.max(activations)

            # Add epsilon to sizes to show all points
            sizes = (norm_activations + epsilon) * 100

            # Plot the points
            scatter = ax.scatter(
                x,
                y,
                z,
                c=norm_activations,
                cmap="viridis",
                s=sizes,
                alpha=0.6,
            )

        # Plot the raw scan data as small points
        for v_idx, layer_idx in enumerate(self.layer_indices):
            v_angle_rad = np.deg2rad(self.vertical_angles[v_idx])
            scan_distances = vertical_scan[:, layer_idx]

            # Convert to cartesian coordinates
            x_scan = scan_distances * np.cos(angles) * np.cos(v_angle_rad)
            y_scan = scan_distances * np.sin(angles) * np.cos(v_angle_rad)
            z_scan = scan_distances * np.sin(v_angle_rad)

            ax.scatter(
                x_scan,
                y_scan,
                z_scan,
                c="red",
                s=1,
                alpha=0.3,
                label=f"{self.vertical_angles[v_idx]}°" if v_idx == 0 else "",
            )

        # Add colorbar
        plt.colorbar(scatter, label="BVC Activation")

        # Set labels and title
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title("3D BVC Activation with Vertical Projections")

        # Set equal aspect ratio
        max_range = (
            np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max()
            / 2.0
        )

        mid_x = (x.max() + x.min()) * 0.5
        mid_y = (y.max() + y.min()) * 0.5
        mid_z = (z.max() + z.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(0, mid_z + max_range)

        # Add legend
        ax.legend()

        # Set initial view angle
        ax.view_init(elev=30, azim=45)

        # If return_plot is True, return the figure object, otherwise show the plot
        if return_plot:
            return fig
        else:
            plt.show()


if __name__ == "__main__":
    # Generate synthetic data for testing
    n_horiz: int = 720  # horizontal resolution
    n_vert: int = 360  # vertical resolution
    max_r: float = 10
    min_r: float = 5

    # Create horizontal angles (0 to 2π)
    horiz_angles: np.ndarray = np.linspace(0, 2 * np.pi, n_horiz, endpoint=False)

    # Create vertical angles (0 to 45 degrees)
    vert_angles: np.ndarray = np.linspace(0, np.pi / 4, n_vert, endpoint=True)

    # Create a meshgrid for both angles
    H, V = np.meshgrid(horiz_angles, vert_angles, indexing="ij")

    # Generate synthetic hemisphere data (720x360)
    # Create a uniform hemisphere at distance max_r
    vertical_scan = np.ones((n_horiz, n_vert)) * max_r

    # Add a "room" by creating a hemisphere of points
    # We'll make a hemisphere with radius max_r and some objects inside

    # Create a hemisphere shell at min_r (inner wall)
    for i in range(n_horiz):
        for j in range(n_vert):
            # Convert to 3D coordinates
            phi = horiz_angles[i]  # azimuth
            theta = vert_angles[j]  # elevation

            # Add some variation to make it interesting
            # Create a bumpy surface with some sine waves
            r = min_r + 0.5 * np.sin(4 * phi) * np.cos(2 * theta)

            # Add some columns
            if np.abs(np.sin(4 * phi)) > 0.95:
                r = min_r - 1.0

            # Add a central pillar
            if np.sqrt((np.cos(phi)) ** 2 + (np.sin(phi)) ** 2) < 0.1:
                r = min_r - 2.0

            vertical_scan[i, j] = r

    # Get the horizontal scan (at 0 degrees elevation)
    distances = vertical_scan[:, 0]

    # Initialize BVC layer with vertical projections
    # Using specific vertical angles and corresponding indices
    vertical_angles = [0, 15, 30, 45]  # degrees
    # Calculate indices for these angles in the 360-point vertical resolution
    layer_indices = [int(angle * (n_vert - 1) / 45) for angle in vertical_angles]

    bvc_layer = BoundaryVectorCellLayer(
        max_dist=12,
        input_dim=720,
        n_hd=8,
        sigma_ang=90,
        sigma_d=0.5,
        vertical_angles=vertical_angles,
        layer_indices=layer_indices,
    )

    # Plot BVC activation with the vertical scan data
    bvc_layer.plot_activation(distances, horiz_angles, vertical_scan)
