# %%
from typing import Union
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


class BoundaryVectorCellLayer:
    def __init__(
        self,
        max_dist: float,
        input_dim: int = 720,
        n_hd: int = 8,
        sigma_ang: float = 90,
        sigma_d: float = 0.5,
        sigma_vert: float = None,  # if None, uses sigma_ang
        vertical_angles: list[float] = [0],
        # [0, 15, 30, 45],  # degrees up from horizontal
        horiz_angles: np.array = np.linspace(
            0, 2 * np.pi, 720, endpoint=False, dtype=np.float32
        ),
        layer_indices: list[int] = [180],  # indices to sample from vertical scan
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
        # Store angle configurations with explicit float32 dtype
        self.vertical_angles = np.array(vertical_angles, dtype=np.float32)
        self.horiz_angles = horiz_angles
        # Store layer indices
        self.layer_indices = layer_indices
        # [x for x in range(360 / 2, 360, 1)]

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
        vert_angles = np.repeat(np.deg2rad(vertical_angles), n_hd * N_dist).astype(
            np.float32
        )
        self.theta_i = vert_angles[np.newaxis, :]

        # Preferred horizontal angles for each BVC
        self.phi_i = np.tile(
            np.linspace(0, 2 * np.pi, input_dim, dtype=np.float32)[horiz_indices],
            len(vertical_angles),
        )[np.newaxis, :]

        # Angular standard deviation for horizontal angle tuning (converted to radians)
        self.sigma_ang = tf.constant(np.deg2rad(sigma_ang), dtype=tf.float32)

        # Angular standard deviation for vertical angle tuning (converted to radians)
        self.sigma_vert = tf.constant(
            np.deg2rad(sigma_vert if sigma_vert is not None else sigma_ang),
            dtype=tf.float32,
        )

        # Distance standard deviation for the Gaussian function
        self.sigma_d = tf.constant(sigma_d, dtype=tf.float32)

    def get_bvc_activation(
        self,
        distances: np.ndarray,
    ) -> tf.Tensor:
        """Calculate the activation of BVCs based on input distances.

        Args:
            distances: 2D array of shape (360, 720) containing vertical scan data in polar coordinates,
                      where rows are vertical angles and columns are horizontal angles.

        Returns:
            Activations of the BVC neurons, including vertical projections.
        """
        # Convert inputs to tensors
        distances = tf.convert_to_tensor(distances, dtype=tf.float32)

        # Initialize list to store activations for each vertical angle
        all_activations = []

        # Process each vertical angle
        for i, v_angle in enumerate(self.vertical_angles):
            # Get the corresponding layer from vertical scan
            layer_idx = self.layer_indices[i]
            vert_distances = distances[layer_idx, :]

            # Calculate start and end indices for this vertical angle's neurons
            start_idx = i * (self.num_bvc // len(self.vertical_angles))
            end_idx = (i + 1) * (self.num_bvc // len(self.vertical_angles))

            # Get input distances and angles for this set of cells
            input_distances = tf.gather(
                vert_distances, self.input_indices[:, start_idx:end_idx]
            )
            input_horiz_angles = tf.gather(
                self.horiz_angles, self.input_indices[:, start_idx:end_idx]
            )

            # Calculate distance tuning
            distance_gaussian = tf.exp(
                -(tf.square(input_distances - self.d_i[:, start_idx:end_idx]))
                / (2 * tf.square(self.sigma_d))
            )

            # Calculate horizontal angle tuning
            angle_diff = input_horiz_angles - self.phi_i[:, start_idx:end_idx]
            horiz_gaussian = tf.exp(
                -(tf.square(angle_diff)) / (2 * tf.square(self.sigma_ang))
            )

            # Calculate vertical angle tuning
            v_angle_rad = tf.constant(np.deg2rad(v_angle), dtype=tf.float32)
            vert_diff = v_angle_rad - self.theta_i[:, start_idx:end_idx]
            vert_gaussian = tf.exp(
                -(tf.square(vert_diff)) / (2 * tf.square(self.sigma_vert))
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
        return_plot: bool = False,
    ) -> Union[None, plt.Figure]:
        """Plot the BVC activation in 3D showing vertical projections.

        Args:
            distances: 2D array of vertical scan data in polar coordinates.
            horiz_angles: 2D array of horizontal angles (azimuth).
            return_plot: If True, returns the plot object instead of showing it.

        Returns:
            The matplotlib Figure object if return_plot is True, otherwise None.
        """
        # Get BVC activations based on vertical scan and angles
        activations = self.get_bvc_activation(distances).numpy()

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
            scan_distances = distances[layer_idx, :]  # Changed from [:, layer_idx]

            # Convert to cartesian coordinates
            x_scan = scan_distances * np.cos(self.horiz_angles) * np.cos(v_angle_rad)
            y_scan = scan_distances * np.sin(self.horiz_angles) * np.cos(v_angle_rad)
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
    # Load the real data
    vertical_boundaries = np.load("first_vertical_scan.npy")

    # Reshape the data from (259200,) to (360, 720)
    reshaped_data = vertical_boundaries.reshape(360, 720)

    distances = reshaped_data

    # Initialize BVC layer with horizontal scan (0 degrees at 180th row)
    vertical_angles = [0, 15, 30, 45]  # horizontal scan
    layer_indices = [
        180,
        180 - 15,
        180 - 30,
        180 - 45,
    ]  # 180th row represents horizontal scan

    bvc_layer = BoundaryVectorCellLayer(
        max_dist=10,  # Adjust based on your data's range
        input_dim=720,
        n_hd=8,
        sigma_ang=90,
        sigma_d=0.5,
        vertical_angles=vertical_angles,
        layer_indices=layer_indices,
    )

    # Plot BVC activation with the real scan data
    bvc_layer.plot_activation(distances)
