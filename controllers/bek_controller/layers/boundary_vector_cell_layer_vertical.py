# %%
from typing import Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import tensorflow as tf
from vis_3d_scan import get_scan_points


class BoundaryVectorCellLayer:
    def __init__(
        self,
        max_dist: float,
        n_hd: int,
        sigma_ang: float,
        sigma_d: float,
        sigma_vert: float = None,
    ) -> None:
        """Initialize the boundary vector cell (BVC) layer.

        This layer models neurons that respond to obstacles at specific distances and angles, including vertical sensitivity.

        Args:
            max_dist: Max distance that the BVCs respond to.
            n_hd: Number of head direction cells (number of preferred horizontal angles).
            sigma_ang: Standard deviation for horizontal angular tuning (in degrees).
            sigma_d: Standard deviation for distance tuning.
            sigma_vert: Standard deviation for vertical angular tuning (in degrees). If None, set to sigma_ang / 2.
        """
        if sigma_vert is None:
            sigma_vert = sigma_ang / 4

        # Preferred distances
        self.d_i = np.linspace(0, max_dist, num=50)  # Adjust number as needed
        N_dist = len(self.d_i)

        # Preferred horizontal angles
        self.phi_i = np.repeat(np.linspace(0, 2 * np.pi, n_hd, endpoint=False), N_dist)
        self.d_i = np.tile(self.d_i, n_hd)

        # Preferred vertical angles (zero for horizontal plane)
        self.phi_i_vert = np.zeros_like(self.phi_i)

        # Total number of BVC neurons
        self.num_bvc = len(self.d_i)

        # Convert sigmas to radians
        self.sigma_ang = tf.constant(np.deg2rad(sigma_ang), dtype=tf.float32)
        self.sigma_d = tf.constant(sigma_d, dtype=tf.float32)
        self.sigma_vert = tf.constant(np.deg2rad(sigma_vert), dtype=tf.float32)

    def get_bvc_activation(self, points: np.ndarray) -> tf.Tensor:
        """Calculate the activation of BVCs based on input points.

        Args:
            points: np.ndarray of shape (N, 6), containing:
                [:, 0]: x coordinates
                [:, 1]: y coordinates
                [:, 2]: z coordinates
                [:, 3]: latitude angles in radians
                [:, 4]: longitude angles in radians
                [:, 5]: distance values

        Returns:
            Activations of the BVC neurons.
        """
        # Extract distances, horizontal angles, and vertical angles from points
        distances = points[:, 5]  # distance values
        angles = points[:, 4]  # longitude angles
        vertical_angles = points[:, 3]  # latitude angles

        # Convert input arrays to tensors
        distances = tf.convert_to_tensor(distances, dtype=tf.float32)
        angles = tf.convert_to_tensor(angles, dtype=tf.float32)
        vertical_angles = tf.convert_to_tensor(vertical_angles, dtype=tf.float32)

        # Reshape input arrays to (N, 1)
        distances = tf.reshape(distances, (-1, 1))  # shape (N, 1)
        angles = tf.reshape(angles, (-1, 1))  # shape (N, 1)
        vertical_angles = tf.reshape(vertical_angles, (-1, 1))  # shape (N, 1)

        # Expand BVC parameters to (1, M)
        d_i = tf.constant(self.d_i, dtype=tf.float32)[tf.newaxis, :]  # shape (1, M)
        phi_i = tf.constant(self.phi_i, dtype=tf.float32)[tf.newaxis, :]  # shape (1, M)
        phi_i_vert = tf.constant(self.phi_i_vert, dtype=tf.float32)[
            tf.newaxis, :
        ]  # shape (1, M)

        # Compute differences
        delta_d = distances - d_i  # shape (N, M)
        delta_ang = angles - phi_i  # shape (N, M)
        delta_vert = vertical_angles - phi_i_vert  # shape (N, M)

        PI = tf.constant(np.pi, dtype=tf.float32)

        # Compute Gaussians
        distance_gaussian = tf.exp(
            -((delta_d) ** 2) / (2 * self.sigma_d**2)
        ) / tf.sqrt(2 * PI * self.sigma_d**2)
        angular_gaussian = tf.exp(
            -((delta_ang) ** 2) / (2 * self.sigma_ang**2)
        ) / tf.sqrt(2 * PI * self.sigma_ang**2)
        vertical_gaussian = tf.exp(
            -((delta_vert) ** 2) / (2 * self.sigma_vert**2)
        ) / tf.sqrt(2 * PI * self.sigma_vert**2)

        # Compute the product
        gaussians = (
            distance_gaussian * angular_gaussian * vertical_gaussian
        )  # shape (N, M)

        # Sum over input points
        activation = tf.reduce_sum(gaussians, axis=0)  # shape (M,)

        return activation  # shape (M,)

    def plot_activation(
        self,
        points: np.ndarray,
        return_plot: bool = False,
    ) -> Union[None, plt.Figure]:
        # Get BVC activations
        activations = self.get_bvc_activation(
            tf.convert_to_tensor(points, dtype=tf.float32)
        ).numpy()

        # Compute BVC positions
        x_i = self.d_i * np.cos(self.phi_i)
        y_i = self.d_i * np.sin(self.phi_i)
        z_i = self.d_i * np.sin(
            self.phi_i_vert
        )  # Will be zero since phi_i_vert is zero

        # For non-active BVCs, set a small epsilon activation
        epsilon = 1e-3
        activations = np.maximum(activations, epsilon)

        # Normalize activations for plotting
        activations_normalized = activations / np.max(activations)

        # Create the 3D plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection="3d")

        # Downsample environment points for faster plotting
        downsample_rate = 10  # Adjust as needed
        env_points = points[::downsample_rate]

        # Plot the environment points with more vibrant appearance
        ax.scatter(
            env_points[:, 0],
            env_points[:, 1],
            env_points[:, 2],
            c="dodgerblue",  # More vibrant blue color
            s=3,  # Larger point size
            alpha=0.6,  # More opacity
            label="Environment",
        )

        # Plot the BVC activations
        sizes = activations_normalized * 100  # Adjust scaling factor as needed
        colors = plt.cm.viridis(activations_normalized)

        ax.scatter(
            x_i,
            y_i,
            z_i,
            s=sizes,
            c=colors,
            alpha=0.8,
            edgecolor="black",
            label="BVC Activations",
        )

        # Plot the scanner position at (0, 0, 0)
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

        # Set labels
        ax.set_xlabel("X (meters)")
        ax.set_ylabel("Y (meters)")
        ax.set_zlabel("Z (meters)")

        # Set ticks at intervals of 1
        ax.xaxis.set_major_locator(plt.MultipleLocator(1))
        ax.yaxis.set_major_locator(plt.MultipleLocator(1))
        ax.zaxis.set_major_locator(plt.MultipleLocator(1))

        # Set equal aspect ratio
        ax.set_box_aspect([1, 1, 1])

        # Set consistent axis limits
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_zlim(-10, 10)

        # Show or return the plot
        if return_plot:
            return fig
        else:
            plt.show()


if __name__ == "__main__":
    # Load the data
    vertical_boundaries = np.load("first_vertical_scan.npy")

    # Reshape the data from (259200,) to (360, 720)
    reshaped_data = vertical_boundaries.reshape(360, 720)

    # Use the updated function
    points = get_scan_points(reshaped_data)

    # Initialize BVC layer
    bvc_layer = BoundaryVectorCellLayer(
        max_dist=12, n_hd=8, sigma_ang=1, sigma_d=0.2, sigma_vert=0.1
    )
    bvc_layer.plot_activation(points)
