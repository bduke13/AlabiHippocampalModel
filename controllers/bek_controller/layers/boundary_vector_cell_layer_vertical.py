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
        if sigma_vert is None:
            sigma_vert = sigma_ang / 4

        # Preferred distances
        self.d_i = np.linspace(0, max_dist, num=50)  # (M,)
        N_dist = len(self.d_i)

        # Preferred horizontal angles
        self.phi_i = np.repeat(np.linspace(0, 2 * np.pi, n_hd, endpoint=False), N_dist)
        self.d_i = np.tile(self.d_i, n_hd)

        # Preferred vertical angles (zero for horizontal plane)
        self.phi_i_vert = np.zeros_like(self.phi_i)

        # Total number of BVC neurons
        self.num_bvc = len(self.d_i)

        # Convert parameters to tensors and expand dimensions
        self.d_i = tf.constant(self.d_i, dtype=tf.float32)[tf.newaxis, :]  # (1, M)
        self.phi_i = tf.constant(self.phi_i, dtype=tf.float32)[tf.newaxis, :]  # (1, M)
        self.phi_i_vert = tf.constant(self.phi_i_vert, dtype=tf.float32)[
            tf.newaxis, :
        ]  # (1, M)

        # Convert sigmas to radians and precompute constants
        self.sigma_ang = tf.constant(np.deg2rad(sigma_ang), dtype=tf.float32)
        self.sigma_d = tf.constant(sigma_d, dtype=tf.float32)
        self.sigma_vert = tf.constant(np.deg2rad(sigma_vert), dtype=tf.float32)

        self.two_sigma_d_squared = 2 * self.sigma_d**2
        self.two_sigma_ang_squared = 2 * self.sigma_ang**2
        self.two_sigma_vert_squared = 2 * self.sigma_vert**2

    @tf.function
    def get_bvc_activation(self, points: tf.Tensor) -> tf.Tensor:
        """Calculate the activation of BVCs based on input points.

        Args:
            points: tf.Tensor of shape (N, 6), containing:
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
        distances = points[:, 5]  # (N,)
        angles = points[:, 4]  # (N,)
        vertical_angles = points[:, 3]  # (N,)

        # Expand dimensions to (N, 1) for broadcasting
        distances = tf.expand_dims(distances, axis=1)  # (N, 1)
        angles = tf.expand_dims(angles, axis=1)  # (N, 1)
        vertical_angles = tf.expand_dims(vertical_angles, axis=1)  # (N, 1)

        # Compute differences
        delta_d = distances - self.d_i  # (N, M)
        delta_ang = tf.atan2(
            tf.sin(angles - self.phi_i), tf.cos(angles - self.phi_i)
        )  # (N, M)
        delta_vert = tf.atan2(
            tf.sin(vertical_angles - self.phi_i_vert),
            tf.cos(vertical_angles - self.phi_i_vert),
        )  # (N, M)

        # Compute Gaussians without normalization constants
        distance_gaussian = tf.exp(-(delta_d**2) / self.two_sigma_d_squared)  # (N, M)
        angular_gaussian = tf.exp(
            -(delta_ang**2) / self.two_sigma_ang_squared
        )  # (N, M)
        vertical_gaussian = tf.exp(
            -(delta_vert**2) / self.two_sigma_vert_squared
        )  # (N, M)

        # Compute the product
        gaussians = distance_gaussian * angular_gaussian * vertical_gaussian  # (N, M)

        # Sum over input points to get activations for each BVC neuron
        activation = tf.reduce_sum(gaussians, axis=0)  # (M,)

        return activation  # (M,)

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
