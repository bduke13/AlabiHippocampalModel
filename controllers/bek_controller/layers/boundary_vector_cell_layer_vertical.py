# %%
# Import statements remain the same
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
        preferred_vertical_angles: list,
        sigma_d_list: list,
        sigma_ang_list: list,
        sigma_vert_list: list,
        scaling_factors: list,  # New parameter for scaling factors
    ) -> None:
        """
        Initialize the BoundaryVectorCellLayer with per-layer sigma values.

        Args:
            max_dist: Maximum distance for BVCs.
            n_hd: Number of preferred horizontal angles.
            preferred_vertical_angles: List of preferred vertical angles in radians.
            sigma_d_list: List of sigma_d values for each vertical layer.
            sigma_ang_list: List of sigma_ang values for each vertical layer.
            sigma_vert_list: List of sigma_vert values for each vertical layer.
            scaling_factors: List of scaling factors for each vertical layer.
        """
        # Preferred distances
        self.d_i = np.linspace(0, max_dist, num=50)  # (M,)
        N_dist = len(self.d_i)

        # Preferred horizontal angles
        self.phi_i = np.repeat(np.linspace(0, 2 * np.pi, n_hd, endpoint=False), N_dist)
        self.d_i = np.tile(self.d_i, n_hd)

        # Preferred vertical angles are expected to be in radians
        preferred_vertical_angles_rad = preferred_vertical_angles  # Already in radians

        # Generate BVC neurons for each vertical angle
        self.phi_i_vert = []
        self.d_i_all = []
        self.phi_i_all = []
        self.sigma_d_all = []
        self.sigma_ang_all = []
        self.sigma_vert_all = []
        self.scaling_factors_all = []

        for idx, vert_angle in enumerate(preferred_vertical_angles_rad):
            phi_i_vert_layer = [vert_angle] * len(self.phi_i)
            self.phi_i_vert.extend(phi_i_vert_layer)
            self.d_i_all.extend(self.d_i)
            self.phi_i_all.extend(self.phi_i)

            # Assign sigma_d, sigma_ang, sigma_vert for this layer
            sigma_d_layer = [sigma_d_list[idx]] * len(self.phi_i)
            sigma_ang_layer = [sigma_ang_list[idx]] * len(self.phi_i)
            sigma_vert_layer = [sigma_vert_list[idx]] * len(self.phi_i)
            self.sigma_d_all.extend(sigma_d_layer)
            self.sigma_ang_all.extend(sigma_ang_layer)
            self.sigma_vert_all.extend(sigma_vert_layer)

            # Assign scaling factors for this layer
            scaling_factor_layer = [scaling_factors[idx]] * len(self.phi_i)
            self.scaling_factors_all.extend(scaling_factor_layer)

        # Update BVC parameters
        self.d_i = np.array(self.d_i_all)
        self.phi_i = np.array(self.phi_i_all)
        self.phi_i_vert = np.array(self.phi_i_vert)
        self.sigma_d_array = np.array(self.sigma_d_all)
        self.sigma_ang_array = np.array(self.sigma_ang_all)
        self.sigma_vert_array = np.array(self.sigma_vert_all)
        self.scaling_factors_array = np.array(self.scaling_factors_all)

        # Total number of BVC neurons
        self.num_bvc = len(self.d_i)

        # Convert parameters to tensors WITHOUT expanding dimensions
        self.d_i = tf.constant(self.d_i, dtype=tf.float32)  # (M,)
        self.phi_i = tf.constant(self.phi_i, dtype=tf.float32)  # (M,)
        self.phi_i_vert = tf.constant(self.phi_i_vert, dtype=tf.float32)  # (M,)
        self.sigma_d = tf.constant(self.sigma_d_array, dtype=tf.float32)  # (M,)
        self.sigma_ang = tf.constant(self.sigma_ang_array, dtype=tf.float32)  # (M,)
        self.sigma_vert = tf.constant(self.sigma_vert_array, dtype=tf.float32)  # (M,)
        self.scaling_factors = tf.constant(
            self.scaling_factors_array, dtype=tf.float32
        )  # (M,)

        # Precompute denominators
        self.two_sigma_d_squared = 2 * self.sigma_d**2  # (M,)
        self.two_sigma_ang_squared = 2 * self.sigma_ang**2  # (M,)
        self.two_sigma_vert_squared = 2 * self.sigma_vert**2  # (M,)

    @tf.function(input_signature=[tf.TensorSpec([None, 6], tf.float32)])
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
        # Ensure points are float32
        points = tf.cast(points, tf.float32)

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

        # Apply scaling factors
        activation = activation * self.scaling_factors  # (M,)

        # Apply non-linear activation function
        activation = tf.pow(activation, 2)

        # Apply thresholding
        threshold = tf.constant(0.01, dtype=tf.float32)  # Adjust threshold as needed
        activation = tf.where(
            activation < threshold, tf.zeros_like(activation), activation
        )

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

        print("activations.shape:", activations.shape)  # For debugging
        # Compute BVC positions
        d_i = self.d_i.numpy().flatten()
        phi_i = self.phi_i.numpy().flatten()
        phi_i_vert = self.phi_i_vert.numpy().flatten()
        print("d_i.shape:", d_i.shape)  # For debugging
        print("phi_i.shape:", phi_i.shape)  # For debugging
        print("phi_i_vert.shape:", phi_i_vert.shape)  # For debugging

        x_i = d_i * np.cos(phi_i) * np.cos(phi_i_vert)
        y_i = d_i * np.sin(phi_i) * np.cos(phi_i_vert)
        z_i = d_i * np.sin(phi_i_vert)

        # For non-active BVCs, set a small epsilon activation
        epsilon = 1e-2
        activations = np.maximum(activations, epsilon)

        # Normalize activations for plotting
        activations_normalized = activations / np.max(activations)

        print(
            "activations_normalized.shape:", activations_normalized.shape
        )  # For debugging

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

        # Unique vertical angles for plotting
        unique_vert_angles = np.unique(phi_i_vert)
        colors_list = [
            "red",
            "green",
            "orange",
            "purple",
            "cyan",
        ]  # Extend colors as needed

        for idx, vert_angle in enumerate(unique_vert_angles):
            mask = phi_i_vert == vert_angle
            x_i_layer = x_i[mask]
            y_i_layer = y_i[mask]
            z_i_layer = z_i[mask]
            activations_layer = activations_normalized[mask]
            sizes = activations_layer * 100  # Adjust scaling factor as needed

            # Use different colors for different layers
            color = colors_list[idx % len(colors_list)]
            ax.scatter(
                x_i_layer,
                y_i_layer,
                z_i_layer,
                s=sizes,
                c=color,
                alpha=0.8,
                edgecolor="black",
                label=f"BVCs at {vert_angle:.2f} rad",
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

        # To avoid duplicate labels in the legend
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys())

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

    # Define preferred vertical angles and corresponding sigma values
    preferred_vertical_angles = [0, 0.2]  # Angles in radians (~0° and ~11.5°)
    sigma_d_list = [0.2, 0.2]  # Keep sigmas small
    sigma_ang_list = [0.025, 0.025]  # Keep sigmas small
    sigma_vert_list = [0.025, 0.025]  # Keep sigmas small

    # Define scaling factors for each layer
    scaling_factors = [1.0, 1.0]  # Boost the higher layer by a factor of 3

    # Initialize BVC layer with per-layer sigma values and scaling factors
    bvc_layer = BoundaryVectorCellLayer(
        max_dist=12,
        n_hd=8,
        preferred_vertical_angles=preferred_vertical_angles,
        sigma_d_list=sigma_d_list,
        sigma_ang_list=sigma_ang_list,
        sigma_vert_list=sigma_vert_list,
        scaling_factors=scaling_factors,
    )
    bvc_layer.plot_activation(points)
