# %%
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from typing import Union
from vis_3d_scan import convert_to_3D


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


class BoundaryVectorCellLayer:
    def __init__(
        self,
        max_dist: float,
        n_hd: int,
        preferred_vertical_angles: list = [0],
        sigma_d_list: list = [0.3],
        sigma_ang_list: list = [0.025],
        sigma_vert_list: list = [0.025],
        scaling_factors: list = [1.0],
        num_bvc_per_dir: int = 50,
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
        self.d_i = np.linspace(0, max_dist, num=num_bvc_per_dir)
        N_dist = len(self.d_i)

        # Preferred horizontal angles
        self.phi_i = np.repeat(np.linspace(0, 2 * np.pi, n_hd, endpoint=False), N_dist)
        self.d_i = np.tile(self.d_i, n_hd)

        # Initialize BVC properties per vertical layer
        self.d_i_all = []
        self.phi_i_all = []
        self.phi_i_vert = []
        self.sigma_d_all = []
        self.sigma_ang_all = []
        self.sigma_vert_all = []
        self.scaling_factors_all = []

        self.num_bvc = num_bvc_per_dir * n_hd * len(preferred_vertical_angles)

        for idx, vert_angle in enumerate(preferred_vertical_angles):
            num_neurons = len(self.phi_i)
            self.d_i_all.extend(self.d_i)
            self.phi_i_all.extend(self.phi_i)
            self.phi_i_vert.extend([vert_angle] * num_neurons)
            self.sigma_d_all.extend([sigma_d_list[idx]] * num_neurons)
            self.sigma_ang_all.extend([sigma_ang_list[idx]] * num_neurons)
            self.sigma_vert_all.extend([sigma_vert_list[idx]] * num_neurons)
            self.scaling_factors_all.extend([scaling_factors[idx]] * num_neurons)

        # Convert to TensorFlow tensors for GPU processing
        self.d_i = tf.constant(self.d_i_all, dtype=tf.float32)  # (M,)
        self.phi_i = tf.constant(self.phi_i_all, dtype=tf.float32)  # (M,)
        self.phi_i_vert = tf.constant(self.phi_i_vert, dtype=tf.float32)  # (M,)
        self.sigma_d = tf.constant(self.sigma_d_all, dtype=tf.float32)  # (M,)
        self.sigma_ang = tf.constant(self.sigma_ang_all, dtype=tf.float32)  # (M,)
        self.sigma_vert = tf.constant(self.sigma_vert_all, dtype=tf.float32)  # (M,)
        self.scaling_factors = tf.constant(self.scaling_factors_all, dtype=tf.float32)

        # Precompute denominators
        self.two_sigma_d_squared = 2 * self.sigma_d**2  # (M,)
        self.two_sigma_ang_squared = 2 * self.sigma_ang**2  # (M,)
        self.two_sigma_vert_squared = 2 * self.sigma_vert**2  # (M,)

    @tf.function()
    def get_scan_points(
        self,
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
            bottom_idx = tf.cast(
                num_rows * bottom_cutoff_percentage * num_cols, tf.int32
            )
            points = points[top_idx:bottom_idx]

        return points

    @tf.function(jit_compile=True)  # Optimize with XLA
    def get_bvc_activation(self, points: tf.Tensor) -> tf.Tensor:
        """Calculate the activation of BVCs based on input points.

        Args:
            points: tf.Tensor of shape (N, 3), containing:
                [:, 0]: latitude angles in radians
                [:, 1]: longitude angles in radians
                [:, 2]: distance values

        Returns:
            tf.Tensor: Activations of the BVC neurons.
        """

        points = self.get_scan_points(points)
        distances = points[:, 2]  # (N,)
        horizontal_angles = points[:, 1]  # (N,)
        vertical_angles = points[:, 0]  # (N,)

        # Expand dimensions for broadcasting
        distances = tf.expand_dims(distances, axis=1)  # (N, 1)
        horizontal_angles = tf.expand_dims(horizontal_angles, axis=1)  # (N, 1)
        vertical_angles = tf.expand_dims(vertical_angles, axis=1)  # (N, 1)

        # Compute deltas
        delta_d = distances - self.d_i  # (N, M)
        delta_ang = tf.atan2(
            tf.sin(horizontal_angles - self.phi_i),
            tf.cos(horizontal_angles - self.phi_i),
        )  # (N, M)
        delta_vert = tf.atan2(
            tf.sin(vertical_angles - self.phi_i_vert),
            tf.cos(vertical_angles - self.phi_i_vert),
        )  # (N, M)

        # Gaussian computations
        PI = tf.constant(np.pi, dtype=tf.float32)
        distance_gaussian = tf.exp(
            -(delta_d**2) / self.two_sigma_d_squared
        ) / tf.sqrt(2 * PI * self.sigma_d**2)
        horizontal_gaussian = tf.exp(
            -(delta_ang**2) / self.two_sigma_ang_squared
        ) / tf.sqrt(2 * PI * self.sigma_ang**2)
        vertical_gaussian = tf.exp(
            -(delta_vert**2) / self.two_sigma_vert_squared
        ) / tf.sqrt(2 * PI * self.sigma_vert**2)

        # Combine Gaussians and sum activations
        activations = (
            distance_gaussian * horizontal_gaussian * vertical_gaussian
        )  # (N, M)
        activations = tf.reduce_sum(activations, axis=0)  # (M,)

        # Apply scaling factors and normalize
        activations *= self.scaling_factors
        activations /= tf.cast(
            tf.shape(points)[0], tf.float32
        )  # Normalize by input points
        return activations

    def plot_activation_distribution(
        self,
        distances: np.ndarray,
        return_plot: bool = False,
    ) -> Union[None, plt.Figure]:
        """Plot the distribution of BVC activations as a histogram.

        Args:
            distances: Input distances to the BVC layer (e.g., from a LiDAR).
            return_plot: If True, returns the plot object instead of showing it.

        Returns:
            The matplotlib Figure object if return_plot is True, otherwise None.
        """
        # Get BVC activations based on distances and angles
        activations = self.get_bvc_activation(distances).numpy()

        # Create histogram plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Plot histogram
        ax1.hist(activations, bins=50, color="skyblue", edgecolor="black")
        ax1.set_title("Distribution of BVC Activations")
        ax1.set_xlabel("Activation Value")
        ax1.set_ylabel("Count")

        # Plot sorted activations to see the distribution curve
        sorted_activations = np.sort(activations)
        ax2.plot(sorted_activations, "b-", linewidth=2)
        ax2.set_title("Sorted BVC Activations")
        ax2.set_xlabel("Neuron Index (sorted by activation)")
        ax2.set_ylabel("Activation Value")

        plt.tight_layout()

        if return_plot:
            return fig
        else:
            plt.show()

    def plot_activation(
        self,
        points: tf.Tensor,
        return_plot: bool = False,
    ) -> Union[None, plt.Figure]:
        """Plot the BVC activations in 3D space.

        Args:
            points: tf.Tensor of shape (N, 3), containing:
                [:, 0]: latitude angles in radians
                [:, 1]: longitude angles in radians
                [:, 2]: distance values
            return_plot: If True, returns the matplotlib figure object.

        Returns:
            None or plt.Figure
        """
        # Get BVC activations
        activations = self.get_bvc_activation(points).numpy()

        # Compute BVC positions
        d_i = self.d_i.numpy()
        phi_i = self.phi_i.numpy()
        phi_i_vert = self.phi_i_vert.numpy()

        # Convert points to x, y, z coordinates
        xyz_coords = convert_to_3D(points).numpy()

        x_i = d_i * np.cos(phi_i) * np.cos(phi_i_vert)
        y_i = d_i * np.sin(phi_i) * np.cos(phi_i_vert)
        z_i = d_i * np.sin(phi_i_vert)

        # For non-active BVCs, set a small epsilon activation
        epsilon = 1e-6
        activations = np.maximum(activations, epsilon)

        # Normalize activations for plotting
        activations_normalized = activations / np.max(activations)

        # Create the 3D plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection="3d")

        # Plot the environment points
        ax.scatter(
            xyz_coords[:, 0],
            xyz_coords[:, 1],
            xyz_coords[:, 2],
            c="dodgerblue",
            s=1,
            alpha=0.6,
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
                alpha=0.4,
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

    def generate_cell_names(self, output_file="cell_names.txt"):
        """Generate a list of names for each BVC neuron and save to a file.

        Args:
            output_file: The file to save the cell names.
        """
        self.cell_names = []
        with open(output_file, "w") as f:
            for i in range(self.num_bvc):
                horizontal_angle_deg = np.degrees(self.phi_i.numpy()[i]) % 360
                vertical_angle_deg = np.degrees(self.phi_i_vert.numpy()[i])
                distance = self.d_i.numpy()[i]
                name = f"BVC_{i}_Horiz{horizontal_angle_deg:.1f}_Vert{vertical_angle_deg:.1f}_Dist{distance:.1f}"
                self.cell_names.append(name)
                f.write(name + "\n")
        print(f"Cell names saved to {output_file}")


if __name__ == "__main__":
    # Set mixed precision policy for better GPU performance
    tf.keras.mixed_precision.set_global_policy("mixed_float16")

    # Load the data
    vertical_boundaries = np.load("first_vertical_scan.npy")

    # Reshape the data from (259200,) to (360, 720)
    reshaped_data = vertical_boundaries.reshape(360, 720)

    preferred_vertical_angles = [0, 0.15, 0.3]
    sigma_d_list = [0.2, 0.2, 0.2]  # sigma_d for each layer
    sigma_ang_list = [0.025, 0.05, 0.05]  # sigma_ang for each layer
    sigma_vert_list = [0.025, 0.1, 0.1]  # sigma_vert for each layer
    scaling_factors = [1.0, 0.5, 0.1]

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

    # Plot the BVC activations
    bvc_layer.plot_activation(points)
    bvc_layer.plot_activation_distribution(points)
