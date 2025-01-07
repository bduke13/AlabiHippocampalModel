from numpy.core.multiarray import dtype
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from vis_3d_scan import convert_to_3D


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
        num_rows: int = 90,
        num_cols: int = 180,
        top_cutoff_percentage: float = 0.0,
        bottom_cutoff_percentage: float = 0.5,
    ) -> None:
        print(
            f"""sigma_d_list {sigma_d_list}
        sigma_ang_list {sigma_ang_list}
        sigma_vert_list {sigma_vert_list}
        num_bvc_per_dir {num_bvc_per_dir}"""
        )
        # Store input parameters
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.preferred_vertical_angles = preferred_vertical_angles

        # Precompute BVC parameters
        d_i = np.linspace(0, max_dist, num=num_bvc_per_dir)
        N_dist = len(d_i)
        phi_i = np.repeat(np.linspace(0, 2 * np.pi, n_hd, endpoint=False), N_dist)
        d_i = np.tile(d_i, n_hd)

        self.num_bvc = num_bvc_per_dir * n_hd * len(preferred_vertical_angles)

        self.d_i_all = []
        self.phi_i_all = []
        self.phi_i_vert = []
        self.sigma_d_all = []
        self.sigma_ang_all = []
        self.sigma_vert_all = []
        self.scaling_factors_all = []

        # Assume preferred_vertical_angles are in radians
        # If they are in degrees, convert: preferred_vertical_angles = np.radians(preferred_vertical_angles)
        for idx, vert_angle in enumerate(preferred_vertical_angles):
            num_neurons = len(phi_i)
            self.d_i_all.extend(d_i)
            self.phi_i_all.extend(phi_i)
            self.phi_i_vert.extend([vert_angle] * num_neurons)
            self.sigma_d_all.extend([sigma_d_list[idx]] * num_neurons)
            self.sigma_ang_all.extend([sigma_ang_list[idx]] * num_neurons)
            self.sigma_vert_all.extend([sigma_vert_list[idx]] * num_neurons)
            self.scaling_factors_all.extend([scaling_factors[idx]] * num_neurons)

        # Convert to tensors
        self.d_i = tf.constant(self.d_i_all, dtype=tf.float32)
        self.phi_i = tf.constant(self.phi_i_all, dtype=tf.float32)
        self.phi_i_vert = tf.constant(self.phi_i_vert, dtype=tf.float32)
        self.sigma_d = tf.constant(self.sigma_d_all, dtype=tf.float32)
        self.sigma_ang = tf.constant(self.sigma_ang_all, dtype=tf.float32)
        self.sigma_vert = tf.constant(self.sigma_vert_all, dtype=tf.float32)
        self.scaling_factors = tf.constant(self.scaling_factors_all, dtype=tf.float32)

        # Precompute denominators and constants
        self.two_sigma_d_squared = 2 * self.sigma_d**2
        self.two_sigma_ang_squared = 2 * self.sigma_ang**2
        self.two_sigma_vert_squared = 2 * self.sigma_vert**2
        self.PI = tf.constant(np.pi, dtype=tf.float32)

        # Generate latitude (vertical) and longitude (horizontal) angles
        pi = tf.constant(np.pi, dtype=tf.float32)
        lon_angles = tf.linspace(
            0.0, 2.0 * pi, self.num_cols
        )  # horizontal angles [0, 2*pi]
        lat_angles = tf.linspace(
            pi / 2.0, -pi / 2.0, self.num_rows
        )  # vertical angles [pi/2, -pi/2]
        lon_mesh, lat_mesh = tf.meshgrid(lon_angles, lat_angles, indexing="xy")

        # Flatten lat and lon
        self.lat_flat = tf.reshape(lat_mesh, [-1])
        self.lon_flat = tf.reshape(lon_mesh, [-1])

        # Compute cutoff indices
        self.top_idx = tf.cast(
            self.num_rows * top_cutoff_percentage * self.num_cols, tf.int32
        )
        self.bottom_idx = tf.cast(
            self.num_rows * bottom_cutoff_percentage * self.num_cols, tf.int32
        )

        # Slice lat, lon according to cutoffs
        lat_slice = self.lat_flat[self.top_idx : self.bottom_idx]
        lon_slice = self.lon_flat[self.top_idx : self.bottom_idx]

        horizontal_angles = tf.expand_dims(lon_slice, axis=1)
        vertical_angles = tf.expand_dims(lat_slice, axis=1)

        # Precompute horizontal Gaussian (using wrapped angle difference)
        horizontal_diff = tf.atan2(
            tf.sin(horizontal_angles - self.phi_i),
            tf.cos(horizontal_angles - self.phi_i),
        )
        horizontal_gaussian_precomputed = tf.exp(
            -(horizontal_diff**2) / self.two_sigma_ang_squared
        ) / tf.sqrt(2 * self.PI * self.sigma_ang**2)

        # Precompute vertical Gaussian (using direct difference, no wrapping)
        vertical_diff = vertical_angles - self.phi_i_vert
        vertical_gaussian_precomputed = tf.exp(
            -(vertical_diff**2) / self.two_sigma_vert_squared
        ) / tf.sqrt(2 * self.PI * self.sigma_vert**2)

        point_gaussian_precomputed = (
            horizontal_gaussian_precomputed * vertical_gaussian_precomputed
        )

        self.point_gaussian_precomputed = tf.constant(
            point_gaussian_precomputed, dtype=tf.float32
        )

    @tf.function(
        jit_compile=True,
        input_signature=[
            tf.TensorSpec(shape=(None, None), dtype=tf.float32),
        ],
    )
    def get_bvc_activation(self, scan_data: tf.Tensor) -> tf.Tensor:
        """
        Calculate the activation of BVCs based on the input scan data.

        Args:
            scan_data: tf.Tensor of shape (num_rows, num_cols) containing distance values.

        Returns:
            tf.Tensor: Activations of the BVC neurons (M,).
        """
        # Flatten the scan data
        dist_flat = tf.reshape(scan_data, [-1])
        dist_slice = dist_flat[self.top_idx : self.bottom_idx]

        # Expand dims
        distances = tf.expand_dims(dist_slice, axis=1)

        # Compute deltas
        delta_d = distances - self.d_i

        # Gaussian computations for distance
        distance_gaussian = tf.exp(
            -(delta_d**2) / self.two_sigma_d_squared
        ) / tf.sqrt(2 * self.PI * self.sigma_d**2)

        # Combine with precomputed horizontal and vertical Gaussians
        bvc_activations = distance_gaussian * self.point_gaussian_precomputed

        # Sum over all points
        bvc_activations = tf.reduce_sum(bvc_activations, axis=0)

        # Normalize by dividing by the maximum value
        max_activation = tf.reduce_max(bvc_activations)
        normalized_activations = bvc_activations / (max_activation * 2)
        return normalized_activations

    def plot_activation_distribution(
        self, scan_data: np.ndarray, return_plot: bool = False
    ):
        activations = self.get_bvc_activation(scan_data).numpy()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        ax1.hist(activations, bins=50, color="skyblue", edgecolor="black")
        ax1.set_title("Distribution of BVC Activations")
        ax1.set_xlabel("Activation Value")
        ax1.set_ylabel("Count")

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

    def plot_activation(self, scan_data: tf.Tensor, return_plot: bool = False):
        """
        Create a 3D radial plot of the scan data and BVCs using spherical coordinates.
        Points are plotted using (r, θ, φ) where:
        r = distance from scan_data
        θ (theta) = horizontal angle (azimuth)
        φ (phi) = vertical angle (elevation)
        """
        # Get BVC activations
        activations = self.get_bvc_activation(scan_data).numpy()
        epsilon = 1e-3
        activations = np.maximum(activations, epsilon)
        activations_normalized = activations / np.max(activations)

        # Get BVC parameters
        d_i = self.d_i.numpy()
        phi_i = self.phi_i.numpy()
        phi_i_vert = self.phi_i_vert.numpy()
        # Get the flattened data
        dist_flat = tf.reshape(scan_data, [-1])
        dist_slice = dist_flat[self.top_idx : self.bottom_idx].numpy()
        lat_slice = self.lat_flat[self.top_idx : self.bottom_idx].numpy()
        lon_slice = self.lon_flat[self.top_idx : self.bottom_idx].numpy()

        # Filter out invalid values
        valid_mask = ~(np.isnan(dist_slice) | np.isinf(dist_slice) | (dist_slice <= 0))
        dist_slice = dist_slice[valid_mask]
        lat_slice = lat_slice[valid_mask]
        lon_slice = lon_slice[valid_mask]

        if len(dist_slice) == 0:
            print("Warning: No valid data points to plot!")
            return None

        # Create the figure
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection="3d")

        # Convert spherical to cartesian coordinates
        # r = distance, theta = horizontal angle, phi = vertical angle
        x = dist_slice * np.cos(lat_slice) * np.cos(lon_slice)
        y = dist_slice * np.cos(lat_slice) * np.sin(lon_slice)
        z = dist_slice * np.sin(lat_slice)

        # Create scatter plot with distance-based coloring
        scatter = ax.scatter(x, y, z, c=dist_slice, cmap="viridis", s=10, alpha=0.6)

        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label("Distance (m)")

        # Add scanner position at origin
        ax.scatter(0, 0, 0, color="red", s=100, marker="o", label="Scanner Position")

        # Add reference sphere (wireframe)
        r = np.max(dist_slice) * 0.5  # radius of reference sphere
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x_sphere = r * np.outer(np.cos(u), np.sin(v))
        y_sphere = r * np.outer(np.sin(u), np.sin(v))
        z_sphere = r * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_wireframe(
            x_sphere, y_sphere, z_sphere, color="gray", alpha=0.2, linewidth=0.5
        )

        # Set labels and title
        ax.set_xlabel("X (meters)")
        ax.set_ylabel("Y (meters)")
        ax.set_zlabel("Z (meters)")
        ax.set_title("3D Radial Scan Visualization")

        # Make the plot cubic
        x_range = x.max() - x.min() if len(x) > 0 else 1
        y_range = y.max() - y.min() if len(y) > 0 else 1
        z_range = z.max() - z.min() if len(z) > 0 else 1

        max_range = np.array([x_range, y_range, z_range]).max() / 2.0
        mid_x = (x.max() + x.min()) * 0.5 if len(x) > 0 else 0
        mid_y = (y.max() + y.min()) * 0.5 if len(y) > 0 else 0
        mid_z = (z.max() + z.min()) * 0.5 if len(z) > 0 else 0
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        # Add BVCs with their activations
        unique_horiz_angles = np.unique(phi_i)

        # Define colors for different horizontal angles (in degrees)
        colors_dict = {}
        angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)  # 8 evenly spaced angles
        colors = ["red", "blue", "green", "purple", "orange", "cyan", "yellow", "brown"]

        for angle, color in zip(angles, colors):
            colors_dict[angle] = color

        # Sort angles so 0 degrees are plotted last (on top)
        sorted_indices = np.argsort(np.abs(unique_horiz_angles))[::-1]

        for idx in sorted_indices:
            horiz_angle = unique_horiz_angles[idx]
            mask = phi_i == horiz_angle

            # Find the closest predefined angle
            angle_diffs = np.abs(np.array(list(colors_dict.keys())) - horiz_angle)
            closest_angle = list(colors_dict.keys())[np.argmin(angle_diffs)]
            color = colors_dict[closest_angle]

            # Convert BVC spherical coordinates to cartesian
            x_bvc = d_i[mask] * np.cos(phi_i_vert[mask]) * np.cos(phi_i[mask])
            y_bvc = d_i[mask] * np.cos(phi_i_vert[mask]) * np.sin(phi_i[mask])
            z_bvc = d_i[mask] * np.sin(phi_i_vert[mask])

            activations_layer = activations_normalized[mask]
            sizes = activations_layer * 100  # Scale the sizes based on activation

            ax.scatter(
                x_bvc,
                y_bvc,
                z_bvc,
                s=sizes,
                c=color,  # using color from colors_dict
                alpha=0.4,
                edgecolor="black",
                label=f"BVCs at {np.degrees(horiz_angle):.1f}° longitude",
            )

        # Add legend
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys())

        if return_plot:
            return fig
        else:
            plt.show()

    def generate_cell_names(self, output_file="cell_names.txt"):
        phi_i_np = self.phi_i.numpy()
        phi_i_vert_np = self.phi_i_vert.numpy()
        d_i_np = self.d_i.numpy()

        with open(output_file, "w") as f:
            for i in range(self.num_bvc):
                horizontal_angle_deg = np.degrees(phi_i_np[i]) % 360
                vertical_angle_deg = np.degrees(phi_i_vert_np[i])
                distance = d_i_np[i]
                name = f"BVC_{i}_Horiz{horizontal_angle_deg:.1f}_Vert{vertical_angle_deg:.1f}_Dist{distance:.1f}"
                f.write(name + "\n")

        print(f"Cell names saved to {output_file}")


if __name__ == "__main__":
    # Load the data (example)
    vertical_boundaries = np.load("first_vertical_scan.npy")

    # Print diagnostic information
    print("Data shape:", vertical_boundaries.shape)
    print("Contains NaN:", np.any(np.isnan(vertical_boundaries)))
    print("Contains Inf:", np.any(np.isinf(vertical_boundaries)))
    print("Min value:", np.min(vertical_boundaries))
    print("Max value:", np.max(vertical_boundaries))
    print(vertical_boundaries.shape)

    preferred_vertical_angles = [0, 0.15, 0.3]  # assumed to be in radians
    sigma_d_list = [0.2, 0.2, 0.2]
    sigma_ang_list = [0.025, 0.05, 0.05]
    sigma_vert_list = [0.025, 0.1, 0.1]
    scaling_factors = [1.0, 0.5, 0.1]

    bvc_layer = BoundaryVectorCellLayer(
        max_dist=12,
        n_hd=8,
        preferred_vertical_angles=preferred_vertical_angles,
        sigma_d_list=sigma_d_list,
        sigma_ang_list=sigma_ang_list,
        sigma_vert_list=sigma_vert_list,
        scaling_factors=scaling_factors,
        num_rows=90,
        num_cols=180,
    )

    # Example usage
    bvc_layer.plot_scan_3d_radial(vertical_boundaries)
    bvc_layer.plot_activation_distribution(vertical_boundaries)
