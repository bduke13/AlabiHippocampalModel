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
        # Store input parameters
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.preferred_vertical_angles = preferred_vertical_angles

        # Precompute angles and flatten once
        pi = tf.constant(np.pi, dtype=tf.float32)
        lon_angles = tf.linspace(0.0, 2.0 * pi, self.num_cols)
        lat_angles = tf.linspace(pi / 2.0, -pi / 2.0, self.num_rows)
        lon_mesh, lat_mesh = tf.meshgrid(lon_angles, lat_angles, indexing="xy")

        # Flatten lat and lon just once and store them as attributes
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

        # Set horizontal and vertical angles as constants
        horizontal_angles = tf.expand_dims(lon_slice, axis=1)
        vertical_angles = tf.expand_dims(lat_slice, axis=1)

        self.horizontal_angles = tf.constant(horizontal_angles, dtype=tf.float32)
        self.vertical_angles = tf.constant(vertical_angles, dtype=tf.float32)

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

    @tf.function(
        jit_compile=True,
        input_signature=[
            tf.TensorSpec(shape=(90, 180), dtype=tf.float32),
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
        delta_ang = tf.atan2(
            tf.sin(self.horizontal_angles - self.phi_i),
            tf.cos(self.horizontal_angles - self.phi_i),
        )
        delta_vert = tf.atan2(
            tf.sin(self.vertical_angles - self.phi_i_vert),
            tf.cos(self.vertical_angles - self.phi_i_vert),
        )

        # Gaussian computations
        distance_gaussian = tf.exp(
            -(delta_d**2) / self.two_sigma_d_squared
        ) / tf.sqrt(2 * self.PI * self.sigma_d**2)
        horizontal_gaussian = tf.exp(
            -(delta_ang**2) / self.two_sigma_ang_squared
        ) / tf.sqrt(2 * self.PI * self.sigma_ang**2)
        vertical_gaussian = tf.exp(
            -(delta_vert**2) / self.two_sigma_vert_squared
        ) / tf.sqrt(2 * self.PI * self.sigma_vert**2)

        # Combine Gaussians and sum
        activations = distance_gaussian * horizontal_gaussian * vertical_gaussian
        activations = tf.reduce_sum(activations, axis=0)

        # Apply scaling and normalize
        activations *= self.scaling_factors
        activations /= tf.cast(tf.size(dist_slice), tf.float32)
        return activations

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
        # Get activations for current cutoff
        activations = self.get_bvc_activation(scan_data).numpy()

        # Flatten and slice lat/lon/dist just like in get_bvc_activation
        dist_flat = tf.reshape(scan_data, [-1])
        dist_slice = dist_flat[self.top_idx : self.bottom_idx].numpy()

        lat_slice = self.lat_flat[self.top_idx : self.bottom_idx].numpy()
        lon_slice = self.lon_flat[self.top_idx : self.bottom_idx].numpy()

        points = np.stack([lat_slice, lon_slice, dist_slice], axis=1)
        points_tf = tf.constant(points, dtype=tf.float32)
        xyz_coords = convert_to_3D(points_tf).numpy()

        d_i = self.d_i.numpy()
        phi_i = self.phi_i.numpy()
        phi_i_vert = self.phi_i_vert.numpy()

        x_i = d_i * np.cos(phi_i) * np.cos(phi_i_vert)
        y_i = d_i * np.sin(phi_i) * np.cos(phi_i_vert)
        z_i = d_i * np.sin(phi_i_vert)

        epsilon = 1e-6
        activations = np.maximum(activations, epsilon)
        activations_normalized = activations / np.max(activations)

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection="3d")

        ax.scatter(
            xyz_coords[:, 0],
            xyz_coords[:, 1],
            xyz_coords[:, 2],
            c="dodgerblue",
            s=1,
            alpha=0.6,
            label="Environment",
        )

        unique_vert_angles = np.unique(phi_i_vert)
        colors_list = ["red", "green", "orange", "purple", "cyan"]
        for idx, vert_angle in enumerate(unique_vert_angles):
            mask = phi_i_vert == vert_angle
            x_i_layer = x_i[mask]
            y_i_layer = y_i[mask]
            z_i_layer = z_i[mask]
            activations_layer = activations_normalized[mask]
            sizes = activations_layer * 100

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

        ax.scatter(
            0, 0, 0, color="red", s=100, marker="o", label="Scanner Position", zorder=5
        )

        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys())

        ax.set_xlabel("X (meters)")
        ax.set_ylabel("Y (meters)")
        ax.set_zlabel("Z (meters)")
        ax.set_box_aspect([1, 1, 1])
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_zlim(-10, 10)

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
    reshaped_data = vertical_boundaries.reshape(360, 720)

    preferred_vertical_angles = [0, 0.15, 0.3]
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
        num_rows=360,
        num_cols=720,
    )

    # Example usage
    bvc_layer.plot_activation(reshaped_data)
    bvc_layer.plot_activation_distribution(reshaped_data)
