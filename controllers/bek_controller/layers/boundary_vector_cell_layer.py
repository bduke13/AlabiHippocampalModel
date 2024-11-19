from typing import Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


class BoundaryVectorCellLayer:
    def __init__(
        self,
        max_dist: float,
        input_shape: Tuple[int, int],  # (height, width)
        n_hd: int,
        n_elevations: int,
        sigma_ang: float,
        sigma_elev: float,
        sigma_d: float,
    ) -> None:
        """Initialize the boundary vector cell (BVC) layer.

        Args:
            max_dist: Max distance that the BVCs respond to.
            input_shape: Tuple (height, width) representing the input data dimensions.
            n_hd: Number of head direction cells (azimuth divisions).
            n_elevations: Number of elevation divisions for BVCs.
            sigma_ang: Standard deviation for angular tuning (in degrees).
            sigma_elev: Standard deviation for elevation angle tuning (in degrees).
            sigma_d: Standard deviation for distance tuning.
        """
        height, width = input_shape

        # Preferred distances
        preferred_distances = np.arange(0, max_dist, sigma_d / 2)
        N_dist = len(preferred_distances)

        # Preferred azimuth angles (head directions)
        preferred_azimuths = np.linspace(0, 2 * np.pi, n_hd, endpoint=False)

        # Preferred elevation angles
        preferred_elevations = np.linspace(0, np.deg2rad(30), n_elevations)

        # Create a grid of preferred distances, azimuths, and elevations
        d_i_grid, phi_i_grid, theta_i_grid = np.meshgrid(
            preferred_distances, preferred_azimuths, preferred_elevations, indexing='ij'
        )

        # Flatten the grids to create arrays of preferred distances and angles
        self.d_i = d_i_grid.flatten()[np.newaxis, :]
        self.phi_i = phi_i_grid.flatten()[np.newaxis, :]
        self.theta_i = theta_i_grid.flatten()[np.newaxis, :]

        # Total number of BVC neurons
        self.num_bvc = self.d_i.size

        # Convert standard deviations to radians
        self.sigma_ang = tf.constant(np.deg2rad(sigma_ang), dtype=tf.float32)
        self.sigma_elev = tf.constant(np.deg2rad(sigma_elev), dtype=tf.float32)
        self.sigma_d = tf.constant(sigma_d, dtype=tf.float32)

    def get_bvc_activation(
        self, distances: np.ndarray, azimuth_angles: np.ndarray, elevation_angles: np.ndarray
    ) -> tf.Tensor:
        """Calculate the activation of BVCs based on input distances and angles.

        Args:
            distances: 2D array of distance readings, shape (height, width).
            azimuth_angles: 2D array of azimuth angles, shape (height, width).
            elevation_angles: 2D array of elevation angles, shape (height, width).

        Returns:
            Activations of the BVC neurons, computed as the sum over all pixels.
        """
        # Flatten the input arrays and cast to tf.float32
        distances_flat = tf.convert_to_tensor(distances.flatten(), dtype=tf.float32)  # Shape: (num_pixels,)
        azimuth_flat = tf.convert_to_tensor(azimuth_angles.flatten(), dtype=tf.float32)  # Shape: (num_pixels,)
        elevation_flat = tf.convert_to_tensor(elevation_angles.flatten(), dtype=tf.float32)  # Shape: (num_pixels,)

        # Compute differences between input and preferred distances and angles
        distance_diff = distances_flat[:, tf.newaxis] - tf.cast(self.d_i, tf.float32)  # Shape: (num_pixels, num_bvc)
        azimuth_diff = tf.math.mod(
            azimuth_flat[:, tf.newaxis] - tf.cast(self.phi_i, tf.float32) + np.pi, 2 * np.pi
        ) - np.pi
        elevation_diff = elevation_flat[:, tf.newaxis] - tf.cast(self.theta_i, tf.float32)

        # Compute Gaussian activations
        distance_gaussian = tf.exp(- (distance_diff ** 2) / (2 * self.sigma_d ** 2))
        azimuth_gaussian = tf.exp(- (azimuth_diff ** 2) / (2 * self.sigma_ang ** 2))
        elevation_gaussian = tf.exp(- (elevation_diff ** 2) / (2 * self.sigma_elev ** 2))

        # Compute activation for each BVC neuron
        activation = tf.reduce_sum(
            distance_gaussian * azimuth_gaussian * elevation_gaussian, axis=0
        )  # Shape: (num_bvc,)

        return activation


    def plot_activation(
        self,
        distances: np.ndarray,
        angles: np.ndarray,
        return_plot: bool = False,
    ) -> Union[None, plt.Figure]:
        """Plot the BVC activation on a polar plot and overlay the raw data.

        This function will plot each BVC's activation and the synthetic boundary.

        Args:
            distances: Input distances to the BVC layer (e.g., from a LiDAR).
            angles: Input angles corresponding to the distance measurements.
            return_plot: If True, returns the plot object instead of showing it.

        Returns:
            The matplotlib Figure object if return_plot is True, otherwise None.
        """
        # Get BVC activations based on distances and angles
        activations = self.get_bvc_activation(distances, angles).numpy()

        # Create a polar plot
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, polar=True)

        # Maximum circle size should be half of sigma_d
        max_circle_size = (
            self.sigma_d.numpy()
        )  # Convert Tensor to a NumPy value if necessary

        # Plot each BVC neuron on the polar plot with size and color based on activation level
        for i in range(self.num_bvc):
            r = self.d_i[0, i]  # Preferred distance
            theta = self.phi_i[0, i]  # Preferred angle
            # Scale size of the circle to be proportional to half of sigma_d, using activations
            size = (
                (activations[i] / np.max(activations)) * max_circle_size * 100
            )  # Adjust scaling factor
            color = plt.cm.viridis(
                activations[i] / np.max(activations)
            )  # Normalize color scale

            # Plot the neuron with activation
            ax.scatter(theta, r, s=size, c=[color], alpha=0.7, edgecolor="black")

        # Plot the boundary for reference
        ax.plot(angles, distances, "r-", linewidth=2, label="Boundary")

        # Set the radial limits dynamically based on max_dist
        ax.set_ylim(
            0, self.d_i.max()
        )  # Ensures plot scales with BVC preferred distances

        # Add a legend and show the plot
        plt.legend()

        # If return_plot is True, return the figure object, otherwise show the plot
        if return_plot:
            return fig
        else:
            plt.show()


if __name__ == "__main__":
    # Generate a synthetic star-shaped boundary (without defaults)
    n_points: int = 720
    max_r: float = 10
    min_r: float = 5
    n_star_peaks: int = 5

    angles: np.ndarray = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    distances: np.ndarray = np.ones(n_points) * min_r  # Start with min_r

    # Create star peaks by alternating between max_r and min_r
    star_interval: int = n_points // (
        n_star_peaks * 2
    )  # Each peak consists of a far and near point
    for i in range(0, n_star_peaks * 2, 2):
        start_idx = i * star_interval
        distances[start_idx : start_idx + star_interval] = max_r

    # Initialize BVC layer
    bvc_layer_with_activation = BoundaryVectorCellLayer(
        max_dist=12, input_dim=720, n_hd=8, sigma_ang=90, sigma_d=0.5
    )

    # Plot BVC activation with the star-shaped boundary
    bvc_layer_with_activation.plot_activation(distances, angles)
