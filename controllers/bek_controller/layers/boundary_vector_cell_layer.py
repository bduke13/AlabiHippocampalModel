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
    ) -> None:
        """Initialize the boundary vector cell (BVC) layer.

        This layer models neurons that respond to obstacles at specific distances and angles. It creates
        8 head directions with 48 neurons for each head direction based on the default parameters,
        resulting in 384 total neurons.

        Args:
            max_dist: Max distance that the BVCs respond to. Units depend on the context of the environment.
            input_dim: Size of input vector to the BVC layer (e.g., 720 for RPLidar).
            n_hd: Number of head direction cells, each representing a preferred angular direction for the BVCs.
            sigma_ang: Standard deviation (tuning width) for the Gaussian function modeling angular tuning of BVCs (in degrees).
            sigma_d: Standard deviation (tuning width) for the Gaussian function modeling distance tuning of BVCs.
        """
        # Compute the number of preferred distances per head direction
        N_dist = len(np.arange(0, max_dist, sigma_d / 2))

        # Preferred distances for each BVC; determines how sensitive each BVC is to specific distances.
        self.d_i = np.tile(np.arange(0, max_dist, sigma_d / 2), n_hd)[np.newaxis, :]

        # Total number of BVC neurons = 8 head directions * 48 preferred distances per head direction.
        self.num_bvc = self.d_i.size

        # Indices to map input LiDAR angles to BVC neurons
        self.input_indices = np.repeat(
            np.linspace(0, input_dim, n_hd, endpoint=False, dtype=int),
            N_dist,
        )[np.newaxis, :]

        # Preferred angles for each BVC, spaced around 360 degrees.
        self.phi_i = np.linspace(0, 2 * np.pi, input_dim)[self.input_indices]

        # Angular standard deviation for the Gaussian function (converted to radians).
        self.sigma_ang = tf.constant(np.deg2rad(sigma_ang), dtype=tf.float32)

        # Distance standard deviation for the Gaussian function.
        self.sigma_d = tf.constant(sigma_d, dtype=tf.float32)

    def get_bvc_activation(
        self, distances: np.ndarray, angles: np.ndarray
    ) -> tf.Tensor:
        """Calculate the activation of BVCs based on input distances and angles.

        Args:
            distances: Array of distance readings, representing obstacles' distances from the sensor.
            angles: Array of angles corresponding to the distance readings.

        Returns:
            Activations of the BVC neurons, computed as the product of Gaussian functions for
            distance and angle tuning.
        """
        PI = tf.constant(np.pi)

        # Compute Gaussian function for distance tuning
        distance_gaussian = tf.exp(
            -((distances[self.input_indices] - self.d_i) ** 2) / (2 * self.sigma_d**2)
        ) / tf.sqrt(2 * PI * self.sigma_d**2)

        # Compute Gaussian function for angular tuning
        angular_gaussian = tf.exp(
            -((angles[self.input_indices] - self.phi_i) ** 2) / (2 * self.sigma_ang**2)
        ) / tf.sqrt(2 * PI * self.sigma_ang**2)

        # Return the product of distance and angular Gaussian functions for BVC activation
        return tf.reduce_sum((distance_gaussian * angular_gaussian), 0)

    def plot_activation_distribution(
        self,
        distances: np.ndarray,
        angles: np.ndarray,
        return_plot: bool = False,
    ) -> Union[None, plt.Figure]:
        """Plot the distribution of BVC activations as a histogram.

        Args:
            distances: Input distances to the BVC layer (e.g., from a LiDAR).
            angles: Input angles corresponding to the distance measurements.
            return_plot: If True, returns the plot object instead of showing it.

        Returns:
            The matplotlib Figure object if return_plot is True, otherwise None.
        """
        # Get BVC activations based on distances and angles
        activations = self.get_bvc_activation(distances, angles).numpy()

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

    # Plot the distribution of activations
    bvc_layer_with_activation.plot_activation_distribution(distances, angles)
