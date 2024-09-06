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
    ):
        """
        Initializes the boundary vector cell (BVC) layer.

        Parameters:
        max_dist (float): Max distance that the BVCs respond to. Units depend on the context of the environment.
        input_dim (int): Size of input vector to the BVC layer (e.g., 720 for RPLidar).
        n_hd (int): Number of head direction cells, each representing a preferred angular direction for the BVCs.
        sigma_ang (float): Standard deviation (tuning width) for the Gaussian function modeling angular tuning of BVCs (in degrees).
        sigma_d (float): Standard deviation (tuning width) for the Gaussian function modeling distance tuning of BVCs.

        This layer models neurons that respond to obstacles at specific distances and angles. It creates
        8 head directions with 48 neurons for each head direction based on the default parameters,
        resulting in 384 total neurons.
        """
        # Preferred distances for each BVC; determines how sensitive each BVC is to specific distances.
        self.d_i = np.tile(np.arange(0, max_dist, sigma_d / 2), n_hd)[np.newaxis, :]

        # Total number of BVC neurons = 8 head directions * 48 preferred distances per head direction.
        self.num_distances = self.d_i.size

        # Indices to map input LiDAR angles to BVC neurons
        self.input_indices = np.repeat(
            np.linspace(0, input_dim, n_hd, endpoint=False, dtype=int),
            max_dist / (sigma_d / 2),
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
        """
        Calculates the activation of BVCs based on input distances and angles (e.g., from a LiDAR).

        Parameters:
        distances (np.ndarray): Array of distance readings, representing obstacles' distances from the sensor.
        angles (np.ndarray): Array of angles corresponding to the distance readings.

        Returns:
        tf.Tensor: Activations of the BVC neurons, computed as the product of Gaussian functions for
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

    def plot_activation_with_boundary(
        self,
        distances: np.ndarray,
        angles: np.ndarray,
        star_distances: np.ndarray,
        star_angles: np.ndarray,
        return_plot: bool = False,
    ):
        """
        Plots the BVC activation on a polar plot and overlays a star-shaped boundary for visualization.

        Parameters:
        distances (np.ndarray): Input distances to the BVC layer (e.g., from a LiDAR).
        angles (np.ndarray): Input angles corresponding to the distance measurements.
        star_distances (np.ndarray): Distances representing the star-shaped boundary.
        star_angles (np.ndarray): Angles representing the star-shaped boundary.
        return_plot (bool): If True, returns the plot object instead of showing it.

        This function will plot each BVC's activation and the synthetic boundary.
        """
        # Get BVC activations based on distances and angles
        activations = self.get_bvc_activation(distances, angles).numpy()

        # Create a polar plot
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, polar=True)

        # Plot each BVC neuron on the polar plot with size and color based on activation level
        for i in range(self.num_distances):
            r = self.d_i[0, i]  # Preferred distance
            theta = self.phi_i[0, i]  # Preferred angle
            size = activations[i] * 300  # Scale size for visibility
            color = plt.cm.viridis(activations[i] / 0.179)  # Normalize color scale

            # Plot the neuron with activation
            ax.scatter(theta, r, s=size, c=[color], alpha=0.7, edgecolor="black")

        # Plot the star boundary for reference
        ax.plot(star_angles, star_distances, "r-", linewidth=2, label="Star Boundary")

        ax.set_ylim(0, 12)
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

    # Generate star-shaped boundary angles
    star_distances, star_angles = distances, angles

    # Initialize BVC layer
    bvc_layer_with_activation = BoundaryVectorCellLayer(
        max_dist=12, input_dim=720, n_hd=8, sigma_ang=90, sigma_d=0.5
    )

    # Plot BVC activation with the star-shaped boundary
    bvc_layer_with_activation.plot_activation_with_boundary(
        distances, angles, star_distances, star_angles
    )
