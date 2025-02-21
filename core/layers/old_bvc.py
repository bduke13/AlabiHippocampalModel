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
        enable_multiscale: bool = False,
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

        # Preferred distances for each BVC
        self.d_i = np.tile(np.arange(0, max_dist, sigma_d / 2), n_hd)[np.newaxis, :]

        # Total number of BVC neurons
        self.num_bvc = self.d_i.size

        # Indices to map input LiDAR angles to BVC neurons
        self.input_indices = np.repeat(
            np.linspace(0, input_dim, n_hd, endpoint=False, dtype=int),
            N_dist,
        )[np.newaxis, :]

        # Preferred angles for each BVC
        self.phi_i = np.linspace(0, 2 * np.pi, input_dim)[self.input_indices]

        self.enable_multiscale = enable_multiscale
        if self.enable_multiscale:
            # Initialize tuning for small-scale and large-scale BVCs
            self.small_sigma_d = tf.constant(0.5, dtype=tf.float32)
            self.small_sigma_ang = tf.constant(np.deg2rad(90), dtype=tf.float32)
            self.large_sigma_d = tf.constant(1.0, dtype=tf.float32)
            self.large_sigma_ang = tf.constant(np.deg2rad(120), dtype=tf.float32)
        else:
            # Use single-scale tuning
            self.sigma_d = tf.constant(sigma_d, dtype=tf.float32)
            self.sigma_ang = tf.constant(np.deg2rad(sigma_ang), dtype=tf.float32)

    def get_bvc_activation(self, distances, angles, scale=None):
        """Calculate the activation of BVCs, optionally using scale-specific tuning."""
        PI = tf.constant(np.pi)
        
        if self.enable_multiscale and scale == "small":
            sigma_d = self.small_sigma_d
            sigma_ang = self.small_sigma_ang
        elif self.enable_multiscale and scale == "large":
            sigma_d = self.large_sigma_d
            sigma_ang = self.large_sigma_ang
        else:  # Default single-scale
            sigma_d = self.sigma_d
            sigma_ang = self.sigma_ang

        # Compute distance tuning
        distance_gaussian = tf.exp(
            -((distances[self.input_indices] - self.d_i) ** 2) / (2 * sigma_d**2)
        ) / tf.sqrt(2 * PI * sigma_d**2)

        # Compute angular tuning
        angular_gaussian = tf.exp(
            -((angles[self.input_indices] - self.phi_i) ** 2) / (2 * sigma_ang**2)
        ) / tf.sqrt(2 * PI * sigma_ang**2)

        # Compute BVC activations
        return tf.reduce_sum((distance_gaussian * angular_gaussian), 0)

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
    
    def plot_bvc_activation_field(self, bvc_index: int, resolution: int = 100) -> None:
        """Plot the activation field of a specific BVC.

        Args:
            bvc_index: Index of the BVC to plot.
            resolution: Resolution of the grid for plotting (number of points along distance and angle axes).
        """
        # Check if the BVC index is valid
        if bvc_index < 0 or bvc_index >= self.num_bvc:
            raise ValueError(f"BVC index must be between 0 and {self.num_bvc - 1}.")

        # Create a grid of distances and angles
        distances = np.linspace(0, self.d_i.max(), resolution)
        angles = np.linspace(0, 2 * np.pi, resolution, endpoint=False)
        distance_grid, angle_grid = np.meshgrid(distances, angles)

        # Compute the activation of the specific BVC at each point in the grid
        activations = np.zeros_like(distance_grid)
        for i in range(resolution):
            for j in range(resolution):
                # Compute the distance and angle for the current grid point
                d = distance_grid[i, j]
                theta = angle_grid[i, j]

                # Compute the distance tuning
                distance_gaussian = np.exp(
                    -((d - self.d_i[0, bvc_index]) ** 2) / (2 * self.sigma_d.numpy() ** 2)
                ) / np.sqrt(2 * np.pi * self.sigma_d.numpy() ** 2)

                # Compute the angular tuning with periodic boundary handling
                angular_gaussian = np.exp(
                    -((np.angle(np.exp(1j * (theta - self.phi_i[0, bvc_index]))) ** 2)
                    / (2 * self.sigma_ang.numpy() ** 2))
                ) / np.sqrt(2 * np.pi * self.sigma_ang.numpy() ** 2)


                # Compute the activation
                activations[i, j] = distance_gaussian * angular_gaussian

        # Normalize activations for better visualization
        activations /= np.max(activations)

        # Create a polar plot
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})

        # Plot the activation field
        c = ax.pcolormesh(angle_grid, distance_grid, activations, cmap="viridis", shading="auto")

        # Add a colorbar
        cbar = plt.colorbar(c, ax=ax)
        cbar.set_label("Activation Strength")

        # Set the radial limits dynamically based on max_dist
        ax.set_ylim(0, self.d_i.max())

        # Add a title
        ax.set_title(f"Activation Field of BVC {bvc_index}")
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

    # Plot the activation field of a specific BVC (e.g., BVC with index 10)
    bvc_layer_with_activation.plot_bvc_activation_field(bvc_index=15)