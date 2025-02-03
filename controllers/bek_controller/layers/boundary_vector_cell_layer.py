# %%
from typing import Union
import numpy as np
import matplotlib.pyplot as plt
import torch


class BoundaryVectorCellLayer:
    """This layer models neurons that respond to obstacles at specific distances and angles. By default it creates
    8 head directions with 48 neurons for each head direction based on the default parameters,
    resulting in 384 total neurons.

    The original equation can be found in Ade's Disseration describing this layer in Equation 3.1 on page 28
    """

    def __init__(
        self,
        max_dist: float,
        input_dim: int,
        n_hd: int,
        sigma_ang: float,
        sigma_d: float,
        num_bvc_per_dir: int = 50,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ) -> None:
        """Initialize the boundary vector cell (BVC) layer.

        Args:
            max_dist: Max distance that the BVCs respond to. Units depend on the context of the environment.
            input_dim: Size of input vector to the BVC layer (e.g., 720 for RPLidar).
            n_hd: Number of head direction cells, each representing a preferred angular direction for the BVCs.
            sigma_ang: Standard deviation (tuning width) for the Gaussian function modeling angular tuning of BVCs (in degrees).
            sigma_d: Standard deviation (tuning width) for the Gaussian function modeling distance tuning of BVCs.
            dtype: PyTorch data type (e.g., torch.float32).
            device: The computational device to use (e.g., "cpu" or "cuda").
        """
        self.device = device
        self.dtype = dtype
        self.n_hd = n_hd

        # Create a torch tensor of lidar angles from 0 to 2π by the resolution of the lidar
        self.lidar_angles = torch.linspace(
            start=0, end=2 * torch.pi, steps=input_dim, dtype=dtype, device=device
        )

        # Create tuned distances per cell
        tuned_dist = torch.arange(
            start=0,
            end=max_dist,
            step=max_dist / num_bvc_per_dir,
            dtype=dtype,
            device=device,
        )
        N_dist = len(tuned_dist)

        # Preferred distances (d_i): shape (1, N_dist*n_hd)
        self.d_i = torch.tile(tuned_dist, (n_hd,))  # shape (N_dist*n_hd,)
        self.d_i = self.d_i.unsqueeze(0)  # shape (1, N_dist*n_hd)

        # Total number of BVC neurons
        self.num_bvc = self.d_i.numel()

        # Indices to map input LiDAR angles to BVC neurons
        hd_indices = torch.linspace(
            start=0,
            end=input_dim - 1,
            steps=n_hd,
            dtype=torch.long,
            device=device,
        )
        self.input_indices = torch.repeat_interleave(hd_indices, repeats=N_dist)
        self.input_indices = self.input_indices.unsqueeze(0)  # shape (1, N_dist*n_hd)

        # Preferred angles for each BVC: shape (1, N_dist*n_hd)
        angles = torch.linspace(
            start=0, end=2 * torch.pi, steps=input_dim, dtype=dtype, device=device
        )
        self.phi_i = angles[self.input_indices]

        # Standard deviations
        self.sigma_ang = torch.tensor(np.deg2rad(sigma_ang), dtype=dtype, device=device)
        self.sigma_d = torch.tensor(sigma_d, dtype=dtype, device=device)

        # Precompute the angular Gaussian part (1, N_dist*n_hd)
        self.angular_gaussian = torch.exp(
            -((self.lidar_angles[self.input_indices] - self.phi_i) ** 2)
            / (2 * self.sigma_ang**2)
        ) / torch.sqrt(2 * torch.pi * self.sigma_ang**2)

        self.bvc_activations = None

    def get_bvc_activation(self, distances: torch.Tensor) -> torch.Tensor:
        """Calculate the activation of BVCs based on input distances.

        Args:
            distances: 1D torch.Tensor of shape [input_dim], containing distance readings.

        Returns:
            Activations of the BVC neurons, computed as the product of Gaussian
            functions for distance and angle tuning, then normalized.
        """
        # Ensure distances is on the correct device and dtype
        distances = distances.to(device=self.device, dtype=self.dtype)

        # Compute distance Gaussian: shape (1, N_dist*n_hd)
        distance_gaussian = torch.exp(
            -((distances[self.input_indices] - self.d_i) ** 2) / (2 * self.sigma_d**2)
        ) / torch.sqrt(2 * torch.pi * self.sigma_d**2)

        # Multiply distance and angular Gaussians -> shape (1, N_dist*n_hd)
        bvc_activations = distance_gaussian * self.angular_gaussian

        # Sum across the repeated dimension, producing shape (N_dist*n_hd,)
        bvc_activations = torch.sum(bvc_activations, dim=0)

        # NOT ORIGINAL TO EQUATIONS - ADDED BY ANDREW G
        # Normalize to range [0, 0.5] by dividing by 2*max(bvc_activations)
        max_activation = torch.max(bvc_activations)
        if max_activation > 0:
            bvc_activations = bvc_activations / (2.0 * max_activation)

        self.bvc_activations = bvc_activations

        return bvc_activations

    def plot_activation(
        self,
        distances: np.ndarray,
        return_plot: bool = False,
    ) -> Union[None, plt.Figure]:
        """Plot the BVC activation on a polar plot and overlay the raw distance data.

        This function computes the BVC activations for the given distances and
        then plots each BVC's preferred angle and distance scaled by the
        neuron's activation. Also plots the boundary (distances) on the same polar axes.

        Args:
            distances: A 1D NumPy array of shape [input_dim], e.g., LiDAR distances.
            return_plot: If True, returns the plot object instead of showing it.

        Returns:
            The matplotlib Figure object if return_plot is True, otherwise None.
        """
        # Convert distances to a torch tensor and compute activations
        distances_tensor = torch.tensor(distances, dtype=self.dtype, device=self.device)
        bvc_activations = self.get_bvc_activation(distances_tensor)
        activations = bvc_activations.detach().cpu().numpy()

        # Convert class parameters to CPU NumPy for plotting
        d_i_np = self.d_i.detach().cpu().numpy()  # shape (1, N_dist*n_hd)
        phi_i_np = self.phi_i.detach().cpu().numpy()  # shape (1, N_dist*n_hd)
        angles_np = self.lidar_angles.detach().cpu().numpy()  # shape (input_dim,)

        # Create a polar plot
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, polar=True)

        # Maximum circle size (just a chosen scaling factor)
        max_circle_size = self.sigma_d.item()

        # Plot each BVC neuron on the polar plot with size and color based on activation
        for i in range(self.num_bvc):
            r = d_i_np[0, i]  # Preferred distance
            theta = phi_i_np[0, i]  # Preferred angle
            act = activations[i]

            if activations.max() > 0:
                size_scale = (act / activations.max()) * max_circle_size * 100
                color_scale = act / activations.max()
            else:
                size_scale = 0
                color_scale = 0

            color = plt.cm.viridis(color_scale)

            ax.scatter(theta, r, s=size_scale, c=[color], alpha=0.7, edgecolor="black")

        # Plot the boundary (raw distance data) in polar form
        ax.plot(angles_np, distances, "r-", linewidth=2, label="Boundary")

        # Set the radial limits (from 0 up to max preferred distance)
        ax.set_ylim(0, d_i_np.max())

        # Add a legend
        plt.legend()

        # If return_plot is True, return the figure object; otherwise, display the plot
        if return_plot:
            return fig
        else:
            plt.show()


if __name__ == "__main__":
    # Example usage with a synthetic star-shaped boundary
    n_points: int = 720
    max_r: float = 10
    min_r: float = 5
    n_star_peaks: int = 5

    distances = np.ones(n_points) * min_r  # Start with min_r
    star_interval: int = n_points // (n_star_peaks * 2)

    # Create the 'star' by alternating segments between max_r and min_r
    for i in range(0, n_star_peaks * 2, 2):
        start_idx = i * star_interval
        distances[start_idx : start_idx + star_interval] = max_r

    # Initialize BVC layer (no angles passed to plot_activation)
    bvc_layer_with_activation = BoundaryVectorCellLayer(
        max_dist=12,
        input_dim=n_points,
        n_hd=8,
        sigma_ang=90,
        sigma_d=1,
        num_bvc_per_dir=50,
    )

    # Plot BVC activation with the star-shaped boundary
    bvc_layer_with_activation.plot_activation(distances)
