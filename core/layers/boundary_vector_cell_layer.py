from typing import Union
import numpy as np
import matplotlib.pyplot as plt
import torch

class BoundaryVectorCellLayer:
    """Implementation of the BVC model based on the given equation from the paper.
    Each BVC is tuned to a preferred distance and angle, with activations modeled
    as the sum of the product of two Gaussian receptive fields (distance and angle)
    for each lidar point compared to the BVC's tuned parameters.

    The original equation can be found in Ade's Dissertation describing this layer in Equation 3.1 on page 43.
    """

    def __init__(
        self,
        n_res: int,  # LiDAR resolution (e.g., 720)
        n_hd: int,  # Number of head directions
        sigma_theta: float,  # Standard deviation for angular tuning (degrees)
        sigma_r: float,  # Standard deviation for distance tuning
        max_dist: float,  # Maximum response distance
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.n_res = n_res
        self.n_hd = n_hd
        self.sigma_theta = torch.tensor(np.deg2rad(sigma_theta), dtype=dtype, device=device)
        self.sigma_r = torch.tensor(sigma_r, dtype=dtype, device=device)
        self.max_dist = max_dist
        self.dtype = dtype
        self.device = device

        # Compute the preferred distances and angles
        n_dist = len(np.arange(0, max_dist, sigma_r / 2))

        # Preferred distances for each BVC
        self.d_i = torch.tensor(
            np.tile(np.arange(0, max_dist, sigma_r / 2), n_hd), dtype=dtype, device=device
        ).unsqueeze(0)

        # Total number of BVC neurons
        self.num_bvc = self.d_i.size(1)

        # Preferred angles for each BVC
        self.input_indices = torch.tensor(
            np.repeat(np.linspace(0, n_res, n_hd, endpoint=False, dtype=int), n_dist),
            dtype=torch.long,
            device=device
        ).unsqueeze(0)

        self.phi_i = torch.tensor(
            np.linspace(0, 2 * np.pi, n_res)[self.input_indices.cpu().numpy()],
            dtype=dtype,
            device=device
        )

    def get_bvc_activation(self, distances: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        """Calculate the activation of BVCs."""
        PI = torch.tensor(np.pi, dtype=self.dtype, device=self.device)

        # Distance tuning
        distance_gaussian = torch.exp(
            -((distances[self.input_indices] - self.d_i) ** 2) / (2 * self.sigma_r**2)
        ) / torch.sqrt(2 * PI * self.sigma_r**2)

        # Angular tuning with periodic boundary handling
        angular_diff = torch.remainder(angles[self.input_indices] - self.phi_i + PI, 2 * PI) - PI
        angular_gaussian = torch.exp(
            -(angular_diff**2) / (2 * self.sigma_theta**2)
        ) / torch.sqrt(2 * PI * self.sigma_theta**2)

        # Compute BVC activations
        return torch.sum(distance_gaussian * angular_gaussian, dim=0)

    def plot_activation(
        self,
        distances: np.ndarray,
        angles: np.ndarray,
        return_plot: bool = False,
    ) -> Union[None, plt.Figure]:
        """Plot the BVC activation on a polar plot and overlay the raw data."""
        distances = torch.tensor(distances, dtype=self.dtype, device=self.device)
        angles = torch.tensor(angles, dtype=self.dtype, device=self.device)

        # Get BVC activations based on distances and angles
        activations = self.get_bvc_activation(distances, angles).cpu().numpy()

        # Create a polar plot
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, polar=True)

        # Maximum circle size should be half of sigma_r
        max_circle_size = self.sigma_r.item()

        # Plot each BVC neuron on the polar plot with size and color based on activation level
        for i in range(self.num_bvc):
            r = self.d_i[0, i].item()  # Preferred distance
            theta = self.phi_i[0, i].item()  # Preferred angle

            size = (activations[i] / np.max(activations)) * max_circle_size * 100
            color = plt.cm.viridis(activations[i] / np.max(activations))

            # Plot the neuron with activation
            ax.scatter(theta, r, s=size, c=[color], alpha=0.7, edgecolor="black")

        # Plot the boundary for reference
        ax.plot(angles.cpu().numpy(), distances.cpu().numpy(), "r-", linewidth=2, label="Boundary")

        # Set the radial limits
        ax.set_ylim(0, self.max_dist)
        plt.legend()

        if return_plot:
            return fig
        else:
            plt.show()


if __name__ == "__main__":
    n_points = 720  # LiDAR resolution
    max_r = 10  # Half the side length of the square (since the square is 20x20)

    # Create angles for the LiDAR
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)

    # Calculate distances to the edges of a 20x20 square centered at the origin
    distances = np.zeros(n_points)
    for i, angle in enumerate(angles):
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)

        # Avoid division by zero and calculate distance to the square boundary
        if abs(cos_angle) > 1e-6 and abs(sin_angle) > 1e-6:
            distances[i] = min(max_r / abs(cos_angle), max_r / abs(sin_angle))
        elif abs(cos_angle) <= 1e-6:
            distances[i] = max_r / abs(sin_angle)
        else:
            distances[i] = max_r / abs(cos_angle)

    # Initialize the BVC layer
    bvc_layer = BoundaryVectorCellLayer(
        n_res=720, n_hd=8, sigma_r=3, sigma_theta=90, max_dist=20, device="cuda"
    )

    # Plot BVC activation with the square-shaped boundary
    bvc_layer.plot_activation(distances, angles)
