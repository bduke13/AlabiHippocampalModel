import torch
import numpy as np
import matplotlib.pyplot as plt


class BoundaryVectorCellLayer:
    """Implementation of the BVC model based on the given equation from the paper.
    Each BVC is tuned to a preferred distance and angle, with activations modeled
    as the sum of the product of two Gaussian receptive fields (distance and angle)
    for each lidar point compared to the BVC's tuned parameters.

    The original equation can be found in Ade's Disseration describing this layer in Equation 3.1 on page 43.
    """

    def __init__(
        self,
        n_res: int,  # LiDAR resolution (e.g., 720)
        n_hd: int,  # Number of head directions
        sigma_theta: float,  # Standard deviation for angular tuning (degrees)
        sigma_r: float,  # Standard deviation for distance tuning
        max_dist: float,  # Maximum response distance
        num_bvc_per_dir: int = 50,  # Number of BVCs per head direction
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ) -> None:
        """Initialize the boundary vector cell (BVC) layer.

        Args:
            max_dist: Max distance that the BVCs respond to. Units depend on the context of the environment.
            n_res: Size of input vector to the BVC layer (e.g., 720 for RPLidar).
            n_hd: Number of head direction cells, each representing a preferred angular direction for the BVCs.
            sigma_theta: Standard deviation (tuning width) for the Gaussian function modeling angular tuning of BVCs (in degrees).
            sigma_r: Standard deviation (tuning width) for the Gaussian function modeling distance tuning of BVCs.
            dtype: PyTorch data type (e.g., torch.float32).
            device: The computational device to use (e.g., "cpu" or "cuda").
        """
        self.device = device
        self.dtype = dtype
        self.n_res = n_res

        # Convert sigma_theta from degrees to radians
        self.sigma_theta = torch.tensor(
            np.deg2rad(sigma_theta), dtype=dtype, device=device
        )
        self.sigma_r = torch.tensor(sigma_r, dtype=dtype, device=device)

        # Create LiDAR angles: 0 to 2π in n_res steps
        self.lidar_angles = torch.linspace(
            0, 2 * torch.pi, steps=n_res, dtype=dtype, device=device
        )

        # Create preferred distances per BVC
        tuned_dist = torch.linspace(
            0, max_dist, steps=num_bvc_per_dir, dtype=dtype, device=device
        )
        N_dist = len(tuned_dist)

        # Generate preferred angles (head directions)
        preferred_angles = torch.linspace(
            0, 2 * torch.pi, steps=n_hd, dtype=dtype, device=device
        )

        # Tile distances and angles for matrix computation
        self.d_i = tuned_dist.repeat(n_hd).unsqueeze(0)  # (1, N_dist * n_hd)
        self.phi_i = preferred_angles.repeat_interleave(N_dist).unsqueeze(
            0
        )  # (1, N_dist * n_hd)

        self.num_bvc = self.d_i.numel()

        # Compute Angular Gaussian Matrix (n_res, n_res)
        lidar_angles_expanded = self.lidar_angles.unsqueeze(0)  # (1, n_res)
        phi_i_expanded = self.phi_i.T  # (n_res, 1)
        self.angular_gaussian_matrix = torch.exp(
            -((lidar_angles_expanded - phi_i_expanded) ** 2) / (2 * self.sigma_theta**2)
        ) / torch.sqrt(2 * torch.pi * self.sigma_theta**2)

        self.bvc_activations = None

    def get_bvc_activation(self, distances: torch.Tensor) -> torch.Tensor:
        """Compute the activation of the BVC neurons given LiDAR distance readings."""
        # Expand distance tensor for broadcasting
        distances_expanded = distances.unsqueeze(0)  # (1, n_res)
        d_i_expanded = self.d_i.T  # (n_res, 1)

        # Compute Distance Gaussian Matrix (n_res, n_res)
        distance_gaussian_matrix = torch.exp(
            -((distances_expanded - d_i_expanded) ** 2) / (2 * self.sigma_r**2)
        ) / torch.sqrt(2 * torch.pi * self.sigma_r**2)

        # Element-wise multiplication and sum along angle axis
        bvc_activations = torch.sum(
            self.angular_gaussian_matrix * distance_gaussian_matrix, dim=1
        )

        # **Normalize activations to [0, 0.5]**
        max_activation = torch.max(bvc_activations)
        if max_activation > 0:
            bvc_activations /= 2.0 * max_activation

        self.bvc_activations = bvc_activations
        return bvc_activations

    def plot_activation(self, distances: np.ndarray) -> None:
        """Visualizes the BVC activations in a polar plot and overlays the boundary (LiDAR readings)."""
        distances_tensor = torch.tensor(distances, dtype=self.dtype, device=self.device)
        activations = self.get_bvc_activation(distances_tensor).detach().cpu().numpy()

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})

        # Scatter plot for BVC activations (Color and Size reflect activation strength)
        ax.scatter(
            self.phi_i.cpu().numpy(),
            self.d_i.cpu().numpy(),
            c=activations,
            cmap="viridis",
            s=activations * 50,
            alpha=0.75,
        )

        # Overlay boundary (LiDAR readings)
        angles_np = self.lidar_angles.detach().cpu().numpy()
        ax.plot(
            angles_np, distances, "r-", linewidth=2, label="Boundary (LiDAR readings)"
        )

        ax.set_title("BVC Activation Heatmap with Boundary")
        ax.legend()
        plt.show()


# Example Usage
if __name__ == "__main__":
    n_points = 720  # LiDAR resolution
    max_r = 10
    min_r = 5
    n_star_peaks = 5

    # Create star-shaped LiDAR distance pattern
    distances = np.ones(n_points) * min_r
    star_interval = n_points // (n_star_peaks * 2)
    for i in range(0, n_star_peaks * 2, 2):
        start_idx = i * star_interval
        distances[start_idx : start_idx + star_interval] = max_r

    # Initialize BVC layer
    bvc_layer = BoundaryVectorCellLayer(
        max_dist=12,
        n_res=n_points,
        n_hd=8,
        sigma_theta=90,  # Angular tuning width
        sigma_r=1,  # Distance tuning width
        num_bvc_per_dir=50,
    )

    # Plot activation with boundary
    bvc_layer.plot_activation(distances)
