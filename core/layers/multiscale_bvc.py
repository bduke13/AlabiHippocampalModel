import numpy as np
import torch


class BoundaryVectorCellLayer:
    """Boundary vector cell layer for LiDAR distance readings."""

    def __init__(
        self,
        n_res: int,
        n_hd: int,
        sigma_theta: float,
        sigma_r: float,
        max_dist: float,
        num_bvc_per_dir: int = 50,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.device = device
        self.dtype = dtype
        self.n_res = n_res

        self.sigma_theta = torch.tensor(np.deg2rad(sigma_theta), dtype=dtype, device=device)
        self.sigma_r = torch.tensor(sigma_r, dtype=dtype, device=device)
        self.inv_two_sigma_r2 = 1.0 / (2.0 * self.sigma_r**2)
        self.distance_gaussian_scale = 1.0 / torch.sqrt(2 * torch.pi * self.sigma_r**2)

        self.lidar_angles = torch.linspace(
            0,
            2 * torch.pi,
            steps=n_res,
            dtype=dtype,
            device=device,
        )

        tuned_dist = torch.linspace(
            0,
            max_dist,
            steps=num_bvc_per_dir,
            dtype=dtype,
            device=device,
        )
        n_dist = len(tuned_dist)
        preferred_angles = torch.linspace(
            0,
            2 * torch.pi,
            steps=n_hd + 1,
            dtype=dtype,
            device=device,
        )[:-1]

        self.d_i = tuned_dist.repeat(n_hd).unsqueeze(0)
        self.d_i_column = self.d_i.T
        self.phi_i = preferred_angles.repeat_interleave(n_dist).unsqueeze(0)
        self.num_bvc = self.d_i.numel()

        lidar_angles_expanded = self.lidar_angles.unsqueeze(0)
        phi_i_expanded = self.phi_i.T
        angular_diff = torch.remainder(
            torch.abs(lidar_angles_expanded - phi_i_expanded),
            2 * torch.pi,
        )
        angular_diff = torch.minimum(angular_diff, 2 * torch.pi - angular_diff)

        self.angular_gaussian_matrix = torch.exp(
            -(angular_diff**2) / (2 * self.sigma_theta**2)
        ) / torch.sqrt(2 * torch.pi * self.sigma_theta**2)

        self.bvc_activations = None

    def _ensure_runtime_constants(self) -> None:
        if not hasattr(self, "d_i_column"):
            self.d_i_column = self.d_i.T
        if not hasattr(self, "inv_two_sigma_r2"):
            self.inv_two_sigma_r2 = 1.0 / (2.0 * self.sigma_r**2)
        if not hasattr(self, "distance_gaussian_scale"):
            self.distance_gaussian_scale = 1.0 / torch.sqrt(2 * torch.pi * self.sigma_r**2)

    def get_bvc_activation(self, distances: torch.Tensor) -> torch.Tensor:
        """Compute BVC activations from LiDAR distance readings."""
        self._ensure_runtime_constants()
        distances_expanded = distances.unsqueeze(0)
        distance_gaussian_matrix = torch.exp(
            -((distances_expanded - self.d_i_column) ** 2) * self.inv_two_sigma_r2
        )
        distance_gaussian_matrix.mul_(self.distance_gaussian_scale)
        distance_gaussian_matrix.mul_(self.angular_gaussian_matrix)

        bvc_activations = torch.sum(distance_gaussian_matrix, dim=1) / self.num_bvc

        self.bvc_activations = bvc_activations
        return bvc_activations
