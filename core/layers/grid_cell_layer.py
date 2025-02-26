import torch
import numpy as np

class OscillatoryInterferenceGridCellLayer:
    def __init__(
        self,
        num_modules: int = 3,
        grid_spacings: list = [0.5, 0.7, 1.2],
        num_cells_per_module: int = 5,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32
    ):
        """Initialize the grid cell layer with oscillatory interference model.

        Args:
            num_modules (int): Number of grid cell modules with different spacings.
            grid_spacings (list): Spatial scales (in meters) for each module.
            num_cells_per_module (int): Number of grid cells per module.
            device (str): Device to run computations on ('cpu' or 'cuda').
            dtype (torch.dtype): Data type for tensors.
        """
        self.device = device
        self.dtype = dtype
        self.num_modules = num_modules
        self.grid_spacings = torch.tensor(grid_spacings, dtype=dtype, device=device)
        self.num_cells_per_module = num_cells_per_module
        self.total_grid_cells = num_modules * num_cells_per_module
        

        # Base directions
        base_dirs = torch.deg2rad(torch.tensor([0, 60, 120], dtype=dtype, device=device))
        # Random orientation offsets for each module (±10°)
        orientation_offsets = (
            torch.rand(num_modules, dtype=dtype, device=device) * torch.deg2rad(torch.tensor(70.0))
            - torch.deg2rad(torch.tensor(10.0))
        )
        # Module-specific preferred directions
        self.preferred_directions = torch.zeros((num_modules, 3), dtype=dtype, device=device)
        for m in range(num_modules):
            self.preferred_directions[m] = base_dirs + orientation_offsets[m]

        # Random phase offsets
        self.phase_offsets = (
            torch.rand((num_modules, num_cells_per_module, 3), dtype=dtype, device=device)
            * 2
            * np.pi
        )

        """
        self.phase_offsets = (
            torch.full((num_modules, num_cells_per_module, 3), 1.0, dtype=dtype, device=device)
            * 2
            * np.pi
        )
        self.phase_offsets = torch.zeros(
            (num_modules, num_cells_per_module, 3), dtype=dtype, device=device
        )
        """

    def get_grid_cell_activations(self, position: torch.Tensor) -> torch.Tensor:
        """Compute grid cell activations based on the current position.

        Args:
            position (torch.Tensor): Robot's position [x, y] in meters.

        Returns:
            torch.Tensor: Normalized grid cell activations (shape: total_grid_cells).
        """
        
        position = position.to(self.device).unsqueeze(0)  # Shape: (1, 2)
        activations = []
        for m in range(self.num_modules):
            # Use module-specific directions
            directions = torch.stack(
                [torch.cos(self.preferred_directions[m]), torch.sin(self.preferred_directions[m])],
                dim=1
            )  # Shape: (3, 2)
            proj = torch.matmul(position, directions.T)  # Shape: (1, 3)
            lambda_m = self.grid_spacings[m]
            phases = (2 * np.pi / lambda_m) * proj + self.phase_offsets[m]  # Shape: (num_cells, 3)
            activation = torch.prod(torch.cos(phases), dim=1) ** 2
            activations.append(activation)
        raw_activations = torch.cat(activations)
        min_val = raw_activations.min()
        max_val = raw_activations.max()
        if max_val > min_val:
            normalized_activations = (raw_activations - min_val) / (2.0 * (max_val - min_val)) + 0.5
        else:
            normalized_activations = torch.ones_like(raw_activations) * 0.5
        return normalized_activations