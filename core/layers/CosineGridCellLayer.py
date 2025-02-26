import torch
import numpy as np

class CosineGridCellLayer:
    def __init__(
        self,
        num_cells: int = 2000,
        size_range: tuple = (0.5, 2.0),
        rotation_range: tuple = (0.0, 25.0),
        spread_range: tuple = (0.5, 1.0),
        x_trans_range: tuple = (-2.0, 2.0),
        y_trans_range: tuple = (-2.0, 2.0),
        device: str = "cpu",
        dtype: torch.dtype = torch.float32
    ):
        """Initialize the grid cell layer with cosine combination patterns.

        Args:
            num_cells (int): Number of grid cells to generate.
            size_range (tuple): Range for the size parameter (min, max).
            rotation_range (tuple): Range for the rotation parameter in degrees (min, max).
            spread_range (tuple): Range for the spread parameter (min, max).
            x_trans_range (tuple): Range for the x translation parameter (min, max).
            y_trans_range (tuple): Range for the y translation parameter (min, max).
            device (str): Device to run computations on ('cpu' or 'cuda').
            dtype (torch.dtype): Data type for tensors (e.g., torch.float32).
        """
        self.device = torch.device(device)
        self.dtype = dtype
        self.num_cells = num_cells

        # Sample parameters for each grid cell from the specified ranges
        self.size_params = torch.empty(num_cells, dtype=dtype, device=self.device).uniform_(*size_range)
        self.rotation_params = torch.empty(num_cells, dtype=dtype, device=self.device).uniform_(*rotation_range)
        self.spread_params = torch.empty(num_cells, dtype=dtype, device=self.device).uniform_(*spread_range)
        self.x_trans_params = torch.empty(num_cells, dtype=dtype, device=self.device).uniform_(*x_trans_range)
        self.y_trans_params = torch.empty(num_cells, dtype=dtype, device=self.device).uniform_(*y_trans_range)

    def grid_cell_pattern(self, x, y, size, rotation, spread, x_trans, y_trans):
        """Compute the grid cell activation for a single position (x, y).

        Args:
            x (float or torch.Tensor): X-coordinate of the position.
            y (float or torch.Tensor): Y-coordinate of the position.
            size (float): Size parameter for the grid pattern.
            rotation (float): Rotation angle in degrees.
            spread (float): Spread parameter for normalization.
            x_trans (float): X-axis translation.
            y_trans (float): Y-axis translation.

        Returns:
            torch.Tensor: Activation value at position (x, y).
        """
        # Ensure inputs are tensors
        x = torch.tensor(x, dtype=self.dtype, device=self.device) if not isinstance(x, torch.Tensor) else x
        y = torch.tensor(y, dtype=self.dtype, device=self.device) if not isinstance(y, torch.Tensor) else y

        # Convert rotation to radians
        theta = torch.deg2rad(rotation)
        cos_t, sin_t = torch.cos(theta), torch.sin(theta)

        # Apply translation
        x = x - x_trans
        y = y - y_trans

        # Rotate coordinates
        x_rot = x * cos_t - y * sin_t
        y_rot = x * sin_t + y * cos_t

        # Compute the pattern using cosine combinations
        z = (
            torch.cos(size * x_rot)
            + torch.cos(0.5 * size * x_rot + size * y_rot)
            + torch.cos(0.5 * size * x_rot - size * y_rot)
        )

        # Normalize and apply spread transformation
        z = z / 3.0
        z = torch.sign(z) * torch.pow(torch.abs(z), 1 / spread)
        return z

    def get_grid_cell_activations(self, position: torch.Tensor) -> torch.Tensor:
        """Compute grid cell activations for a given position [x, y].

        Args:
            position (torch.Tensor): Position tensor of shape [2] containing [x, y].

        Returns:
            torch.Tensor: Grid cell activations of shape [num_cells].
        """
        x, y = position[0], position[1]
        activations = torch.zeros(self.num_cells, dtype=self.dtype, device=self.device)

        # Compute activation for each grid cell
        for i in range(self.num_cells):
            activations[i] = self.grid_cell_pattern(
                x, y,
                size=self.size_params[i],
                rotation=self.rotation_params[i],
                spread=self.spread_params[i],
                x_trans=self.x_trans_params[i],
                y_trans=self.y_trans_params[i]
            )

        return activations