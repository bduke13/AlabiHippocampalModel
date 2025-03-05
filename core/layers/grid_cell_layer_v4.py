import torch
import numpy as np

class OscillatoryInterferenceGridCellLayer:
    def __init__(
        self,
        num_modules: int = 1,
        grid_spacings: list = None,
        num_cells_per_module: int = 1000,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        phase_jitter: float = 0.01
    ):
        """Initialize the grid cell layer with an improved oscillatory interference model.

        Args:
            num_modules (int, optional): Number of grid cell modules with different spacings. Defaults to 3.
            grid_spacings (list, optional): Spatial scales (in meters) for each module. Defaults to [0.5, 0.7, 1.2].
            num_cells_per_module (int, optional): Number of grid cells per module. Defaults to 9.
            device (str, optional): Device to run computations on ('cpu' or 'cuda'). Defaults to "cpu".
            dtype (torch.dtype, optional): Data type for tensors. Defaults to torch.float32.
            phase_jitter (float, optional): Amount of random jitter to add to phase offsets (0-1). Defaults to 0.05.
        """
        self.device = device
        self.dtype = dtype
        self.num_modules = num_modules
        self.phase_jitter = phase_jitter
        
        # Set default grid spacings if None is provided
        if grid_spacings is None:
            # Create logarithmically spaced grid scales (as observed in biology)
            # Grid scales typically increase by a factor of ~1.4 along the dorsoventral axis
            base_scale = 0.3  # Starting scale (meters)
            scaling_factor = 1.4  # Scale increase between modules
            grid_spacings = [base_scale * (scaling_factor ** i) for i in range(num_modules)]
            
        # Ensure grid_spacings has length equal to num_modules
        if len(grid_spacings) != num_modules:
            # If too few, repeat the pattern
            if len(grid_spacings) < num_modules:
                grid_spacings = grid_spacings * (num_modules // len(grid_spacings) + 1)
            # Truncate if too many
            grid_spacings = grid_spacings[:num_modules]
            
        self.grid_spacings = torch.tensor(grid_spacings, dtype=dtype, device=device)
        self.num_cells_per_module = num_cells_per_module
        self.total_grid_cells = num_modules * num_cells_per_module
        
        # Generate module-specific rotations (0-60°) for biological plausibility
        # Each module has a slightly different orientation, as observed in real grid cells
        self.module_rotations = torch.rand(num_modules, dtype=dtype, device=device) * (np.pi / 3)
        
        # Generate preferred directions for each module (with rotation)
        self.preferred_directions = self._generate_preferred_directions()
        
        # Generate biologically plausible phase offsets
        self.phase_offsets = self._generate_biologically_plausible_offsets()
        
    def _generate_preferred_directions(self):
        """Generate preferred directions for each module with module-specific rotations.
        
        Returns:
            torch.Tensor: Preferred directions with shape (num_modules, 3)
        """
        # Standard directions for hexagonal grid (0°, 120°, 240°)
        # Note: Using 240° instead of 60° to maintain proper phase relationships
        base_angles = torch.tensor([0, 2*np.pi/3, 4*np.pi/3], dtype=self.dtype, device=self.device)
        
        # Apply module-specific rotations to create varied grid orientations
        directions = torch.zeros((self.num_modules, 3), dtype=self.dtype, device=self.device)
        for m in range(self.num_modules):
            directions[m] = base_angles + self.module_rotations[m]
            
        return directions
        
    def _generate_biologically_plausible_offsets(self):
        """Generate phase offsets using hexagonal tiling principles for better spatial coverage.
        
        This creates grid cells that uniformly tile space with proper hexagonal patterns,
        plus small random jitter to create biological variability.
        
        Returns:
            torch.Tensor: Phase offsets with shape (num_modules, num_cells_per_module, 3)
        """
        phase_offsets = torch.zeros(
            (self.num_modules, self.num_cells_per_module, 3), 
            dtype=self.dtype, 
            device=self.device
        )
        
        for m in range(self.num_modules):
            # Calculate how many cells we need in each dimension for proper tiling
            cells_per_side = int(np.ceil(np.sqrt(self.num_cells_per_module)))
            
            # Generate direction vectors with module-specific rotation
            dirs = torch.stack([
                torch.cos(self.preferred_directions[m]),
                torch.sin(self.preferred_directions[m])
            ], dim=1)  # Shape: (3, 2)
            
            # Create a set of cell positions that uniformly tile space
            cell_count = 0
            for i in range(cells_per_side):
                for j in range(cells_per_side):
                    if cell_count >= self.num_cells_per_module:
                        break
                        
                    # Calculate normalized position within the grid's unit cell (0-1)
                    # For a hexagonal grid, offset alternate rows for better packing
                    row_offset = 0.5 if j % 2 == 1 else 0.0
                    x_pos = (i + row_offset) / cells_per_side
                    y_pos = j / cells_per_side
                    
                    # Add small random jitter for biological plausibility
                    jitter = self.phase_jitter
                    jittered_x = (x_pos + (torch.rand(1, device=self.device, dtype=self.dtype) * 2 - 1) * jitter) % 1.0
                    jittered_y = (y_pos + (torch.rand(1, device=self.device, dtype=self.dtype) * 2 - 1) * jitter) % 1.0
                    
                    # Position as a 2D vector
                    pos = torch.tensor([jittered_x, jittered_y], dtype=self.dtype, device=self.device)
                    
                    # Project position onto each direction to get phase offsets
                    # This ensures proper hexagonal tiling
                    for d in range(3):
                        # Project position onto this direction
                        # (dot product of position with direction vector)
                        projection = pos[0] * dirs[d, 0] + pos[1] * dirs[d, 1]
                        
                        # Convert projection to phase (0-2π)
                        phase_offsets[m, cell_count, d] = (projection * 2 * np.pi) % (2 * np.pi)
                    
                    cell_count += 1
        
        return phase_offsets

    def get_grid_cell_activations(self, position: torch.Tensor) -> torch.Tensor:
        """Compute grid cell activations based on the current position.

        Args:
            position (torch.Tensor): Robot's position [x, y] in meters.

        Returns:
            torch.Tensor: Normalized grid cell activations (shape: total_grid_cells).
        """
        position = position.to(self.device)
        if position.dim() == 1:
            position = position.unsqueeze(0)  # Shape: (1, 2)
            
        all_activations = []
        
        for m in range(self.num_modules):
            # Direction vectors for this module
            dirs = torch.stack([
                torch.cos(self.preferred_directions[m]),
                torch.sin(self.preferred_directions[m])
            ], dim=1)  # Shape: (3, 2)
            
            # Module-specific grid spacing (lambda)
            lambda_m = self.grid_spacings[m]
            
            # Calculate activations for each cell in this module
            module_activations = []
            
            for c in range(self.num_cells_per_module):
                # Initialize component activations for the 3 directions
                components = torch.ones(position.shape[0], 3, device=self.device, dtype=self.dtype)
                
                for d in range(3):
                    # Project position onto this direction
                    proj = torch.matmul(position, dirs[d])  # Shape: (batch_size,)
                    
                    # Calculate phase with scaling and offset
                    # The 2π/λ term converts spatial position to phase
                    phase = (2 * np.pi / lambda_m) * proj + self.phase_offsets[m, c, d]
                    
                    # Calculate oscillatory component (cosine)
                    components[:, d] = torch.cos(phase)
                
                # Oscillatory interference: multiply all components along dim 1
                # Then square to get positive values with sharper peaks
                activation = torch.prod(components, dim=1) ** 2  # Shape: (batch_size,)
                module_activations.append(activation)
            
            # Stack all cells in this module - along a new dimension
            module_tensor = torch.stack(module_activations, dim=1)  # Shape: (batch_size, num_cells_per_module)
            all_activations.append(module_tensor)
        
        # Concatenate all modules along the cell dimension
        # Shape: (batch_size, total_grid_cells)
        activations = torch.cat(all_activations, dim=1)
        
        # If batch size is 1, squeeze the first dimension
        if activations.size(0) == 1:
            activations = activations.squeeze(0)
        
        # Normalize activations to [0, 1] range
        min_val = activations.min()
        max_val = activations.max()
        if max_val > min_val:
            normalized_activations = (activations - min_val) / (max_val - min_val)
        else:
            normalized_activations = torch.ones_like(activations) * 0.5
            
        return normalized_activations

    def visualize_grid_fields(self, size=100, extent=(-5, 5, -5, 5)):
        """Visualize the firing fields of all grid cells.
        
        Args:
            size (int): Resolution of the visualization grid
            extent (tuple): The spatial extent to visualize (xmin, xmax, ymin, ymax)
            
        Returns:
            torch.Tensor: Activation maps for all grid cells (shape: total_grid_cells, size, size)
        """
        # Create a grid of positions
        x = torch.linspace(extent[0], extent[1], size, device=self.device, dtype=self.dtype)
        y = torch.linspace(extent[2], extent[3], size, device=self.device, dtype=self.dtype)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        positions = torch.stack([xx.flatten(), yy.flatten()], dim=1)
        
        # Calculate activations for all positions at once (batched computation)
        all_activations = self.get_grid_cell_activations(positions)  # Shape: (positions, total_grid_cells)
        
        # Reshape activations into grid fields
        grid_fields = all_activations.t().reshape(self.total_grid_cells, size, size)
            
        return grid_fields
        
    def get_module_parameters(self):
        """Get the parameters of each grid cell module.
        
        Returns:
            dict: Dictionary containing module parameters
        """
        return {
            "grid_spacings": self.grid_spacings.cpu().numpy(),
            "module_rotations": self.module_rotations.cpu().numpy() * (180 / np.pi),  # Convert to degrees
            "num_cells_per_module": self.num_cells_per_module,
            "total_grid_cells": self.total_grid_cells
        }