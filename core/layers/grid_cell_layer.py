import torch
import numpy as np

class GridCellLayer:
    def __init__(
        self,
        num_cells: int = 3000,  # Total number of grid cells
        size_range: tuple = (0.5, 0.5),  # Range for the grid scale parameter
        rotation_range: tuple = (0, 90),  # Range for rotation angles (in degrees)
        spread_range: tuple = (1.2, 1.2),  # Range for the spread parameter
        x_trans_range: tuple = (-1.0, 1.0),  # Range for x-axis translation
        y_trans_range: tuple = (-1.0, 1.0),  # Range for y-axis translation
        scale_multiplier: float = 5.0,  # Multiplier to adjust grid scale
        frequency_divisor: float = 1.0,  # Divisor to increase spacing between grid fields
        threshold: float = 0.7,  # Activation threshold value (0.0 to 1.0)
        threshold_type: str = 'soft',  # Type of thresholding ('hard', 'soft', 'binary')
        sparsity: float = None,  # If provided, keep only top X% of activations
        normalization: str = 'per-cell',  # Normalization method ('none', 'global', 'per-cell', 'soft', 'local')
        local_group_size: int = 10,  # Group size for local normalization
        device: str = "cpu",
        dtype: torch.dtype = torch.float32
    ):
        """Initialize the grid cell layer with simplified model.
        
        Args:
            num_cells (int): Total number of grid cells.
            size_range (tuple): Range for the grid scale parameter.
            rotation_range (tuple): Range for rotation angles (in degrees).
            spread_range (tuple): Range for the spread parameter.
            x_trans_range (tuple): Range for x-axis translation.
            y_trans_range (tuple): Range for y-axis translation.
            device (str): Device to run computations on ('cpu' or 'cuda').
            dtype (torch.dtype): Data type for tensors.
        """
        # Set device
        self.device_str = device
        self.device = torch.device(device)
        self.dtype = dtype
        self.total_grid_cells = num_cells
        
        # Save thresholding parameters
        self.threshold = threshold
        self.threshold_type = threshold_type
        self.sparsity = sparsity
        
        # Save normalization parameters
        self.normalization = normalization
        self.local_group_size = local_group_size
        
        # Save frequency divisor parameter
        self.frequency_divisor = frequency_divisor
        
        # Initialize per-cell normalization parameters if needed
        if normalization == 'per-cell':
            # These will be updated during the first few calls
            self.cell_min = torch.ones(num_cells, dtype=self.dtype, device=self.device) * -1.0
            self.cell_max = torch.ones(num_cells, dtype=self.dtype, device=self.device) * 1.0
            self.min_max_updated = False
        
        # Set random seed for reproducibility
        torch.manual_seed(42)
        
        # Initialize parameter tensors directly on the specified device
        # Apply scale multiplier to size parameters to make grid cells smaller
        adjusted_size_range = (size_range[0] * scale_multiplier, size_range[1] * scale_multiplier)
        self.size_params = torch.FloatTensor(num_cells).uniform_(*adjusted_size_range).to(self.device)
        self.rotation_params = torch.FloatTensor(num_cells).uniform_(*rotation_range).to(self.device)
        self.spread_params = torch.FloatTensor(num_cells).uniform_(*spread_range).to(self.device)
        self.x_trans_params = torch.FloatTensor(num_cells).uniform_(*x_trans_range).to(self.device)
        self.y_trans_params = torch.FloatTensor(num_cells).uniform_(*y_trans_range).to(self.device)
        
        # Precompute rotation matrices for efficiency
        theta_rad = torch.deg2rad(self.rotation_params)
        self.cos_theta = torch.cos(theta_rad)
        self.sin_theta = torch.sin(theta_rad)

    def _normalize_activations(self, activations):
        """Apply the selected normalization method to the activations.
        
        Args:
            activations (torch.Tensor): Raw activations to normalize
            
        Returns:
            torch.Tensor: Normalized activations
        """
        if self.normalization == 'none':
            # Return raw activations without normalization
            return activations
            
        elif self.normalization == 'global':
            # Global min-max normalization across all cells
            min_val = activations.min()
            max_val = activations.max()
            if max_val > min_val:
                return (activations - min_val) / (max_val - min_val)
            else:
                return torch.ones_like(activations) * 0.5
                
        elif self.normalization == 'per-cell':
            # Update min/max values for each cell if needed
            if not self.min_max_updated:
                self.cell_min = torch.minimum(self.cell_min, activations)
                self.cell_max = torch.maximum(self.cell_max, activations)
                
            # Per-cell normalization
            denominator = self.cell_max - self.cell_min
            # Avoid division by zero
            denominator = torch.where(denominator > 0, denominator, torch.ones_like(denominator))
            normalized = (activations - self.cell_min) / denominator
            
            # Clamp to handle potential issues during early calls
            return torch.clamp(normalized, 0.0, 1.0)
            
        elif self.normalization == 'soft':
            # Soft normalization using sigmoid
            # Scale and shift to map typical activation range to sigmoid's sensitive region
            scaled = 3.0 * activations  # Scale factor can be adjusted
            return torch.sigmoid(scaled)
            
        elif self.normalization == 'local':
            # Local normalization within groups of cells
            normalized = torch.zeros_like(activations)
            
            # Process each group separately
            for i in range(0, self.total_grid_cells, self.local_group_size):
                end_idx = min(i + self.local_group_size, self.total_grid_cells)
                group = activations[i:end_idx]
                
                # Normalize within this group
                group_min = group.min()
                group_max = group.max()
                if group_max > group_min:
                    normalized[i:end_idx] = (group - group_min) / (group_max - group_min)
                else:
                    normalized[i:end_idx] = torch.ones_like(group) * 0.5
                    
            return normalized
            
        else:
            raise ValueError(f"Unknown normalization method: {self.normalization}")
            
    def get_grid_cell_activations(self, position, threshold=None, threshold_type=None, sparsity=None, normalization=None):
        """Compute grid cell activations based on the current position.
        
        Args:
            position (torch.Tensor or array-like): Robot's position [x, y] in meters.
            threshold (float, optional): Override the instance threshold value.
            threshold_type (str, optional): Override the instance threshold type.
            sparsity (float, optional): Override the instance sparsity value.
            normalization (str, optional): Override the instance normalization method.
            
        Returns:
            torch.Tensor: Normalized grid cell activations (shape: total_grid_cells).
        """
        # Use instance parameters if not overridden
        threshold = threshold if threshold is not None else self.threshold
        threshold_type = threshold_type if threshold_type is not None else self.threshold_type
        sparsity = sparsity if sparsity is not None else self.sparsity
        normalization = normalization if normalization is not None else self.normalization
        # Convert position to tensor if it's not already
        if not isinstance(position, torch.Tensor):
            position = torch.tensor(position, dtype=self.dtype)
            
        # Make sure position is on the correct device
        position = position.to(self.device)
        
        # Ensure position is 1D tensor [x, y]
        if position.dim() > 1:
            position = position.squeeze()
        
        # Extract x and y coordinates
        x, y = position[0], position[1]
        
        # Apply translation for all cells at once
        x_translated = x - self.x_trans_params
        y_translated = y - self.y_trans_params
        
        # Rotate coordinates for all cells
        x_rot = x_translated * self.cos_theta - y_translated * self.sin_theta
        y_rot = x_translated * self.sin_theta + y_translated * self.cos_theta
        
        # Create grid patterns using cosine combinations for all cells
        # Apply frequency divisor to increase spacing between activations
        effective_size = self.size_params / self.frequency_divisor
        
        z1 = torch.cos(effective_size * x_rot)
        z2 = torch.cos(0.5 * effective_size * x_rot + effective_size * y_rot)
        z3 = torch.cos(0.5 * effective_size * x_rot - effective_size * y_rot)
        
        # Sum and normalize
        z = (z1 + z2 + z3) / 3.0
        
        # Apply spread transformation
        activations = torch.sign(z) * torch.pow(torch.abs(z), 1 / self.spread_params)
        
        # Apply normalization
        normalized_activations = self._normalize_activations(activations)
        
        # Apply thresholding if requested
        if threshold is not None:
            if threshold_type == 'hard':
                # Hard threshold: set values below threshold to zero
                normalized_activations = torch.where(
                    normalized_activations >= threshold, 
                    normalized_activations, 
                    torch.zeros_like(normalized_activations)
                )
            elif threshold_type == 'soft':
                # Soft threshold: gradually suppress values below threshold
                # Use sigmoid to create a smooth transition at the threshold point
                steepness = 10  # Controls how sharp the transition is
                shift = normalized_activations - threshold
                scaling = torch.sigmoid(steepness * shift)
                normalized_activations = normalized_activations * scaling
            elif threshold_type == 'binary':
                # Binary threshold: values above threshold become 1, below become 0
                normalized_activations = (normalized_activations >= threshold).float()
        
        # Apply sparsity if requested (keep only top X% of activations)
        if sparsity is not None:
            if sparsity <= 0 or sparsity > 1:
                raise ValueError("Sparsity must be between 0 and 1")
                
            # Calculate the activation threshold for the desired sparsity
            k = int(self.total_grid_cells * sparsity)
            if k > 0:
                # Sort activations and find the threshold value
                sorted_activations, _ = torch.sort(normalized_activations, descending=True)
                sparsity_threshold = sorted_activations[k-1]
                
                # Apply the threshold
                normalized_activations = torch.where(
                    normalized_activations >= sparsity_threshold,
                    normalized_activations,
                    torch.zeros_like(normalized_activations)
                )
        
        return normalized_activations
    
    def visualize_manifold(self, num_samples=1000, grid_size=100, axis_size=5):
        """
        Visualize the manifold structure of the grid cell representations.
        
        Args:
            num_samples (int): Number of random positions to sample.
            grid_size (int): Size of the grid for sampling positions.
            axis_size (float): Size of the spatial region to sample from.
            
        Returns:
            dict: Dictionary containing data for visualization, including PCA results.
        """
        # Sample random positions in the environment
        positions = torch.rand(num_samples, 2, device=self.device) * 2 * axis_size - axis_size
        
        # Compute grid cell activations for each position
        activations = torch.zeros((num_samples, self.total_grid_cells), device=self.device)
        for i in range(num_samples):
            activations[i] = self.get_grid_cell_activations(positions[i])
        
        # Move data to CPU for PCA
        activations_np = activations.cpu().numpy()
        
        # Perform PCA
        from sklearn.decomposition import PCA
        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(activations_np)
        
        return {
            'pca_result': pca_result,
            'positions': positions.cpu().numpy(),
            'explained_variance_ratio': pca.explained_variance_ratio_
        }