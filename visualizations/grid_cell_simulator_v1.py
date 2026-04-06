import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from tkinter import ttk
import torch
from dataclasses import dataclass
from typing import Tuple, Optional

class GridCellLayer:
    """Simplified version of the grid cell layer from the reference code."""
    
    def __init__(
        self,
        num_cells: int = 400,
        size_range: tuple = (0.5, 0.5),
        rotation_range: tuple = (0, 90),
        spread_range: tuple = (1.2, 1.2),
        x_trans_range: tuple = (-1.0, 1.0),
        y_trans_range: tuple = (-1.0, 1.0),
        scale_multiplier: float = 5.0,
        frequency_divisor: float = 1.0,
        threshold: float = 0.7,
        threshold_type: str = 'soft',
        sparsity: float = None,
        normalization: str = 'per-cell',
        local_group_size: int = 10,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32
    ):
        """Initialize the grid cell layer with the given parameters."""
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
        """Apply the selected normalization method to the activations."""
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
        """Compute grid cell activations based on the current position."""
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


@dataclass
class GridCellParams:
    """Container for grid cell parameters."""
    num_cells: int = 400
    size_range: Tuple[float, float] = (0.5, 0.5)
    rotation_range: Tuple[float, float] = (0, 90)
    spread_range: Tuple[float, float] = (1.2, 1.2)
    x_trans_range: Tuple[float, float] = (-1.0, 1.0)
    y_trans_range: Tuple[float, float] = (-1.0, 1.0)
    scale_multiplier: float = 5.0
    frequency_divisor: float = 1.0
    threshold: float = 0.7
    threshold_type: str = 'soft'
    sparsity: Optional[float] = None
    normalization: str = 'per-cell'
    local_group_size: int = 10


class GridCellSimulator:
    """Interactive grid cell simulator with UI and visualization."""
    
    def __init__(self, master):
        """Initialize the simulator UI."""
        self.master = master
        master.title("Grid Cell Simulator")
        master.geometry("1200x800")
        
        # Set default world size
        self.world_width = 10.0
        self.world_height = 10.0
        
        # Initialize grid cell parameters
        self.params = GridCellParams()
        
        # Create grid cell layer
        self.grid_cell_layer = None
        
        # Configure layout
        self.create_ui()
        
        # Initialize grid cell layer and plots
        self.initialize_grid_cell_layer()
        
    def create_ui(self):
        """Create the user interface."""
        # Create main frame
        main_frame = ttk.Frame(self.master)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create left panel for parameters
        param_frame = ttk.LabelFrame(main_frame, text="Parameters")
        param_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # World size
        ttk.Label(param_frame, text="World Size").grid(row=0, column=0, sticky=tk.W, pady=5)
        
        world_size_frame = ttk.Frame(param_frame)
        world_size_frame.grid(row=0, column=1, sticky=tk.W, pady=5)
        
        ttk.Label(world_size_frame, text="Width:").pack(side=tk.LEFT)
        self.world_width_var = tk.StringVar(value=str(self.world_width))
        ttk.Entry(world_size_frame, textvariable=self.world_width_var, width=5).pack(side=tk.LEFT, padx=2)
        
        ttk.Label(world_size_frame, text="Height:").pack(side=tk.LEFT, padx=(10, 0))
        self.world_height_var = tk.StringVar(value=str(self.world_height))
        ttk.Entry(world_size_frame, textvariable=self.world_height_var, width=5).pack(side=tk.LEFT, padx=2)
        
        # Grid cell parameters
        row = 1
        
        # Number of cells
        ttk.Label(param_frame, text="Number of Cells:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.num_cells_var = tk.StringVar(value=str(self.params.num_cells))
        ttk.Entry(param_frame, textvariable=self.num_cells_var, width=10).grid(row=row, column=1, sticky=tk.W, pady=5)
        row += 1
        
        # Size range
        ttk.Label(param_frame, text="Size Range:").grid(row=row, column=0, sticky=tk.W, pady=5)
        
        size_range_frame = ttk.Frame(param_frame)
        size_range_frame.grid(row=row, column=1, sticky=tk.W, pady=5)
        
        ttk.Label(size_range_frame, text="Min:").pack(side=tk.LEFT)
        self.size_min_var = tk.StringVar(value=str(self.params.size_range[0]))
        ttk.Entry(size_range_frame, textvariable=self.size_min_var, width=5).pack(side=tk.LEFT, padx=2)
        
        ttk.Label(size_range_frame, text="Max:").pack(side=tk.LEFT, padx=(10, 0))
        self.size_max_var = tk.StringVar(value=str(self.params.size_range[1]))
        ttk.Entry(size_range_frame, textvariable=self.size_max_var, width=5).pack(side=tk.LEFT, padx=2)
        row += 1
        
        # Rotation range
        ttk.Label(param_frame, text="Rotation Range (°):").grid(row=row, column=0, sticky=tk.W, pady=5)
        
        rotation_range_frame = ttk.Frame(param_frame)
        rotation_range_frame.grid(row=row, column=1, sticky=tk.W, pady=5)
        
        ttk.Label(rotation_range_frame, text="Min:").pack(side=tk.LEFT)
        self.rotation_min_var = tk.StringVar(value=str(self.params.rotation_range[0]))
        ttk.Entry(rotation_range_frame, textvariable=self.rotation_min_var, width=5).pack(side=tk.LEFT, padx=2)
        
        ttk.Label(rotation_range_frame, text="Max:").pack(side=tk.LEFT, padx=(10, 0))
        self.rotation_max_var = tk.StringVar(value=str(self.params.rotation_range[1]))
        ttk.Entry(rotation_range_frame, textvariable=self.rotation_max_var, width=5).pack(side=tk.LEFT, padx=2)
        row += 1
        
        # Spread range
        ttk.Label(param_frame, text="Spread Range:").grid(row=row, column=0, sticky=tk.W, pady=5)
        
        spread_range_frame = ttk.Frame(param_frame)
        spread_range_frame.grid(row=row, column=1, sticky=tk.W, pady=5)
        
        ttk.Label(spread_range_frame, text="Min:").pack(side=tk.LEFT)
        self.spread_min_var = tk.StringVar(value=str(self.params.spread_range[0]))
        ttk.Entry(spread_range_frame, textvariable=self.spread_min_var, width=5).pack(side=tk.LEFT, padx=2)
        
        ttk.Label(spread_range_frame, text="Max:").pack(side=tk.LEFT, padx=(10, 0))
        self.spread_max_var = tk.StringVar(value=str(self.params.spread_range[1]))
        ttk.Entry(spread_range_frame, textvariable=self.spread_max_var, width=5).pack(side=tk.LEFT, padx=2)
        row += 1
        
        # Translation ranges
        ttk.Label(param_frame, text="X Translation Range:").grid(row=row, column=0, sticky=tk.W, pady=5)
        
        x_trans_range_frame = ttk.Frame(param_frame)
        x_trans_range_frame.grid(row=row, column=1, sticky=tk.W, pady=5)
        
        ttk.Label(x_trans_range_frame, text="Min:").pack(side=tk.LEFT)
        self.x_trans_min_var = tk.StringVar(value=str(self.params.x_trans_range[0]))
        ttk.Entry(x_trans_range_frame, textvariable=self.x_trans_min_var, width=5).pack(side=tk.LEFT, padx=2)
        
        ttk.Label(x_trans_range_frame, text="Max:").pack(side=tk.LEFT, padx=(10, 0))
        self.x_trans_max_var = tk.StringVar(value=str(self.params.x_trans_range[1]))
        ttk.Entry(x_trans_range_frame, textvariable=self.x_trans_max_var, width=5).pack(side=tk.LEFT, padx=2)
        row += 1
        
        ttk.Label(param_frame, text="Y Translation Range:").grid(row=row, column=0, sticky=tk.W, pady=5)
        
        y_trans_range_frame = ttk.Frame(param_frame)
        y_trans_range_frame.grid(row=row, column=1, sticky=tk.W, pady=5)
        
        ttk.Label(y_trans_range_frame, text="Min:").pack(side=tk.LEFT)
        self.y_trans_min_var = tk.StringVar(value=str(self.params.y_trans_range[0]))
        ttk.Entry(y_trans_range_frame, textvariable=self.y_trans_min_var, width=5).pack(side=tk.LEFT, padx=2)
        
        ttk.Label(y_trans_range_frame, text="Max:").pack(side=tk.LEFT, padx=(10, 0))
        self.y_trans_max_var = tk.StringVar(value=str(self.params.y_trans_range[1]))
        ttk.Entry(y_trans_range_frame, textvariable=self.y_trans_max_var, width=5).pack(side=tk.LEFT, padx=2)
        row += 1
        
        # Scale multiplier
        ttk.Label(param_frame, text="Scale Multiplier:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.scale_multiplier_var = tk.StringVar(value=str(self.params.scale_multiplier))
        ttk.Entry(param_frame, textvariable=self.scale_multiplier_var, width=10).grid(row=row, column=1, sticky=tk.W, pady=5)
        row += 1
        
        # Frequency divisor
        ttk.Label(param_frame, text="Frequency Divisor:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.frequency_divisor_var = tk.StringVar(value=str(self.params.frequency_divisor))
        ttk.Entry(param_frame, textvariable=self.frequency_divisor_var, width=10).grid(row=row, column=1, sticky=tk.W, pady=5)
        row += 1
        
        # Threshold
        ttk.Label(param_frame, text="Threshold:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.threshold_var = tk.StringVar(value=str(self.params.threshold))
        ttk.Entry(param_frame, textvariable=self.threshold_var, width=10).grid(row=row, column=1, sticky=tk.W, pady=5)
        row += 1
        
        # Threshold type
        ttk.Label(param_frame, text="Threshold Type:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.threshold_type_var = tk.StringVar(value=self.params.threshold_type)
        ttk.Combobox(param_frame, textvariable=self.threshold_type_var, values=['hard', 'soft', 'binary'], width=10).grid(row=row, column=1, sticky=tk.W, pady=5)
        row += 1
        
        # Sparsity
        ttk.Label(param_frame, text="Sparsity (0-1):").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.sparsity_var = tk.StringVar(value=str(self.params.sparsity) if self.params.sparsity is not None else "")
        ttk.Entry(param_frame, textvariable=self.sparsity_var, width=10).grid(row=row, column=1, sticky=tk.W, pady=5)
        row += 1
        
        # Normalization
        ttk.Label(param_frame, text="Normalization:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.normalization_var = tk.StringVar(value=self.params.normalization)
        ttk.Combobox(param_frame, textvariable=self.normalization_var, values=['none', 'global', 'per-cell', 'soft', 'local'], width=10).grid(row=row, column=1, sticky=tk.W, pady=5)
        row += 1
        
        # Local group size
        ttk.Label(param_frame, text="Local Group Size:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.local_group_size_var = tk.StringVar(value=str(self.params.local_group_size))
        ttk.Entry(param_frame, textvariable=self.local_group_size_var, width=10).grid(row=row, column=1, sticky=tk.W, pady=5)
        row += 1
        
        # Add parameter for selecting which grid cell to visualize
        ttk.Label(param_frame, text="Selected Cell Index:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.selected_cell_var = tk.StringVar(value="0")
        ttk.Entry(param_frame, textvariable=self.selected_cell_var, width=10).grid(row=row, column=1, sticky=tk.W, pady=5)
        row += 1
        
        # Add a frame for displaying uniformity metrics
        metrics_frame = ttk.LabelFrame(param_frame, text="Uniformity Metrics")
        metrics_frame.grid(row=row, column=0, columnspan=2, sticky=tk.W+tk.E, pady=10)
        
        # Label for the coefficient of variation
        ttk.Label(metrics_frame, text="Coefficient of Variation:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.cv_var = tk.StringVar(value="N/A")
        ttk.Label(metrics_frame, textvariable=self.cv_var).grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Label for the entropy
        ttk.Label(metrics_frame, text="Normalized Entropy:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.entropy_var = tk.StringVar(value="N/A")
        ttk.Label(metrics_frame, textvariable=self.entropy_var).grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Label for the Gini coefficient
        ttk.Label(metrics_frame, text="Gini Coefficient:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.gini_var = tk.StringVar(value="N/A")
        ttk.Label(metrics_frame, textvariable=self.gini_var).grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        
        row += 1
        
        # Add update button
        ttk.Button(param_frame, text="Update Simulation", command=self.update_simulation).grid(row=row, column=0, columnspan=2, pady=10)
        row += 1
        
        # Create right panel for plots
        plot_frame = ttk.Frame(main_frame)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Top plot for single grid cell
        self.single_cell_frame = ttk.LabelFrame(plot_frame, text="Single Grid Cell")
        self.single_cell_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Bottom plot for average activations
        self.avg_act_frame = ttk.LabelFrame(plot_frame, text="Average Grid Cell Activations")
        self.avg_act_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Initialize plot figures
        self.single_cell_fig = Figure(figsize=(6, 4), dpi=100)
        self.single_cell_canvas = FigureCanvasTkAgg(self.single_cell_fig, master=self.single_cell_frame)
        self.single_cell_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.avg_act_fig = Figure(figsize=(6, 4), dpi=100)
        self.avg_act_canvas = FigureCanvasTkAgg(self.avg_act_fig, master=self.avg_act_frame)
        self.avg_act_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add navigation toolbars
        self.single_cell_toolbar = NavigationToolbar2Tk(self.single_cell_canvas, self.single_cell_frame)
        self.single_cell_toolbar.update()
        
        self.avg_act_toolbar = NavigationToolbar2Tk(self.avg_act_canvas, self.avg_act_frame)
        self.avg_act_toolbar.update()
    
    def calculate_uniformity_metrics(self, activations):
        """Calculate various metrics to measure the uniformity of activations."""
        # Flatten the activations
        flat_activations = activations.flatten()
        
        # Coefficient of Variation (CV) - lower values indicate more uniform distribution
        mean = np.mean(flat_activations)
        std = np.std(flat_activations)
        cv = std / mean if mean > 0 else float('inf')
        
        # Normalize activations for entropy calculation
        # Add a small constant to avoid log(0)
        epsilon = 1e-10
        normalized = flat_activations / (np.sum(flat_activations) + epsilon)
        
        # Calculate entropy - higher values indicate more uniform distribution
        entropy = -np.sum(normalized * np.log2(normalized + epsilon))
        # Normalize entropy (divide by log2(n) where n is the number of elements)
        max_entropy = np.log2(len(flat_activations))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        # Calculate Gini coefficient - lower values indicate more uniform distribution
        # Sort the array
        sorted_activations = np.sort(flat_activations)
        # Calculate the cumulative sum
        cumsum = np.cumsum(sorted_activations)
        # Calculate the Lorenz curve points
        lorenz_curve = cumsum / cumsum[-1] if cumsum[-1] > 0 else cumsum
        # Calculate the area under the Lorenz curve
        n = len(lorenz_curve)
        area_under_lorenz = np.sum(lorenz_curve) / n
        # Gini coefficient
        gini = 1 - 2 * area_under_lorenz
        
        return {
            'cv': cv,
            'entropy': normalized_entropy,
            'gini': gini
        }
        
    def get_params_from_ui(self):
        """Get grid cell parameters from UI inputs."""
        try:
            # Parse world size
            self.world_width = float(self.world_width_var.get())
            self.world_height = float(self.world_height_var.get())
            
            # Parse grid cell parameters
            num_cells = int(self.num_cells_var.get())
            size_range = (float(self.size_min_var.get()), float(self.size_max_var.get()))
            rotation_range = (float(self.rotation_min_var.get()), float(self.rotation_max_var.get()))
            spread_range = (float(self.spread_min_var.get()), float(self.spread_max_var.get()))
            x_trans_range = (float(self.x_trans_min_var.get()), float(self.x_trans_max_var.get()))
            y_trans_range = (float(self.y_trans_min_var.get()), float(self.y_trans_max_var.get()))
            scale_multiplier = float(self.scale_multiplier_var.get())
            frequency_divisor = float(self.frequency_divisor_var.get())
            threshold = float(self.threshold_var.get())
            threshold_type = self.threshold_type_var.get()
            sparsity_str = self.sparsity_var.get()
            sparsity = float(sparsity_str) if sparsity_str else None
            normalization = self.normalization_var.get()
            local_group_size = int(self.local_group_size_var.get())
            
            # Update params
            self.params = GridCellParams(
                num_cells=num_cells,
                size_range=size_range,
                rotation_range=rotation_range,
                spread_range=spread_range,
                x_trans_range=x_trans_range,
                y_trans_range=y_trans_range,
                scale_multiplier=scale_multiplier,
                frequency_divisor=frequency_divisor,
                threshold=threshold,
                threshold_type=threshold_type,
                sparsity=sparsity,
                normalization=normalization,
                local_group_size=local_group_size
            )
            
            return True
        except ValueError as e:
            tk.messagebox.showerror("Input Error", f"Invalid parameter value: {str(e)}")
            return False
    
    def initialize_grid_cell_layer(self):
        """Initialize or update the grid cell layer with current parameters."""
        self.grid_cell_layer = GridCellLayer(
            num_cells=self.params.num_cells,
            size_range=self.params.size_range,
            rotation_range=self.params.rotation_range,
            spread_range=self.params.spread_range,
            x_trans_range=self.params.x_trans_range,
            y_trans_range=self.params.y_trans_range,
            scale_multiplier=self.params.scale_multiplier,
            frequency_divisor=self.params.frequency_divisor,
            threshold=self.params.threshold,
            threshold_type=self.params.threshold_type,
            sparsity=self.params.sparsity,
            normalization=self.params.normalization,
            local_group_size=self.params.local_group_size
        )
        
        # Update plots
        self.update_plots()
    
    def update_simulation(self):
        """Update the simulation with new parameters."""
        if self.get_params_from_ui():
            self.initialize_grid_cell_layer()
    
    def update_plots(self):
        """Update both visualization plots."""
        if self.grid_cell_layer is None:
            return
        
        # Clear existing plots to prevent legend duplication
        self.single_cell_fig.clear()
        self.avg_act_fig.clear()
        
        # Create grid of positions
        resolution = 100
        x = np.linspace(-self.world_width/2, self.world_width/2, resolution)
        y = np.linspace(-self.world_height/2, self.world_height/2, resolution)
        X, Y = np.meshgrid(x, y)
        
        # Compute activations for all positions
        activations = np.zeros((resolution, resolution, self.params.num_cells))
        for i in range(resolution):
            for j in range(resolution):
                pos = torch.tensor([X[i, j], Y[i, j]], dtype=torch.float32)
                act = self.grid_cell_layer.get_grid_cell_activations(pos).cpu().numpy()
                activations[i, j, :] = act
        
        # Plot single grid cell
        try:
            selected_cell = int(self.selected_cell_var.get())
            if selected_cell < 0 or selected_cell >= self.params.num_cells:
                selected_cell = 0
        except ValueError:
            selected_cell = 0
        
        # Update single cell plot
        ax1 = self.single_cell_fig.add_subplot(111)
        single_cell_data = activations[:, :, selected_cell]
        im1 = ax1.imshow(
            single_cell_data,
            extent=[-self.world_width/2, self.world_width/2, -self.world_height/2, self.world_height/2],
            origin='lower',
            cmap='viridis',
            interpolation='bilinear'
        )
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_title(f'Grid Cell {selected_cell} Activation')
        self.single_cell_fig.colorbar(im1, ax=ax1, label='Activation')
        
        # Compute average activations and plot
        avg_activations = np.mean(activations, axis=2)
        
        # Calculate uniformity metrics for average activations
        metrics = self.calculate_uniformity_metrics(avg_activations)
        
        # Update metric displays
        self.cv_var.set(f"{metrics['cv']:.4f}")
        self.entropy_var.set(f"{metrics['entropy']:.4f}")
        self.gini_var.set(f"{metrics['gini']:.4f}")
        
        # Plot average activations
        ax2 = self.avg_act_fig.add_subplot(111)
        im2 = ax2.imshow(
            avg_activations,
            extent=[-self.world_width/2, self.world_width/2, -self.world_height/2, self.world_height/2],
            origin='lower',
            cmap='viridis',
            interpolation='bilinear'
        )
        
        # Add uniformity metrics to the plot title
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_title(f'Average Grid Cell Activation\nCV: {metrics["cv"]:.4f} | Entropy: {metrics["entropy"]:.4f} | Gini: {metrics["gini"]:.4f}')
        self.avg_act_fig.colorbar(im2, ax=ax2, label='Activation')
        
        # Draw the updated figures
        self.single_cell_fig.tight_layout()
        self.avg_act_fig.tight_layout()
        self.single_cell_canvas.draw()
        self.avg_act_canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = GridCellSimulator(root)
    root.mainloop()
