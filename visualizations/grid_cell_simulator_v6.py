import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.patches as patches
import tkinter as tk
from tkinter import ttk, messagebox
import torch
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, List
from scipy import ndimage
from skimage import measure

# -----------------------------
# World definitions (as given)
# -----------------------------
WORLD_CONFIGS: Dict[str, Dict] = {
    "20x20_multi_goal": {
        "size": [20.0, 20.0],  # [width, height]
        "obstacles": []
    },
    "20x20_cross_multi_goal": {
        "size": [20.0, 20.0],
        "obstacles": [
            {
                "type": "rectangle",
                "name": "horizontal_wall",
                # Bounds: [-6, -0.15] to [6, 0.15]
                "bounds": [[-6.0, -0.15], [6.0, 0.15]]
            },
            {
                "type": "rectangle",
                "name": "vertical_wall",
                # Bounds: [-0.15, -6] to [0.15, 6]
                "bounds": [[-0.15, -6.0], [0.15, 6.0]]
            }
        ]
    },
    "20x20_maze_multi_goal": {
        "size": [20.0, 20.0],
        "obstacles": [
            {
                "type": "rectangle",
                "name": "MazeMid_HorizLeft",
                # Bounds: x=[-10, 0], y=[2.85, 3.15]
                "bounds": [[-10.0, 2.85], [0.0, 3.15]]
            },
            {
                "type": "rectangle",
                "name": "MazeMid_HorizRight",
                # Bounds: x=[0, 10], y=[-3.15, -2.85]
                "bounds": [[0.0, -3.15], [10.0, -2.85]]
            },
            {
                "type": "rectangle",
                "name": "MazeMid_VertLeft",
                # Bounds: x=[-4.15, -4.0], y=[-4.0, 3.0]
                "bounds": [[-4.15, -4.0], [-3.85, 3.0]]
            },
            {
                "type": "rectangle",
                "name": "MazeMid_VertRight",
                # Bounds: x=[2.85, -3.0], y=[3.15, 4.0]
                "bounds": [[2.85, -3.0], [3.15, 4.0]]
            },
            {
                "type": "rectangle",
                "name": "MazeBottom_VertCenter",
                # Bounds: x=[-0.15, 0.15], y=[-10.0, -7.0]
                "bounds": [[-0.15, -10.0], [0.15, -7.0]]
            },
            {
                "type": "rectangle",
                "name": "MazeTop_VertCenter",
                # Bounds: x=[-0.65, -0.35], y=[7.0, 10.0]
                "bounds": [[-0.65, 7.0], [-0.35, 10.0]]
            }
        ]
    }
}

# -----------------------------------------------------
# Grid cell layer (modules + phase-based translations)
# -----------------------------------------------------
class GridCellLayer:
    """
    Module-based grid cells with phase-based translations (no translation ranges).
    Rotations are equally spaced per module; size & spread are fixed per module.
    """

    def __init__(
        self,
        num_modules: int = 8,
        cells_per_module: int = 50,
        size_range: Tuple[float, float] = (0.5, 0.5),
        spread_range: Tuple[float, float] = (1.2, 1.2),
        scale_multiplier: float = 5.0,
        frequency_divisor: float = 1.0,
        threshold: float = 0.7,
        threshold_type: str = 'soft',
        sparsity: Optional[float] = None,
        normalization: str = 'per-cell',
        local_group_size: int = 10,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.device = torch.device(device)
        self.dtype = dtype

        if num_modules <= 0 or cells_per_module <= 0:
            raise ValueError("num_modules and cells_per_module must be positive integers.")
        self.num_modules = int(num_modules)
        self.cells_per_module = int(cells_per_module)
        self.total_grid_cells = self.num_modules * self.cells_per_module

        # Store core params
        self.scale_multiplier = scale_multiplier
        self.frequency_divisor = frequency_divisor
        self.threshold = threshold
        self.threshold_type = threshold_type
        self.sparsity = sparsity
        self.normalization = normalization
        self.local_group_size = local_group_size

        if normalization == 'per-cell':
            self.cell_min = torch.ones(self.total_grid_cells, dtype=self.dtype, device=self.device) * -1.0
            self.cell_max = torch.ones(self.total_grid_cells, dtype=self.dtype, device=self.device) * 1.0
            self.min_max_updated = False

        torch.manual_seed(42)

        adjusted_size_range = (size_range[0] * scale_multiplier, size_range[1] * scale_multiplier)

        # Per-module rotations (equally spaced)
        rot_per_module = torch.linspace(0.0, 360.0, steps=self.num_modules + 1, dtype=self.dtype)[:-1]

        # Per-module size & spread (fixed within module)
        size_per_module   = torch.empty(self.num_modules, dtype=self.dtype).uniform_(*adjusted_size_range)
        spread_per_module = torch.empty(self.num_modules, dtype=self.dtype).uniform_(*spread_range)

        # Build per-cell params
        self.rotation_params = rot_per_module.repeat_interleave(self.cells_per_module).to(self.device)
        self.size_params     = size_per_module.repeat_interleave(self.cells_per_module).to(self.device)
        self.spread_params   = spread_per_module.repeat_interleave(self.cells_per_module).to(self.device)

        # Phase-based translations (uniform phases -> translations via K^{-1})
        x_trans, y_trans = [], []
        sobol = torch.quasirandom.SobolEngine(dimension=2, scramble=True)

        for m in range(self.num_modules):
            s = size_per_module[m].item()
            theta_deg = rot_per_module[m].item()
            theta = np.deg2rad(theta_deg)

            K = torch.tensor([[s, 0.0],
                              [s/2.0, (np.sqrt(3.0)*s)/2.0]], dtype=self.dtype)
            K_inv = torch.linalg.inv(K)

            u = sobol.draw(self.cells_per_module).to(self.dtype)       # [0,1)
            phi = 2.0 * np.pi * u                                      # [0,2π)

            # t_rot = K^{-1} phi
            K_inv_b = K_inv.unsqueeze(0).expand(self.cells_per_module, -1, -1)
            phi_b   = phi.unsqueeze(-1)
            t_rot   = torch.bmm(K_inv_b, phi_b).squeeze(-1)

            # world-frame rotation
            c, s_tr = np.cos(theta), np.sin(theta)
            R = torch.tensor([[ c, -s_tr],
                              [ s_tr,  c]], dtype=self.dtype)
            t_world = (R @ t_rot.T).T
            x_trans.append(t_world[:, 0])
            y_trans.append(t_world[:, 1])

        self.x_trans_params = torch.cat(x_trans, dim=0).to(self.device)
        self.y_trans_params = torch.cat(y_trans, dim=0).to(self.device)

        # Precompute trig
        theta_rad = torch.deg2rad(self.rotation_params)
        self.cos_theta = torch.cos(theta_rad)
        self.sin_theta = torch.sin(theta_rad)

    def _normalize_activations(self, activations: torch.Tensor) -> torch.Tensor:
        if self.normalization == 'none':
            return activations
        elif self.normalization == 'global':
            mn, mx = activations.min(), activations.max()
            return (activations - mn) / (mx - mn) if mx > mn else torch.ones_like(activations) * 0.5
        elif self.normalization == 'per-cell':
            if not getattr(self, "min_max_updated", True):
                self.cell_min = torch.minimum(self.cell_min, activations)
                self.cell_max = torch.maximum(self.cell_max, activations)
            rngs = self.cell_max - self.cell_min
            rngs = torch.where(rngs < 1e-8, torch.ones_like(rngs), rngs)
            return (activations - self.cell_min) / rngs
        elif self.normalization == 'local':
            n = activations.shape[0]
            gs = self.local_group_size
            normalized = torch.zeros_like(activations)
            for start in range(0, n, gs):
                end = min(start + gs, n)
                group = activations[start:end]
                mn, mx = group.min(), group.max()
                normalized[start:end] = (group - mn) / (mx - mn) if mx > mn else torch.ones_like(group) * 0.5
            return normalized
        else:
            return activations

    def _apply_threshold(self, activations: torch.Tensor) -> torch.Tensor:
        if self.threshold_type == 'none':
            return activations
        elif self.threshold_type == 'hard':
            return torch.where(activations >= self.threshold, activations, torch.zeros_like(activations))
        elif self.threshold_type == 'soft':
            scale = 1.0 / (1.0 - self.threshold + 1e-8)
            out = (activations - self.threshold) * scale
            return torch.clamp(out, 0.0, 1.0)
        else:
            return activations

    def _apply_sparsity(self, activations: torch.Tensor) -> torch.Tensor:
        if self.sparsity is None or self.sparsity <= 0.0:
            return activations
        k = max(1, int(self.sparsity * len(activations)))
        vals, _ = torch.topk(activations, k)
        cutoff = vals[-1]
        return torch.where(activations >= cutoff, activations, torch.zeros_like(activations))

    def get_grid_cell_activations(self, position: torch.Tensor) -> torch.Tensor:
        """
        position: [2] or [B,2]
        returns: [C] or [B,C] grid cell activations
        """
        position = position.to(self.device)
        if position.dim() == 1:
            position = position.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        B = position.shape[0]
        x, y = position[:, 0], position[:, 1]

        dx = x.unsqueeze(1) - self.x_trans_params.unsqueeze(0)
        dy = y.unsqueeze(1) - self.y_trans_params.unsqueeze(0)

        c, s = self.cos_theta.unsqueeze(0), self.sin_theta.unsqueeze(0)
        rx =  c * dx + s * dy
        ry = -s * dx + c * dy

        sz = self.size_params.unsqueeze(0)
        freq = (2.0 * np.pi) / (sz * self.frequency_divisor)

        z1 = torch.cos(freq * rx)
        z2 = torch.cos(freq * (rx / 2.0 + (np.sqrt(3.0) / 2.0) * ry))
        z3 = torch.cos(freq * (rx / 2.0 - (np.sqrt(3.0) / 2.0) * ry))
        raw = (z1 + z2 + z3) / 3.0

        # Use power law to shape peaks (NOT Gaussian envelope)
        spread = self.spread_params.unsqueeze(0)
        act = torch.sign(raw) * torch.pow(torch.abs(raw), 1 / spread)

        act = self._normalize_activations(act)
        act = self._apply_threshold(act)
        act = self._apply_sparsity(act)

        if squeeze:
            act = act.squeeze(0)
        return act


@dataclass
class GridCellParams:
    num_modules: int = 8
    cells_per_module: int = 50
    size_range: Tuple[float, float] = (0.5, 0.5)
    spread_range: Tuple[float, float] = (1.2, 1.2)
    scale_multiplier: float = 5.0
    frequency_divisor: float = 1.0
    threshold: float = 0.7
    threshold_type: str = 'soft'
    sparsity: Optional[float] = None
    normalization: str = 'per-cell'
    local_group_size: int = 10


class GridCellSimulator:
    def __init__(self, root):
        self.root = root
        self.root.title("Grid Cell Simulator with Blob-Aware Masking")
        self.root.geometry("1200x900")

        # World state
        self.world_name = "20x20_cross_multi_goal"
        self.world_w, self.world_h = WORLD_CONFIGS[self.world_name]["size"]
        self.world_obstacles = WORLD_CONFIGS[self.world_name]["obstacles"]

        # Grid layer
        self.grid_layer: Optional[GridCellLayer] = None
        self.params = GridCellParams()

        # Precomputed masks for efficiency
        self.blob_aware_masks = None  # Will be (H, W, C) numpy array
        self.resolution = 128  # Default resolution for plotting

        self.build_gui()
        self.initialize_grid_layer()

    def build_gui(self):
        main = ttk.Frame(self.root); main.pack(fill=tk.BOTH, expand=True)

        # Left controls
        left = ttk.Frame(main, width=250); left.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        row = 0

        # World selector
        ttk.Label(left, text="World:").grid(row=row, column=0, sticky=tk.W); row+=1
        self.world_var = tk.StringVar(value=self.world_name)
        cb = ttk.Combobox(left, textvariable=self.world_var, values=list(WORLD_CONFIGS.keys()), state='readonly')
        cb.grid(row=row, column=0, columnspan=2, sticky=tk.EW, pady=5); row+=1
        cb.bind("<<ComboboxSelected>>", self.on_world_change)

        ttk.Separator(left, orient=tk.HORIZONTAL).grid(row=row, column=0, columnspan=2, sticky=tk.EW, pady=10); row+=1

        # Grid layer params
        self.num_modules_var = tk.StringVar(value=str(self.params.num_modules))
        self.cells_per_module_var = tk.StringVar(value=str(self.params.cells_per_module))
        self.size_min_var = tk.StringVar(value=str(self.params.size_range[0]))
        self.size_max_var = tk.StringVar(value=str(self.params.size_range[1]))
        self.spread_min_var = tk.StringVar(value=str(self.params.spread_range[0]))
        self.spread_max_var = tk.StringVar(value=str(self.params.spread_range[1]))
        self.scale_multiplier_var = tk.StringVar(value=str(self.params.scale_multiplier))
        self.frequency_divisor_var = tk.StringVar(value=str(self.params.frequency_divisor))
        self.threshold_var = tk.StringVar(value=str(self.params.threshold))
        self.threshold_type_var = tk.StringVar(value=self.params.threshold_type)
        self.sparsity_var = tk.StringVar(value="")
        self.norm_var = tk.StringVar(value=self.params.normalization)
        self.local_group_size_var = tk.StringVar(value=str(self.params.local_group_size))
        self.sel_cell_var = tk.StringVar(value="0")

        # Display params
        ttk.Label(left, text="Num Modules:").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(left, textvariable=self.num_modules_var, width=10).grid(row=row, column=1, sticky=tk.W, pady=2); row+=1
        ttk.Label(left, text="Cells/Module:").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(left, textvariable=self.cells_per_module_var, width=10).grid(row=row, column=1, sticky=tk.W, pady=2); row+=1
        ttk.Label(left, text="Size Min:").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(left, textvariable=self.size_min_var, width=10).grid(row=row, column=1, sticky=tk.W, pady=2); row+=1
        ttk.Label(left, text="Size Max:").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(left, textvariable=self.size_max_var, width=10).grid(row=row, column=1, sticky=tk.W, pady=2); row+=1
        ttk.Label(left, text="Spread Min:").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(left, textvariable=self.spread_min_var, width=10).grid(row=row, column=1, sticky=tk.W, pady=2); row+=1
        ttk.Label(left, text="Spread Max:").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(left, textvariable=self.spread_max_var, width=10).grid(row=row, column=1, sticky=tk.W, pady=2); row+=1
        ttk.Label(left, text="Scale Mult:").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(left, textvariable=self.scale_multiplier_var, width=10).grid(row=row, column=1, sticky=tk.W, pady=2); row+=1
        ttk.Label(left, text="Freq Div:").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(left, textvariable=self.frequency_divisor_var, width=10).grid(row=row, column=1, sticky=tk.W, pady=2); row+=1
        ttk.Label(left, text="Threshold:").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(left, textvariable=self.threshold_var, width=10).grid(row=row, column=1, sticky=tk.W, pady=2); row+=1
        ttk.Label(left, text="Threshold Type:").grid(row=row, column=0, sticky=tk.W)
        ttk.Combobox(left, textvariable=self.threshold_type_var, values=['none','hard','soft'], state='readonly', width=8).grid(row=row, column=1, sticky=tk.W, pady=2); row+=1
        ttk.Label(left, text="Sparsity (opt):").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(left, textvariable=self.sparsity_var, width=10).grid(row=row, column=1, sticky=tk.W, pady=2); row+=1
        ttk.Label(left, text="Normalization:").grid(row=row, column=0, sticky=tk.W)
        ttk.Combobox(left, textvariable=self.norm_var, values=['none','global','per-cell','local'], state='readonly', width=8).grid(row=row, column=1, sticky=tk.W, pady=2); row+=1
        ttk.Label(left, text="Local Group:").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(left, textvariable=self.local_group_size_var, width=10).grid(row=row, column=1, sticky=tk.W, pady=2); row+=1

        ttk.Separator(left, orient=tk.HORIZONTAL).grid(row=row, column=0, columnspan=2, sticky=tk.EW, pady=10); row+=1

        ttk.Label(left, text="Selected Cell:").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(left, textvariable=self.sel_cell_var, width=10).grid(row=row, column=1, sticky=tk.W, pady=5); row+=1

        ttk.Button(left, text="Update Simulation", command=self.update_simulation).grid(row=row, column=0, columnspan=2, pady=10)

        # Right plots
        right = ttk.Frame(main); right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.single_frame = ttk.LabelFrame(right, text="Single Grid Cell (blob-aware masked)")
        self.single_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.avg_frame = ttk.LabelFrame(right, text="Average Activation (blob-aware masked)")
        self.avg_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.fig_single = Figure(figsize=(6, 4), dpi=100)
        self.canvas_single = FigureCanvasTkAgg(self.fig_single, master=self.single_frame)
        self.canvas_single.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.fig_avg = Figure(figsize=(6, 4), dpi=100)
        self.canvas_avg = FigureCanvasTkAgg(self.fig_avg, master=self.avg_frame)
        self.canvas_avg.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.tb1 = NavigationToolbar2Tk(self.canvas_single, self.single_frame); self.tb1.update()
        self.tb2 = NavigationToolbar2Tk(self.canvas_avg, self.avg_frame); self.tb2.update()

    # ---------- World change ----------
    def on_world_change(self, _evt=None):
        name = self.world_var.get()
        if name in WORLD_CONFIGS:
            self.world_name = name
            self.world_w, self.world_h = WORLD_CONFIGS[name]["size"]
            self.world_obstacles = WORLD_CONFIGS[name]["obstacles"]
            # Re-run plots with new world immediately
            self.precompute_blob_aware_masks()
            self.update_plots()

    # ---------- Params & layer ----------
    def read_params(self) -> bool:
        try:
            num_modules = int(self.num_modules_var.get())
            cells_per_module = int(self.cells_per_module_var.get())
            size_range = (float(self.size_min_var.get()), float(self.size_max_var.get()))
            spread_range = (float(self.spread_min_var.get()), float(self.spread_max_var.get()))
            scale_multiplier = float(self.scale_multiplier_var.get())
            frequency_divisor = float(self.frequency_divisor_var.get())
            threshold = float(self.threshold_var.get())
            threshold_type = self.threshold_type_var.get()
            sparsity = float(self.sparsity_var.get()) if self.sparsity_var.get() else None
            normalization = self.norm_var.get()
            local_group = int(self.local_group_size_var.get())

            self.params = GridCellParams(
                num_modules=num_modules,
                cells_per_module=cells_per_module,
                size_range=size_range,
                spread_range=spread_range,
                scale_multiplier=scale_multiplier,
                frequency_divisor=frequency_divisor,
                threshold=threshold,
                threshold_type=threshold_type,
                sparsity=sparsity,
                normalization=normalization,
                local_group_size=local_group
            )
            return True
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))
            return False

    def initialize_grid_layer(self):
        self.grid_layer = GridCellLayer(
            num_modules=self.params.num_modules,
            cells_per_module=self.params.cells_per_module,
            size_range=self.params.size_range,
            spread_range=self.params.spread_range,
            scale_multiplier=self.params.scale_multiplier,
            frequency_divisor=self.params.frequency_divisor,
            threshold=self.params.threshold,
            threshold_type=self.params.threshold_type,
            sparsity=self.params.sparsity,
            normalization=self.params.normalization,
            local_group_size=self.params.local_group_size,
        )
        self.precompute_blob_aware_masks()
        self.update_plots()

    def update_simulation(self):
        if self.read_params():
            self.initialize_grid_layer()

    # ---------- Blob-aware mask builder ----------
    def build_obstacle_mask(self, resolution: int) -> np.ndarray:
        """
        Returns a (H,W) mask with 1.0 in free space and 0.0 in obstacles.
        Coordinates are centered: x ∈ [-W/2, W/2], y ∈ [-H/2, H/2].
        """
        xs = np.linspace(-self.world_w/2, self.world_w/2, resolution)
        ys = np.linspace(-self.world_h/2, self.world_h/2, resolution)
        X, Y = np.meshgrid(xs, ys)
        mask = np.ones_like(X, dtype=np.float32)

        for obs in self.world_obstacles:
            if obs.get("type") == "rectangle":
                (x1, y1), (x2, y2) = obs["bounds"]
                xmin, xmax = min(x1, x2), max(x1, x2)
                ymin, ymax = min(y1, y2), max(y1, y2)
                inside = (X >= xmin) & (X <= xmax) & (Y >= ymin) & (Y <= ymax)
                mask[inside] = 0.0

        return mask  # (H,W)

    def precompute_blob_aware_masks(self):
        """
        Precompute blob-aware masks for all grid cells.
        Each grid cell has a hexagonal tiling pattern of activation blobs.
        We process each blob independently to mask only the parts that straddle walls.
        """
        if self.grid_layer is None:
            return

        print("\n" + "="*60)
        print("Precomputing blob-aware masks...")
        print("="*60)
        
        resolution = self.resolution
        xs = np.linspace(-self.world_w/2, self.world_w/2, resolution)
        ys = np.linspace(-self.world_h/2, self.world_h/2, resolution)
        X, Y = np.meshgrid(xs, ys)

        # Build base obstacle mask
        obstacle_mask = self.build_obstacle_mask(resolution)  # (H,W) in {0,1}

        # Compute all activations
        C = self.grid_layer.total_grid_cells
        activ = np.zeros((resolution, resolution, C), dtype=np.float32)
        
        print(f"\nStep 1/2: Computing activations for {C} grid cells...")
        for i in range(resolution):
            if i % 20 == 0:
                print(f"  Progress: {100*i//resolution}% ({i}/{resolution} rows)")
            for j in range(resolution):
                pos = torch.tensor([X[i, j], Y[i, j]], dtype=torch.float32)
                a = self.grid_layer.get_grid_cell_activations(pos).cpu().numpy()
                activ[i, j, :] = a

        # Initialize blob-aware masks - start with all ones, we'll only mask specific regions
        # We'll apply obstacle mask at the very end
        self.blob_aware_masks = np.ones((resolution, resolution, C), dtype=np.float32)

        print(f"\nStep 2/2: Processing individual blobs for each grid cell...")
        # For each grid cell, identify blobs in the tiling pattern and handle wall intersections
        for cell_idx in range(C):
            if cell_idx % 50 == 0:
                print(f"  Progress: {100*cell_idx//C}% (cell {cell_idx}/{C})")
            
            cell_activ = activ[:, :, cell_idx]
            
            # Threshold to identify significant activations (blobs in the hexagonal pattern)
            max_activ = np.max(cell_activ)
            if max_activ == 0:
                continue
                
            # Use a threshold to capture blob boundaries
            threshold = 0.15 * max_activ
            binary = cell_activ > threshold
            
            if not np.any(binary):
                continue
            
            # Label connected components (individual blobs in the hexagonal tiling pattern)
            # IMPORTANT: We need to label on the FULL activation, not split by obstacles yet
            labeled, num_blobs = ndimage.label(binary)
            
            # Process each blob in the tiling pattern independently
            for blob_id in range(1, num_blobs + 1):
                blob_mask = (labeled == blob_id)
                
                # Check if this specific blob intersects with any obstacles
                blob_in_free = blob_mask & (obstacle_mask == 1)
                blob_in_obstacle = blob_mask & (obstacle_mask == 0)
                
                # Only process if blob actually straddles a wall
                # (has pixels in both free space AND obstacle space)
                if np.any(blob_in_free) and np.any(blob_in_obstacle):
                    # This blob straddles a wall - split it and keep only the stronger side
                    blob_mask_to_keep = self.process_single_blob_with_obstacles(
                        blob_mask, obstacle_mask, cell_activ
                    )
                    # Mask out the weaker parts of this blob
                    blob_to_mask = blob_mask & ~blob_mask_to_keep
                    self.blob_aware_masks[:, :, cell_idx][blob_to_mask] = 0.0
                
                # If blob is entirely in free space: keep it (do nothing)
                # If blob is entirely in obstacle: will be masked by obstacle_mask later
        
        # Finally, apply obstacle mask to ensure walls are always masked
        self.blob_aware_masks *= obstacle_mask[:, :, np.newaxis]

        print("\n" + "="*60)
        print("Blob-aware mask precomputation complete!")
        print("="*60 + "\n")

    def process_single_blob_with_obstacles(self, blob_mask, obstacle_mask, activation):
        """
        Process a single blob that intersects with obstacles.
        Keep activation on the side with more total activation, mask the rest.
        
        This operates only within the region of this specific blob.
        
        Args:
            blob_mask: Boolean array indicating this blob's location
            obstacle_mask: Float array (1=free, 0=obstacle)
            activation: Float array with activation values
            
        Returns:
            Boolean mask indicating which parts of this blob to keep
        """
        # Split blob by obstacles using connected components on free space only
        free_blob = blob_mask & (obstacle_mask == 1)
        labeled_free, num_regions = ndimage.label(free_blob)
        
        if num_regions <= 1:
            # No splitting occurred (blob doesn't actually straddle a wall, or is entirely in obstacle)
            return free_blob
        
        # Calculate total activation in each region
        region_activations = []
        for region_id in range(1, num_regions + 1):
            region_mask = (labeled_free == region_id)
            total_activation = np.sum(activation[region_mask])
            region_activations.append((region_id, total_activation))
        
        # Find region with maximum activation
        if region_activations:
            best_region_id = max(region_activations, key=lambda x: x[1])[0]
            # Keep only the best region (the side of the wall with more activation)
            return labeled_free == best_region_id
        else:
            return free_blob

    # ---------- Plotting ----------
    def update_plots(self):
        if self.grid_layer is None or self.blob_aware_masks is None:
            return

        self.fig_single.clear()
        self.fig_avg.clear()

        resolution = self.resolution
        xs = np.linspace(-self.world_w/2, self.world_w/2, resolution)
        ys = np.linspace(-self.world_h/2, self.world_h/2, resolution)
        X, Y = np.meshgrid(xs, ys)

        # Compute activations (H,W,C)
        C = self.grid_layer.total_grid_cells
        activ = np.zeros((resolution, resolution, C), dtype=np.float32)
        for i in range(resolution):
            for j in range(resolution):
                pos = torch.tensor([X[i, j], Y[i, j]], dtype=torch.float32)
                a = self.grid_layer.get_grid_cell_activations(pos).cpu().numpy()
                activ[i, j, :] = a

        # Apply blob-aware masks
        activ_masked = activ * self.blob_aware_masks

        # ---- single cell ----
        try:
            idx = int(self.sel_cell_var.get())
            if idx < 0 or idx >= C: idx = 0
        except ValueError:
            idx = 0

        ax1 = self.fig_single.add_subplot(111)
        im1 = ax1.imshow(
            activ_masked[:, :, idx],
            extent=[-self.world_w/2, self.world_w/2, -self.world_h/2, self.world_h/2],
            origin='lower', cmap='viridis', interpolation='bilinear'
        )
        ax1.set_xlabel('X'); ax1.set_ylabel('Y')
        m_idx = idx // self.grid_layer.cells_per_module
        ax1.set_title(f"Grid Cell {idx} (Module {m_idx}) — blob-aware masked")
        self.fig_single.colorbar(im1, ax=ax1, label='Activation')

        # Draw obstacle outlines
        self.draw_obstacles(ax1)

        # ---- average ----
        avg = np.mean(activ_masked, axis=2)

        ax2 = self.fig_avg.add_subplot(111)
        im2 = ax2.imshow(
            avg,
            extent=[-self.world_w/2, self.world_w/2, -self.world_h/2, self.world_h/2],
            origin='lower', cmap='viridis', interpolation='bilinear'
        )
        ax2.set_xlabel('X'); ax2.set_ylabel('Y')
        ax2.set_title("Average Grid Cell Activation (blob-aware masked)")
        self.fig_avg.colorbar(im2, ax=ax2, label='Activation')

        # Obstacle outlines
        self.draw_obstacles(ax2)

        self.fig_single.tight_layout(); self.fig_avg.tight_layout()
        self.canvas_single.draw(); self.canvas_avg.draw()

    def draw_obstacles(self, ax):
        for obs in self.world_obstacles:
            if obs.get("type") == "rectangle":
                (x1, y1), (x2, y2) = obs["bounds"]
                xmin, xmax = min(x1, x2), max(x1, x2)
                ymin, ymax = min(y1, y2), max(y1, y2)
                rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                         linewidth=1.0, edgecolor='k', facecolor='none')
                ax.add_patch(rect)


if __name__ == "__main__":
    root = tk.Tk()
    app = GridCellSimulator(root)
    root.mainloop()