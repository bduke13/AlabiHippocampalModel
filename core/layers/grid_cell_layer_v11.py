import torch
import numpy as np
from typing import Optional, Tuple

try:
    from scipy import ndimage as _ndimage
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

from core.robot.webots_worlds import get_world_config


class GridCellLayer:
    """
    Module-based grid cells with phase-based translations and one-time blob-aware masking.

    - Uses frequency-based formulation: freq = 2π / (size / frequency_divisor),
      applied in the cosine triads as cos(freq * x).
    - Adds module grouping and phase-based translations per visualizations/grid_cell_simulator_v6.py.
    - Computes a world-aware blob mask once at initialization and applies it per-position with
      minimal overhead (single index + elementwise multiply).
    - Keeps only per-cell normalization and soft threshold behaviors (as used by controllers).
    """

    def __init__(
        self,
        num_modules: int,
        cells_per_module: int,
        size_range: Tuple[float, float] = (0.5, 0.5),
        spread_range: Tuple[float, float] = (1.2, 1.2),
        scale_multiplier: float = 1.0,
        frequency_divisor: float = 1.0,
        threshold: float = 0.7,
        threshold_type: str = 'soft',
        normalization: str = 'per-cell',
        world_name: Optional[str] = None,
        mask_resolution: int = 128,
        obstacle_dilation: float = 0.2,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        # Core device/dtype
        self.device = torch.device(device)
        self.dtype = dtype

        # Modules/cells
        if num_modules <= 0 or cells_per_module <= 0:
            raise ValueError("num_modules and cells_per_module must be positive integers.")
        self.num_modules = int(num_modules)
        self.cells_per_module = int(cells_per_module)
        self.total_grid_cells = self.num_modules * self.cells_per_module

        # Parameters
        self.scale_multiplier = float(scale_multiplier)
        self.frequency_divisor = float(frequency_divisor)
        self.threshold = float(threshold)
        self.threshold_type = threshold_type
        self.normalization = normalization

        # Per-cell normalization params
        if self.normalization == 'per-cell':
            self.cell_min = torch.ones(self.total_grid_cells, dtype=self.dtype, device=self.device) * -1.0
            self.cell_max = torch.ones(self.total_grid_cells, dtype=self.dtype, device=self.device) * 1.0
            self.min_max_updated = False

        # Build per-module parameters: equally spaced rotations; size/spread fixed per module
        torch.manual_seed(42)

        # Apply scale multiplier to size range
        adjusted_size_range = (size_range[0] * self.scale_multiplier, size_range[1] * self.scale_multiplier)

        rot_per_module = torch.linspace(0.0, 360.0, steps=self.num_modules + 1, dtype=self.dtype)[:-1]
        size_per_module = torch.empty(self.num_modules, dtype=self.dtype).uniform_(*adjusted_size_range)
        spread_per_module = torch.empty(self.num_modules, dtype=self.dtype).uniform_(*spread_range)

        # Expand to per-cell tensors on device
        self.rotation_params = rot_per_module.repeat_interleave(self.cells_per_module).to(self.device)
        self.size_params = size_per_module.repeat_interleave(self.cells_per_module).to(self.device)
        self.spread_params = spread_per_module.repeat_interleave(self.cells_per_module).to(self.device)

        # Phase-based translations: t = R(theta) K^{-1} phi
        # Use base size (before scaling) for K to keep translation range independent of grid scale
        x_trans, y_trans = [], []
        sobol = torch.quasirandom.SobolEngine(dimension=2, scramble=True)
        base_size = (size_range[0] + size_range[1]) / 2.0  # Average of size_range before scaling

        for m in range(self.num_modules):
            s = size_per_module[m].item()
            theta_deg = rot_per_module[m].item()
            theta = float(np.deg2rad(theta_deg))

            # K constructed from base size to keep translations small and independent of scale
            K = torch.tensor([[base_size, 0.0],
                              [base_size / 2.0, (np.sqrt(3.0) * base_size) / 2.0]], dtype=self.dtype)
            K_inv = torch.linalg.inv(K)

            # Uniform phases in [0, 2pi)
            u = sobol.draw(self.cells_per_module).to(self.dtype)
            phi = 2.0 * np.pi * u

            K_inv_b = K_inv.unsqueeze(0).expand(self.cells_per_module, -1, -1)
            phi_b = phi.unsqueeze(-1)
            t_rot = torch.bmm(K_inv_b, phi_b).squeeze(-1)  # module frame

            # Rotate to world frame by module rotation theta
            c, s = float(np.cos(theta)), float(np.sin(theta))
            R = torch.tensor([[c, -s], [s, c]], dtype=self.dtype)
            t_world = (R @ t_rot.T).T
            x_trans.append(t_world[:, 0])
            y_trans.append(t_world[:, 1])

        self.x_trans_params = torch.cat(x_trans, dim=0).to(self.device)
        self.y_trans_params = torch.cat(y_trans, dim=0).to(self.device)

        # Precompute rotation trig per cell
        theta_rad = torch.deg2rad(self.rotation_params)
        self.cos_theta = torch.cos(theta_rad)
        self.sin_theta = torch.sin(theta_rad)

        # World/mask metadata
        self.world = get_world_config(world_name) if world_name is not None else None
        self.world_w, self.world_h = (self.world["size"] if self.world else [20.0, 20.0])
        self.world_obstacles = (self.world["obstacles"] if self.world else [])
        self.mask_resolution = int(mask_resolution)
        self.obstacle_dilation = float(obstacle_dilation)
        self._build_blob_mask_once()

    # ---------------------
    # Internal helpers
    # ---------------------
    def _normalize_activations(self, activations: torch.Tensor) -> torch.Tensor:
        if self.normalization == 'per-cell':
            if not getattr(self, "min_max_updated", True):
                self.cell_min = torch.minimum(self.cell_min, activations)
                self.cell_max = torch.maximum(self.cell_max, activations)
            rng = self.cell_max - self.cell_min
            rng = torch.where(rng > 1e-8, rng, torch.ones_like(rng))
            out = (activations - self.cell_min) / rng
            return torch.clamp(out, 0.0, 1.0)
        else:
            return activations

    def _apply_soft_threshold(self, activations: torch.Tensor) -> torch.Tensor:
        if self.threshold_type != 'soft':
            return activations
        # Linear soft threshold scaled to [0,1]
        scale = 1.0 / (1.0 - self.threshold + 1e-8)
        out = (activations - self.threshold) * scale
        return torch.clamp(out, 0.0, 1.0)

    def _build_obstacle_mask(self, resolution: int) -> np.ndarray:
        xs = np.linspace(-self.world_w / 2.0, self.world_w / 2.0, resolution)
        ys = np.linspace(-self.world_h / 2.0, self.world_h / 2.0, resolution)
        X, Y = np.meshgrid(xs, ys)
        mask = np.ones_like(X, dtype=np.float32)
        for obs in self.world_obstacles:
            if obs.get("type") == "rectangle":
                (x1, y1), (x2, y2) = obs["bounds"]
                xmin, xmax = min(x1, x2), max(x1, x2)
                ymin, ymax = min(y1, y2), max(y1, y2)
                inside = (X >= xmin) & (X <= xmax) & (Y >= ymin) & (Y <= ymax)
                mask[inside] = 0.0
        # Optional dilation to ensure thin walls fully separate blobs when resolution is low
        if self.obstacle_dilation > 0.0 and _HAS_SCIPY:
            # Convert dilation (meters) to approximately that many pixels along each axis
            px_w = int(np.ceil(self.obstacle_dilation * (resolution / self.world_w)))
            px_h = int(np.ceil(self.obstacle_dilation * (resolution / self.world_h)))
            px = max(px_w, px_h)
            if px > 0:
                mask = 1.0 - _ndimage.binary_dilation(1.0 - mask, iterations=px)
        return mask

    def _compute_activations_grid(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Compute activations over a grid for all cells (no mask applied)."""
        H, W = X.shape
        C = self.total_grid_cells
        activ = np.zeros((H, W, C), dtype=np.float32)
        # Vectorized per-row computation to reduce Python overhead
        for i in range(H):
            x_row = torch.tensor(X[i, :], dtype=self.dtype, device=self.device)  # [W]
            y_row = torch.tensor(Y[i, :], dtype=self.dtype, device=self.device)
            # Broadcast over cells: [W,1] vs [C]
            dx = x_row.unsqueeze(1) - self.x_trans_params.unsqueeze(0)  # [W,C]
            dy = y_row.unsqueeze(1) - self.y_trans_params.unsqueeze(0)  # [W,C]
            c, s = self.cos_theta.unsqueeze(0), self.sin_theta.unsqueeze(0)
            rx = c * dx + s * dy
            ry = -s * dx + c * dy
            sz = (self.size_params / self.frequency_divisor).unsqueeze(0)  # [1,C]
            freq = (2.0 * np.pi) / sz
            z1 = torch.cos(freq * rx)
            z2 = torch.cos(freq * (rx / 2.0 + (np.sqrt(3.0) / 2.0) * ry))
            z3 = torch.cos(freq * (rx / 2.0 - (np.sqrt(3.0) / 2.0) * ry))
            z = (z1 + z2 + z3) / 3.0
            spread = self.spread_params.unsqueeze(0)
            a_row = torch.sign(z) * torch.pow(torch.abs(z), 1.0 / spread)
            # Per-cell normalization (running min/max) not applied here; we just return raw
            activ[i, :, :] = a_row.detach().cpu().numpy()
        return activ

    def _process_single_blob_with_obstacles(self, blob_mask: np.ndarray, obstacle_mask: np.ndarray, activation: np.ndarray) -> np.ndarray:
        # Split blob by obstacles using connected components on free space only
        free_blob = blob_mask & (obstacle_mask == 1)
        if not _HAS_SCIPY:
            # Fallback: keep free portion without further splitting
            return free_blob
        labeled_free, num_regions = _ndimage.label(free_blob)
        if num_regions <= 1:
            return free_blob
        best_region = None
        best_sum = -1.0
        for region_id in range(1, num_regions + 1):
            region_mask = (labeled_free == region_id)
            s = float(activation[region_mask].sum())
            if s > best_sum:
                best_sum = s
                best_region = region_mask
        return best_region if best_region is not None else free_blob

    def _build_blob_mask_once(self) -> None:
        # Precompute per-cell blob-aware mask over the world grid
        H = W = int(self.mask_resolution)
        xs = np.linspace(-self.world_w / 2.0, self.world_w / 2.0, W)
        ys = np.linspace(-self.world_h / 2.0, self.world_h / 2.0, H)
        X, Y = np.meshgrid(xs, ys)
        obstacle_mask = self._build_obstacle_mask(self.mask_resolution)  # (H,W) in {0,1}

        # Compute activations grid (no mask applied)
        activ = self._compute_activations_grid(X, Y)  # (H,W,C)

        # Start with ones; we will zero out parts of blobs that straddle obstacles
        blob_masks = np.ones_like(activ, dtype=np.float32)

        # If SciPy unavailable, fallback to global obstacle masking only
        if not _HAS_SCIPY:
            # Debug: surface the silent fallback so we can see it in Webots logs
            self._mask_mode = "fallback_no_scipy"
            print("[GridCellLayer] WARNING: SciPy not available; using obstacle-only mask (no blob splitting).###############################################################################################")
            blob_masks *= obstacle_mask[:, :, None]
            self._mask = torch.tensor(blob_masks, dtype=self.dtype, device=self.device)
            # Precompute index transforms
            self._x_scale = (W - 1) / self.world_w
            self._y_scale = (H - 1) / self.world_h
            print(f"[GridCellLayer] Built blob-aware mask using mode={self._mask_mode}, resolution={self.mask_resolution} (SciPy available={_HAS_SCIPY}).################################################################")
            return

        # Full blob-aware processing
        self._mask_mode = "scipy_blob_split"
        C = activ.shape[2]
        for c in range(C):
            cell_activ = activ[:, :, c]
            m = float(cell_activ.max())
            if m <= 0.0:
                continue
            thresh = 0.15 * m
            binary = (cell_activ > thresh)
            if not binary.any():
                continue
            labeled, num_blobs = _ndimage.label(binary)
            if num_blobs == 0:
                continue
            for blob_id in range(1, num_blobs + 1):
                blob = (labeled == blob_id)
                blob_in_free = blob & (obstacle_mask == 1)
                blob_in_obs = blob & (obstacle_mask == 0)
                if blob_in_free.any() and blob_in_obs.any():
                    keep_mask = self._process_single_blob_with_obstacles(blob, obstacle_mask, cell_activ)
                    zero_region = blob & (~keep_mask)
                    blob_masks[:, :, c][zero_region] = 0.0

        # Always enforce obstacles as zero
        blob_masks *= obstacle_mask[:, :, None]

        # Store mask tensor on device
        self._mask = torch.tensor(blob_masks, dtype=self.dtype, device=self.device)

        # Precompute continuous->index transforms
        self._x_scale = (W - 1) / self.world_w
        self._y_scale = (H - 1) / self.world_h

        # Debug: indicate which masking path was used
        print(f"[GridCellLayer] Built blob-aware mask using mode={self._mask_mode}, resolution={self.mask_resolution} (SciPy available={_HAS_SCIPY}).")

    # ---------------------
    # Public API
    # ---------------------
    def get_grid_cell_activations(
        self,
        position,
        threshold: Optional[float] = None,
        threshold_type: Optional[str] = None,
        sparsity: Optional[float] = None,
        normalization: Optional[str] = None,
        *,
        use_mask: bool = True,
    ) -> torch.Tensor:
        """Compute grid cell activations at a position and apply one-time mask.

        Only per-cell normalization and soft threshold are supported (others are ignored).
        """
        # Resolve params
        thr = self.threshold if threshold is None else float(threshold)
        thr_type = self.threshold_type if threshold_type is None else threshold_type
        norm = self.normalization if normalization is None else normalization

        # Position tensor [x,y]
        if not isinstance(position, torch.Tensor):
            position = torch.tensor(position, dtype=self.dtype)
        position = position.to(self.device)
        if position.dim() > 1:
            position = position.squeeze()
        x, y = position[0], position[1]

        # Translate and rotate for all cells
        dx = x - self.x_trans_params
        dy = y - self.y_trans_params
        rx = self.cos_theta * dx + self.sin_theta * dy
        ry = -self.sin_theta * dx + self.cos_theta * dy

        sz = self.size_params / self.frequency_divisor
        freq = (2.0 * np.pi) / sz
        z1 = torch.cos(freq * rx)
        z2 = torch.cos(freq * (rx / 2.0 + (np.sqrt(3.0) / 2.0) * ry))
        z3 = torch.cos(freq * (rx / 2.0 - (np.sqrt(3.0) / 2.0) * ry))
        z = (z1 + z2 + z3) / 3.0

        spread = self.spread_params
        acts = torch.sign(z) * torch.pow(torch.abs(z), 1.0 / spread)

        # Normalization
        if norm == 'per-cell':
            if not getattr(self, "min_max_updated", True):
                self.cell_min = torch.minimum(self.cell_min, acts)
                self.cell_max = torch.maximum(self.cell_max, acts)
            rng = self.cell_max - self.cell_min
            rng = torch.where(rng > 1e-8, rng, torch.ones_like(rng))
            acts = (acts - self.cell_min) / rng
            acts = torch.clamp(acts, 0.0, 1.0)

        # Soft threshold only (if requested)
        if thr_type == 'soft':
            scale = 1.0 / (1.0 - thr + 1e-8)
            acts = torch.clamp((acts - thr) * scale, 0.0, 1.0)

        # Apply per-position mask slice
        if use_mask and hasattr(self, "_mask") and self._mask is not None:
            # Nearest-neighbor index into mask grid
            xi = int(round((float(x.item()) + self.world_w / 2.0) * self._x_scale))
            yi = int(round((float(y.item()) + self.world_h / 2.0) * self._y_scale))
            xi = max(0, min(self.mask_resolution - 1, xi))
            yi = max(0, min(self.mask_resolution - 1, yi))
            acts = acts * self._mask[yi, xi, :]

        return acts
