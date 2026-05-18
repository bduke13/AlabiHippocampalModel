import torch
import numpy as np
from typing import Optional, Tuple

try:
    from scipy import ndimage as _ndimage
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

from core.robot.webots_worlds import (
    expanded_obstacle_corners,
    get_world_config,
    obstacle_contains_points,
)


class GridCellLayer:
    """
    Grid cell layer with advanced obstacle-aware masking and edge smoothing.

    Features:
    - Frequency-based grid code with module rotations and Sobol phase translations
    - Obstacle-only masking with wall-based blob splitting (steps 2+3)
    - Morphological smoothing to restore biological circular shapes (sigma=2.0)
    - Compatible with v11 API for drop-in replacement in existing models
    """

    def __init__(
        self,
        num_modules: int,
        cells_per_module: int,
        spread_range: Tuple[float, float] = (1.2, 1.2),
        scale_multiplier: float = 1.0,
        translation_scale: float = 1.0,
        threshold: float = 0.7,
        threshold_type: str = "soft",
        normalization: str = "per-cell",
        world_name: Optional[str] = None,
        mask_resolution: int = 128,
        wall_split_thresh: float = 0.2,
        smooth_sigma: float = 1.5,
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

        self.scale_multiplier = float(scale_multiplier)
        self.translation_scale = float(translation_scale)
        self.threshold = float(threshold)
        self.threshold_type = threshold_type
        self.normalization = normalization
        self.wall_split_thresh = float(wall_split_thresh)
        self.smooth_sigma = float(smooth_sigma)

        if self.normalization == "per-cell":
            self.cell_min = torch.ones(self.total_grid_cells, dtype=self.dtype, device=self.device) * -1.0
            self.cell_max = torch.ones(self.total_grid_cells, dtype=self.dtype, device=self.device) * 1.0
            self.min_max_updated = False

        # Module params
        torch.manual_seed(42)
        rot_per_module = torch.linspace(0.0, 360.0, steps=self.num_modules + 1, dtype=self.dtype)[:-1]
        size_per_module = torch.full((self.num_modules,), self.scale_multiplier, dtype=self.dtype)
        spread_per_module = torch.empty(self.num_modules, dtype=self.dtype).uniform_(*spread_range)

        self.rotation_params = rot_per_module.repeat_interleave(self.cells_per_module).to(self.device)
        self.size_params = size_per_module.repeat_interleave(self.cells_per_module).to(self.device)
        self.spread_params = spread_per_module.repeat_interleave(self.cells_per_module).to(self.device)

        # Phase translations
        x_trans, y_trans = [], []
        sobol = torch.quasirandom.SobolEngine(dimension=2, scramble=True)
        base_size = 1.0
        for m in range(self.num_modules):
            theta_deg = rot_per_module[m].item()
            theta = float(np.deg2rad(theta_deg))
            K = torch.tensor([[base_size, 0.0], [base_size / 2.0, (np.sqrt(3.0) * base_size) / 2.0]], dtype=self.dtype)
            K_inv = torch.linalg.inv(K)
            u = sobol.draw(self.cells_per_module).to(self.dtype)
            phi = 2.0 * np.pi * u
            K_inv_b = K_inv.unsqueeze(0).expand(self.cells_per_module, -1, -1)
            phi_b = phi.unsqueeze(-1)
            t_rot = torch.bmm(K_inv_b, phi_b).squeeze(-1)
            c, s = float(np.cos(theta)), float(np.sin(theta))
            R = torch.tensor([[c, -s], [s, c]], dtype=self.dtype)
            t_world = (R @ t_rot.T).T
            x_trans.append(t_world[:, 0])
            y_trans.append(t_world[:, 1])

        self.x_trans_params = (torch.cat(x_trans, dim=0) * self.translation_scale).to(self.device)
        self.y_trans_params = (torch.cat(y_trans, dim=0) * self.translation_scale).to(self.device)

        theta_rad = torch.deg2rad(self.rotation_params)
        self.cos_theta = torch.cos(theta_rad)
        self.sin_theta = torch.sin(theta_rad)

        # World/mask
        self.world = get_world_config(world_name) if world_name is not None else None
        self.world_w, self.world_h = (self.world["size"] if self.world else [20.0, 20.0])
        self.world_obstacles = (self.world["obstacles"] if self.world else [])
        self.mask_resolution = int(mask_resolution)
        self._build_advanced_mask()

    def _build_obstacle_mask(self) -> np.ndarray:
        """Build obstacle-only mask (free space = 1, obstacles = 0)."""
        xs = np.linspace(-self.world_w / 2.0, self.world_w / 2.0, self.mask_resolution)
        ys = np.linspace(-self.world_h / 2.0, self.world_h / 2.0, self.mask_resolution)
        X, Y = np.meshgrid(xs, ys)
        mask = np.ones_like(X, dtype=np.float32)
        for obs in self.world_obstacles:
            if obs.get("type") == "rectangle":
                inside = obstacle_contains_points(X, Y, obs)
                mask[inside] = 0.0
        return mask.astype(np.float32)

    def _build_obstacle_mask_inflated(self, resolution: int, world_w: float, world_h: float) -> np.ndarray:
        """Build obstacle mask on an inflated grid (for inflate-smooth-crop)."""
        xs = np.linspace(-world_w / 2.0, world_w / 2.0, resolution)
        ys = np.linspace(-world_h / 2.0, world_h / 2.0, resolution)
        X, Y = np.meshgrid(xs, ys)
        mask = np.ones_like(X, dtype=np.float32)
        for obs in self.world_obstacles:
            if obs.get("type") == "rectangle":
                inside = obstacle_contains_points(X, Y, obs)
                mask[inside] = 0.0
        return mask.astype(np.float32)

    def _compute_activations_grid(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Compute raw activations over a grid for all cells (no mask applied)."""
        H, W = X.shape
        C = self.total_grid_cells
        activ = np.zeros((H, W, C), dtype=np.float32)
        for i in range(H):
            x_row = torch.tensor(X[i, :], dtype=self.dtype, device=self.device)
            y_row = torch.tensor(Y[i, :], dtype=self.dtype, device=self.device)
            dx = x_row.unsqueeze(1) - self.x_trans_params.unsqueeze(0)
            dy = y_row.unsqueeze(1) - self.y_trans_params.unsqueeze(0)
            c, s = self.cos_theta.unsqueeze(0), self.sin_theta.unsqueeze(0)
            rx = c * dx + s * dy
            ry = -s * dx + c * dy
            freq = (2.0 * np.pi) / self.size_params.unsqueeze(0)
            z1 = torch.cos(freq * rx)
            z2 = torch.cos(freq * (rx / 2.0 + (np.sqrt(3.0) / 2.0) * ry))
            z3 = torch.cos(freq * (rx / 2.0 - (np.sqrt(3.0) / 2.0) * ry))
            z = (z1 + z2 + z3) / 3.0
            spread = self.spread_params.unsqueeze(0)
            a_row = torch.sign(z) * torch.pow(torch.abs(z), 1.0 / spread)
            activ[i, :, :] = a_row.detach().cpu().numpy()
        return activ

    def _build_advanced_mask(self) -> None:
        """
        Build per-cell masks using full pipeline: obstacle masking + wall splitting + smoothing.

        Steps:
        1. Build obstacle mask on inflated grid
        2. Compute raw activations on inflated grid
        3. For each cell: apply wall-based splitting (steps 2+3 from v12)
        4. Apply morphological smoothing to restore biological shapes
        5. Crop back to original world bounds
        """
        if not _HAS_SCIPY:
            print("[GridCellLayer v13] WARNING: SciPy not available; using simple obstacle-only mask.")
            self._build_simple_mask()
            return

        # Inflate-Smooth-Crop: Use larger grid to avoid edge artifacts during smoothing
        res = self.mask_resolution  # Original resolution (e.g., 128)
        buffer_meters = 1.0  # Add 0.5m buffer on each side
        inflated_w = self.world_w + buffer_meters
        inflated_h = self.world_h + buffer_meters

        # Calculate inflated resolution to maintain same pixels/meter
        pixels_per_meter = res / self.world_w
        inflated_res = int(np.round(inflated_w * pixels_per_meter))

        # Build inflated grid
        xs = np.linspace(-inflated_w / 2.0, inflated_w / 2.0, inflated_res)
        ys = np.linspace(-inflated_h / 2.0, inflated_h / 2.0, inflated_res)
        X, Y = np.meshgrid(xs, ys)

        # Step 1: Build obstacle mask on inflated grid
        obstacle_mask_inflated = self._build_obstacle_mask_inflated(inflated_res, inflated_w, inflated_h)
        free_mask_inflated = obstacle_mask_inflated > 0

        # Step 2: Compute raw activations on inflated grid
        activ = self._compute_activations_grid(X, Y)  # (inflated_res, inflated_res, C)

        # Step 3: Process each cell
        C = self.total_grid_cells
        final_masks_inflated = np.zeros((inflated_res, inflated_res, C), dtype=np.float32)

        struct = _ndimage.generate_binary_structure(2, 1)

        for c in range(C):
            cell_activ = activ[:, :, c]
            cell_mask = self._build_cell_mask_with_smoothing(
                cell_activ, free_mask_inflated, obstacle_mask_inflated, struct,
                inflated_res, inflated_w, inflated_h
            )
            final_masks_inflated[:, :, c] = cell_mask

        # Step 4: Crop inflated masks back to original world bounds
        crop_start = (inflated_res - res) // 2
        crop_end = crop_start + res
        final_masks = final_masks_inflated[crop_start:crop_end, crop_start:crop_end, :]

        # Store mask tensor
        self._mask = torch.tensor(final_masks, dtype=self.dtype, device=self.device)
        self._x_scale = (res - 1) / self.world_w
        self._y_scale = (res - 1) / self.world_h

        print(f"[GridCellLayer v13] Built {C} cell masks with wall splitting + edge smoothing (sigma={self.smooth_sigma}, res={res}).")

    def _build_cell_mask_with_smoothing(
        self,
        activations: np.ndarray,
        free_mask: np.ndarray,
        obstacle_mask: np.ndarray,
        struct: np.ndarray,
        inflated_res: int,
        inflated_w: float,
        inflated_h: float,
    ) -> np.ndarray:
        """
        Build mask for a single cell using steps 2+3 + smoothing.

        Pipeline:
        1. Threshold activations to get raw components
        2. Apply wall-based splitting for each component
        3. Keep largest fragment per component
        4. Apply morphological smoothing
        """
        res = inflated_res
        m = float(activations.max())
        if m <= 0.0:
            return np.zeros((res, res), dtype=np.float32)

        # Threshold to get components
        threshold_frac = 0.15
        raw_bin = activations > (threshold_frac * m)
        raw_label, raw_num = _ndimage.label(raw_bin, structure=struct)
        if raw_num == 0:
            return np.zeros((res, res), dtype=np.float32)

        new_mask = np.zeros((res, res), dtype=np.float32)

        def bounds_to_idx(xmin, xmax, ymin, ymax):
            xi0 = int(np.floor((xmin + inflated_w / 2.0) * (inflated_res - 1) / inflated_w))
            xi1 = int(np.ceil((xmax + inflated_w / 2.0) * (inflated_res - 1) / inflated_w))
            yi0 = int(np.floor((ymin + inflated_h / 2.0) * (inflated_res - 1) / inflated_h))
            yi1 = int(np.ceil((ymax + inflated_h / 2.0) * (inflated_res - 1) / inflated_h))
            return (max(0, min(inflated_res - 1, xi0)), max(0, min(inflated_res - 1, xi1)),
                    max(0, min(inflated_res - 1, yi0)), max(0, min(inflated_res - 1, yi1)))

        grid_xs = np.linspace(-inflated_w / 2.0, inflated_w / 2.0, inflated_res)
        grid_ys = np.linspace(-inflated_h / 2.0, inflated_h / 2.0, inflated_res)
        obstacle_slices = []
        for obs in self.world_obstacles:
            if obs.get("type") != "rectangle":
                continue
            corners = np.asarray(expanded_obstacle_corners(obs), dtype=np.float64)
            xmin, xmax = float(corners[:, 0].min()), float(corners[:, 0].max())
            ymin, ymax = float(corners[:, 1].min()), float(corners[:, 1].max())
            xi0, xi1, yi0, yi1 = bounds_to_idx(xmin, xmax, ymin, ymax)
            local_x, local_y = np.meshgrid(grid_xs[xi0 : xi1 + 1], grid_ys[yi0 : yi1 + 1])
            obs_slice = np.zeros((res, res), dtype=bool)
            obs_slice[yi0 : yi1 + 1, xi0 : xi1 + 1] = obstacle_contains_points(local_x, local_y, obs)
            obstacle_slices.append(obs_slice)

        # Process each raw component
        for rid in range(1, raw_num + 1):
            comp = (raw_label == rid)
            if not comp.any():
                continue

            cy, cx = np.nonzero(comp)
            comp_ymin, comp_ymax = cy.min(), cy.max()
            comp_xmin, comp_xmax = cx.min(), cx.max()

            # Wall-based splitting
            for obs_slice in obstacle_slices:
                inter_obs = comp & obs_slice
                if not inter_obs.any():
                    continue

                frac = inter_obs.sum() / comp.sum()
                if frac < self.wall_split_thresh:
                    continue

                cut = _ndimage.binary_dilation(obs_slice, structure=np.ones((3, 3), dtype=bool))
                cut[:comp_ymin, :] = False
                cut[comp_ymax + 1 :, :] = False
                cut[:, :comp_xmin] = False
                cut[:, comp_xmax + 1 :] = False
                comp = comp & (~cut)

            # Intersect with free space and keep largest fragment
            inter = comp & free_mask
            if not inter.any():
                continue
            inter_label, inter_num = _ndimage.label(inter, structure=struct)
            if inter_num == 0:
                continue

            best_region = None
            best_area = -1
            for cid in range(1, inter_num + 1):
                area = int((inter_label == cid).sum())
                if area > best_area:
                    best_area = area
                    best_region = (inter_label == cid)
            if best_region is not None:
                new_mask[best_region] = 1.0

        # Enforce obstacles zero
        new_mask *= free_mask.astype(np.float32)

        # Apply smoothing
        if self.smooth_sigma > 0:
            new_mask = self._smooth_activation_blobs(new_mask, free_mask, activations)

        return new_mask

    def _smooth_activation_blobs(
        self,
        mask: np.ndarray,
        free_mask: np.ndarray,
        activations: np.ndarray,
    ) -> np.ndarray:
        """
        Smooth activation blobs using aggressive morphological operations.
        Uses activation-based expansion + morphological closing/opening.
        """
        struct = _ndimage.generate_binary_structure(2, 1)
        labeled, num_blobs = _ndimage.label(mask > 1e-6, structure=struct)

        if num_blobs == 0:
            return mask

        smoothed_mask = np.zeros_like(mask, dtype=np.float32)
        sigma = self.smooth_sigma

        for blob_id in range(1, num_blobs + 1):
            blob = (labeled == blob_id)
            ys, xs = np.nonzero(blob)
            if len(ys) < 3:
                smoothed_mask[blob] = 1.0
                continue

            # Bounding box with padding
            pad = max(5, int(np.ceil(sigma * 5)))
            y0, y1 = max(0, ys.min() - pad), min(mask.shape[0], ys.max() + pad + 1)
            x0, x1 = max(0, xs.min() - pad), min(mask.shape[1], xs.max() + pad + 1)

            blob_local = blob[y0:y1, x0:x1].astype(np.float32)
            free_region = free_mask[y0:y1, x0:x1].astype(np.float32)
            activ_region = activations[y0:y1, x0:x1]

            # Normalize activations
            activ_max = activ_region.max()
            if activ_max > 0:
                activ_norm = activ_region / activ_max
            else:
                activ_norm = activ_region

            # Step 1: Activation-based expansion
            activation_threshold = max(0.03, 0.15 - sigma * 0.08)
            expanded_from_activations = (activ_norm > activation_threshold).astype(np.float32)
            expanded_from_activations *= free_region
            blob_expanded = np.maximum(blob_local, expanded_from_activations)

            # Keep only component connected to original blob
            blob_expanded_labeled, num_comp = _ndimage.label(blob_expanded, structure=struct)
            if num_comp > 1:
                center_y, center_x = int(np.mean(ys)) - y0, int(np.mean(xs)) - x0
                if (0 <= center_y < blob_expanded_labeled.shape[0] and
                    0 <= center_x < blob_expanded_labeled.shape[1]):
                    center_label = blob_expanded_labeled[center_y, center_x]
                    if center_label > 0:
                        blob_expanded = (blob_expanded_labeled == center_label).astype(np.float32)

            # Step 2: Aggressive morphological closing
            closing_radius = max(2, int(sigma * 2.5))
            struct_close = self._create_circular_struct(closing_radius)
            blob_smooth = _ndimage.binary_closing(blob_expanded, structure=struct_close).astype(np.float32)
            blob_smooth *= free_region

            # Step 3: Opening to reduce over-expansion
            opening_radius = max(1, int(sigma * 1.5))
            struct_open = self._create_circular_struct(opening_radius)
            blob_smooth = _ndimage.binary_opening(blob_smooth, structure=struct_open).astype(np.float32)
            blob_smooth *= free_region

            # Final connectivity check
            if blob_smooth.sum() > 0:
                blob_smooth_labeled, num_final = _ndimage.label(blob_smooth, structure=struct)
                if num_final > 1:
                    center_y, center_x = int(np.mean(ys)) - y0, int(np.mean(xs)) - x0
                    if (0 <= center_y < blob_smooth_labeled.shape[0] and
                        0 <= center_x < blob_smooth_labeled.shape[1]):
                        center_label = blob_smooth_labeled[center_y, center_x]
                        if center_label > 0:
                            blob_smooth = (blob_smooth_labeled == center_label).astype(np.float32)

            smoothed_mask[y0:y1, x0:x1] = np.maximum(smoothed_mask[y0:y1, x0:x1], blob_smooth)

        smoothed_mask *= free_mask.astype(np.float32)
        return smoothed_mask

    @staticmethod
    def _create_circular_struct(radius: int) -> np.ndarray:
        """Create a circular structuring element for morphological operations."""
        size = 2 * radius + 1
        y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
        circle = (x**2 + y**2 <= radius**2)
        return circle.astype(np.uint8)

    def _build_simple_mask(self) -> None:
        """Fallback: simple obstacle-only mask when SciPy not available."""
        res = self.mask_resolution
        xs = np.linspace(-self.world_w / 2.0, self.world_w / 2.0, res)
        ys = np.linspace(-self.world_h / 2.0, self.world_h / 2.0, res)
        X, Y = np.meshgrid(xs, ys)

        obstacle_mask = self._build_obstacle_mask()
        activ = self._compute_activations_grid(X, Y)

        # Apply obstacle mask to all cells
        final_masks = activ * obstacle_mask[:, :, None]

        self._mask = torch.tensor(final_masks, dtype=self.dtype, device=self.device)
        self._x_scale = (res - 1) / self.world_w
        self._y_scale = (res - 1) / self.world_h

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
        """
        Compute grid cell activations at a position with optional masking.

        Args:
            position: [x, y] coordinates
            threshold: Activation threshold (default: self.threshold)
            threshold_type: "soft" or "hard" (default: self.threshold_type)
            sparsity: Ignored (kept for API compatibility)
            normalization: "per-cell" or None (default: self.normalization)
            use_mask: Whether to apply spatial mask

        Returns:
            Tensor of shape (total_grid_cells,) with activations
        """
        thr = self.threshold if threshold is None else float(threshold)
        thr_type = self.threshold_type if threshold_type is None else threshold_type
        norm = self.normalization if normalization is None else normalization

        if not isinstance(position, torch.Tensor):
            position = torch.tensor(position, dtype=self.dtype)
        position = position.to(self.device)
        if position.dim() > 1:
            position = position.squeeze()
        x, y = position[0], position[1]

        # Compute activations
        dx = x - self.x_trans_params
        dy = y - self.y_trans_params
        rx = self.cos_theta * dx + self.sin_theta * dy
        ry = -self.sin_theta * dx + self.cos_theta * dy

        freq = (2.0 * np.pi) / self.size_params
        z1 = torch.cos(freq * rx)
        z2 = torch.cos(freq * (rx / 2.0 + (np.sqrt(3.0) / 2.0) * ry))
        z3 = torch.cos(freq * (rx / 2.0 - (np.sqrt(3.0) / 2.0) * ry))
        z = (z1 + z2 + z3) / 3.0

        spread = self.spread_params
        acts = torch.sign(z) * torch.pow(torch.abs(z), 1.0 / spread)

        # Normalization
        if norm == "per-cell":
            if not getattr(self, "min_max_updated", True):
                self.cell_min = torch.minimum(self.cell_min, acts)
                self.cell_max = torch.maximum(self.cell_max, acts)
            rng = self.cell_max - self.cell_min
            rng = torch.where(rng > 1e-8, rng, torch.ones_like(rng))
            acts = (acts - self.cell_min) / rng
            acts = torch.clamp(acts, 0.0, 1.0)

        # Threshold
        if thr_type == "soft":
            scale = 1.0 / (1.0 - thr + 1e-8)
            acts = torch.clamp((acts - thr) * scale, 0.0, 1.0)
        elif thr_type == "hard":
            acts = torch.where(acts >= thr, acts, torch.zeros_like(acts))

        # Apply mask
        if use_mask and hasattr(self, "_mask") and self._mask is not None:
            xi = int(round((float(x.item()) + self.world_w / 2.0) * self._x_scale))
            yi = int(round((float(y.item()) + self.world_h / 2.0) * self._y_scale))
            xi = max(0, min(self.mask_resolution - 1, xi))
            yi = max(0, min(self.mask_resolution - 1, yi))
            acts = acts * self._mask[yi, xi, :]

        return acts
