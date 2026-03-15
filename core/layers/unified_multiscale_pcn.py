"""
Unified Multi-Scale Place Cell Network with adaptive cross-scale inhibition.

This implementation combines all scales (small, medium, large) into a single unified
network with:
1. Gaussian boundary-based cross-scale inhibition
2. Unified cross-scale STDP creating single 1750×1750 adjacency matrix
3. Biologically plausible scale selection based on environmental context

Key parameters:
- Total cells: 1750 (1000 small + 500 medium + 250 large)
- Optimal distances: d_opt^0 = 0.7m, d_opt^1 = 2.5m, d_opt^2 = 5.0m
- Tuning width: σ_tune = 1.5m
- Cross-scale inhibition strength: Γ^cross = 0.35
"""

import numpy as np
import torch
from numpy.random import default_rng
from typing import Optional, List, Dict, Union

# Set fixed seed for reproducibility
torch.manual_seed(5)


class UnifiedMultiScalePCN:
    """
    Unified place cell network combining multiple scales with adaptive inhibition.

    Implements Gaussian boundary-based cross-scale inhibition where each scale
    has an optimal distance from boundaries. Near boundaries, small scales dominate.
    In open areas, large scales dominate.

    Uses unified cross-scale STDP to create a single adjacency matrix connecting
    all place cells across scales.
    """

    def __init__(
        self,
        scale_configs: List[Dict],  # [{"num_pc": 2000, "sigma_r": 0.5}, ...]
        bvc_layer=None,
        bvc_layers: Optional[List] = None,
        timestep: int = 32 * 3,
        n_hd: int = 8,
        enable_ojas: bool = False,
        enable_stdp: bool = False,
        w_in_init_ratio: float = 0.25,
        grid_influence: float = 0.3,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
        gamma_pp: float = 0.5,
        gamma_pb: float = 0.3,
        gamma_pg: float = 0.3,
        gamma_cross: Union[float, List[float]] = 0.35,  # Cross-scale inhibition strength
        sigma_tune: float = 1.5,    # Backward-compatible fallback
        gate_mode: str = "normal",  # "normal" | "no_gate_no_inhibition" | "no_gate_with_inhibition"
    ):
        """
        Initialize unified multi-scale place cell network.

        Args:
            bvc_layer: Boundary Vector Cell layer providing input (fallback shared layer)
            bvc_layers: Optional per-scale list of BVC layers (preferred)
            scale_configs: List of scale configurations, e.g.:
                [{"num_pc": 2000, "sigma_r": 0.5, "name": "small"}, ...]
            timestep: Simulation timestep in milliseconds
            n_hd: Number of head direction cells
            enable_ojas: Enable Oja's learning rule
            enable_stdp: Enable cross-scale STDP
            w_in_init_ratio: Initial weight connection probability
            device: Computation device
            dtype: Data type
            gamma_pp: Within-scale recurrent inhibition
            gamma_pb: Afferent inhibition from BVCs
            gamma_cross: Cross-scale inhibition strength
            sigma_tune: Backward-compatible fallback if per-scale tuning is absent.
        """
        self.device = device
        self.dtype = dtype
        # Support per-scale BVC layers (preferred), with backward-compatible single-layer fallback.
        if bvc_layers is not None:
            self.bvc_layers = bvc_layers
        elif bvc_layer is not None:
            self.bvc_layers = [bvc_layer for _ in scale_configs]
        else:
            raise ValueError("UnifiedMultiScalePCN requires either bvc_layer or bvc_layers.")

        # Store scale configurations
        self.scale_configs = scale_configs
        self.num_scales = len(scale_configs)

        # Calculate total number of place cells across all scales
        self.num_pc_per_scale = [cfg["num_pc"] for cfg in scale_configs]
        self.num_pc_total = sum(self.num_pc_per_scale)
        self.num_grid_per_scale = [cfg.get("num_grid_cells", 0) for cfg in scale_configs]
        self.num_grid_total = sum(self.num_grid_per_scale)
        self.num_bvc_per_scale = [layer.num_bvc for layer in self.bvc_layers]
        self.num_bvc_total = sum(self.num_bvc_per_scale)

        # Define optimal distances for each scale directly from scale configs.
        self.d_opt = torch.tensor(
            [cfg["d_opt"] for cfg in scale_configs],
            dtype=dtype,
            device=device
        )
        self.d_opt_per_pc = None
        self._build_d_opt_per_pc()

        # Store scale parameters
        self.sigma_r_per_scale = torch.tensor(
            [cfg["sigma_r"] for cfg in scale_configs],
            dtype=dtype,
            device=device
        )

        # Inhibition and GC-mix parameters are scale-specific (expanded per-PC).
        self.gamma_pp_per_pc = torch.cat([
            torch.full((cfg["num_pc"],), float(cfg.get("gamma_pp", gamma_pp)), dtype=dtype, device=device)
            for cfg in scale_configs
        ])
        self.gamma_pb_per_pc = torch.cat([
            torch.full((cfg["num_pc"],), float(cfg.get("gamma_pb", gamma_pb)), dtype=dtype, device=device)
            for cfg in scale_configs
        ])
        self.gamma_pg_per_pc = torch.cat([
            torch.full((cfg["num_pc"],), float(cfg.get("gamma_pg", gamma_pg)), dtype=dtype, device=device)
            for cfg in scale_configs
        ])
        self.grid_influence_per_pc = torch.cat([
            torch.full((cfg["num_pc"],), float(cfg.get("grid_influence", grid_influence)), dtype=dtype, device=device)
            for cfg in scale_configs
        ])
        # Keep scalar attributes for backward compatibility/debug prints.
        self.gamma_pp = float(torch.mean(self.gamma_pp_per_pc).item())
        self.gamma_pb = float(torch.mean(self.gamma_pb_per_pc).item())
        self.gamma_pg = float(torch.mean(self.gamma_pg_per_pc).item())
        if isinstance(gamma_cross, (list, tuple, np.ndarray, torch.Tensor)):
            gamma_cross_values = [float(v) for v in gamma_cross]
            if len(gamma_cross_values) != self.num_scales:
                raise ValueError(
                    f"gamma_cross length ({len(gamma_cross_values)}) must match num_scales ({self.num_scales})."
                )
        else:
            gamma_cross_values = [
                float(cfg.get("gamma_cross", gamma_cross)) for cfg in scale_configs
            ]
        self.gamma_cross_per_scale = torch.tensor(
            gamma_cross_values, dtype=dtype, device=device
        )
        # Keep scalar mean for backward compatibility with existing code paths.
        self.gamma_cross = float(torch.mean(self.gamma_cross_per_scale).item())
        # Per-scale Gaussian tuning width tied to sensory width:
        # sigma_tune_s = sigma_tune_k_s * sigma_r_s
        sigma_tune_k_values = [
            float(cfg.get("sigma_tune_k", 1.0)) for cfg in scale_configs
        ]
        self.sigma_tune_k_per_scale = torch.tensor(
            sigma_tune_k_values, dtype=dtype, device=device
        )
        self.sigma_tune_per_scale = torch.clamp(
            self.sigma_tune_k_per_scale * self.sigma_r_per_scale, min=1e-3
        )
        # Keep scalar mean for backward compatibility/debugging.
        self.sigma_tune = float(torch.mean(self.sigma_tune_per_scale).item())
        self.tau_p = 0.5
        self.tau = timestep / 1000.0
        # Optional one-sided preference for the largest scale:
        # preserve Gaussian approach to d_opt, but avoid decay beyond d_opt.
        largest_cfg = scale_configs[-1] if len(scale_configs) > 0 else {}
        self.large_scale_one_sided = bool(largest_cfg.get("large_scale_one_sided", False))
        self.large_scale_plateau = float(largest_cfg.get("large_scale_plateau", 1.0))
        self.large_scale_plateau = min(1.0, max(0.0, self.large_scale_plateau))

        # Learning parameters
        self.enable_ojas = enable_ojas
        self.enable_stdp = enable_stdp
        assert gate_mode in ("normal", "no_gate_no_inhibition", "no_gate_with_inhibition"), \
            f"Unknown gate_mode '{gate_mode}'"
        self.gate_mode = gate_mode
        self.alpha_pb = np.sqrt(0.5)
        self.grid_influence = float(torch.mean(self.grid_influence_per_pc).item())

        # Per-scale grid/BVC mixing and learning parameters
        self.alpha_pb_per_pc = torch.cat([
            torch.full((cfg["num_pc"],), float(cfg.get("alpha_pb", np.sqrt(0.5))), dtype=dtype, device=device)
            for cfg in scale_configs
        ])
        self.alpha_pg_per_pc = torch.cat([
            torch.full((cfg["num_pc"],), float(cfg.get("alpha_pg", np.sqrt(0.5))), dtype=dtype, device=device)
            for cfg in scale_configs
        ])
        # Initialize unified weight matrices
        rng = default_rng()

        # Scale-local BVC->PC input weights: (num_pc_total, num_bvc_total)
        # Each scale's place cells connect only to that same scale's BVC population.
        self.w_in = torch.zeros(
            (self.num_pc_total, self.num_bvc_total), dtype=dtype, device=device
        )

        # Input weights from Grid Cells: (num_pc_total, num_grid_total)
        if self.num_grid_total > 0:
            self.grid_boundaries = [0]
            gc_cumsum = 0
            for n_gc in self.num_grid_per_scale:
                gc_cumsum += n_gc
                self.grid_boundaries.append(gc_cumsum)

            self.w_grid = torch.zeros(
                (self.num_pc_total, self.num_grid_total),
                dtype=dtype,
                device=device
            )

            # Initialize block-wise using each scale's configured ratio.
            for scale_idx, cfg in enumerate(scale_configs):
                pc_start = sum(self.num_pc_per_scale[:scale_idx])
                pc_end = pc_start + cfg["num_pc"]
                gc_start = self.grid_boundaries[scale_idx]
                gc_end = self.grid_boundaries[scale_idx + 1]

                if gc_end <= gc_start:
                    continue

                ratio = float(cfg.get("w_grid_init_ratio", 0.25))
                w_grid_block = rng.binomial(
                    n=1, p=ratio, size=(cfg["num_pc"], gc_end - gc_start)
                )
                self.w_grid[pc_start:pc_end, gc_start:gc_end] = torch.tensor(
                    w_grid_block, dtype=dtype, device=device
                )

            # Block-diagonal mask: only scale-local connections are legal.
            # Used to zero out any off-diagonal growth from Oja updates.
            self.w_grid_block_mask = torch.zeros_like(self.w_grid, dtype=torch.bool)
            for scale_idx in range(self.num_scales):
                pc_s = sum(self.num_pc_per_scale[:scale_idx])
                pc_e = pc_s + self.num_pc_per_scale[scale_idx]
                gc_s = self.grid_boundaries[scale_idx]
                gc_e = self.grid_boundaries[scale_idx + 1]
                self.w_grid_block_mask[pc_s:pc_e, gc_s:gc_e] = True
        else:
            self.grid_boundaries = [0]
            self.w_grid = None
            self.w_grid_block_mask = None

        # Unified recurrent weight matrix: (n_hd, num_pc_total, num_pc_total)
        # This single matrix connects ALL place cells across scales
        self.w_rec_unified = torch.zeros(
            (n_hd, self.num_pc_total, self.num_pc_total),
            dtype=dtype,
            device=device
        )

        # Unified activation vector for all place cells
        self.place_cell_activations = torch.zeros(
            self.num_pc_total,
            dtype=dtype,
            device=device
        )

        # Activation update variable
        self.activation_update = torch.zeros(
            self.num_pc_total,
            dtype=dtype,
            device=device
        )

        # BVC activations (concatenated over scales)
        self.bvc_activations = torch.zeros(
            self.num_bvc_total,
            dtype=dtype,
            device=device
        )

        # Grid activations across all scales (concatenated)
        self.grid_cell_activations = None
        if self.num_grid_total > 0:
            self.grid_cell_activations = torch.zeros(
                self.num_grid_total,
                dtype=dtype,
                device=device
            )
        self.last_grid_diagnostics = None

        # Pre-allocated intermediate buffers for hot-path (avoid per-step GPU allocation)
        self._buf_bvc_inh   = torch.zeros(self.num_pc_total, dtype=dtype, device=device)
        self._buf_grid_inh  = torch.zeros(self.num_pc_total, dtype=dtype, device=device)
        self._buf_rec_inh   = torch.zeros(self.num_pc_total, dtype=dtype, device=device)
        self._buf_cross_inh = torch.zeros(self.num_pc_total, dtype=dtype, device=device)
        self._buf_ones      = torch.ones(self.num_pc_total,  dtype=dtype, device=device)

        # Grid diagnostics gating: run every _diag_interval steps to avoid per-step GPU sync
        self._diag_interval  = 10
        self._diag_step_count = 0

        # Eligibility traces for STDP
        self.place_cell_trace = None
        self.hd_cell_trace = None
        if enable_stdp:
            self.place_cell_trace = torch.zeros(
                self.num_pc_total,
                dtype=dtype,
                device=device
            )
            self.hd_cell_trace = torch.zeros(
                (n_hd, 1, 1),
                dtype=dtype,
                device=device
            )

        # Store scale boundaries for indexing
        self.scale_boundaries = [0]
        cumsum = 0
        for num_pc in self.num_pc_per_scale:
            cumsum += num_pc
            self.scale_boundaries.append(cumsum)
        self.bvc_boundaries = [0]
        bvc_cumsum = 0
        for num_bvc in self.num_bvc_per_scale:
            bvc_cumsum += num_bvc
            self.bvc_boundaries.append(bvc_cumsum)

        # Initialize scale-local BVC blocks in w_in using each scale's configured ratio.
        for scale_idx, cfg in enumerate(scale_configs):
            pc_start = self.scale_boundaries[scale_idx]
            pc_end = self.scale_boundaries[scale_idx + 1]
            bvc_start = self.bvc_boundaries[scale_idx]
            bvc_end = self.bvc_boundaries[scale_idx + 1]
            ratio = float(cfg.get("w_in_init_ratio", w_in_init_ratio))
            w_in_block = rng.binomial(
                n=1, p=ratio, size=(pc_end - pc_start, bvc_end - bvc_start)
            )
            self.w_in[pc_start:pc_end, bvc_start:bvc_end] = torch.tensor(
                w_in_block, dtype=dtype, device=device
            )
        self.w_in = torch.nn.Parameter(self.w_in, requires_grad=False)

        print(f"[UnifiedMultiScalePCN] Initialized with {self.num_pc_total} total cells")
        print(f"  Grid cells (total): {self.num_grid_total}")
        print(f"  BVC cells (total): {self.num_bvc_total}")
        print(f"  Scale boundaries: {self.scale_boundaries}")
        print(f"  Optimal distances: {self.d_opt.cpu().numpy()}")
        print(
            f"  Cross-scale inhibition per scale: {self.gamma_cross_per_scale.detach().cpu().numpy()}, "
            f"mean={self.gamma_cross:.4f}, sigma_tune_per_scale={self.sigma_tune_per_scale.detach().cpu().numpy()}"
        )

    def get_scale_activations(self, scale_idx: int) -> torch.Tensor:
        """Get activations for a specific scale."""
        start = self.scale_boundaries[scale_idx]
        end = self.scale_boundaries[scale_idx + 1]
        return self.place_cell_activations[start:end]

    def compute_scale_preference(self, proximity: float) -> torch.Tensor:
        """
        Compute Gaussian scale preference g_s(d) for each scale.

        g_s(d) = exp(-(d - d_opt^s)^2 / (2 * sigma_tune^2))
        where d is boundary proximity.
        """
        proximity_t = torch.as_tensor(proximity, dtype=self.dtype, device=self.device)
        sigma2 = torch.clamp(self.sigma_tune_per_scale ** 2, min=1e-6)
        pref = torch.exp(-((proximity_t - self.d_opt) ** 2) / (2.0 * sigma2))
        if self.large_scale_one_sided and self.num_scales > 0:
            last_idx = self.num_scales - 1
            plateau = torch.as_tensor(self.large_scale_plateau, dtype=self.dtype, device=self.device)
            pref[last_idx] = torch.where(
                proximity_t >= self.d_opt[last_idx],
                plateau,
                pref[last_idx],
            )
        return torch.clamp(pref, min=0.0, max=1.0)

    def _build_d_opt_per_pc(self) -> None:
        """
        Build per-place-cell optimal distance values with optional within-scale jitter.

        Each scale can specify:
        - d_opt_jitter_std: Gaussian std around scale d_opt
        - d_opt_jitter_range: Clamp range (+/- around scale d_opt)
        - d_opt_jitter_seed: Optional deterministic seed per scale
        """
        d_opt_blocks = []
        for scale_idx, cfg in enumerate(self.scale_configs):
            n_pc = int(cfg["num_pc"])
            base = float(cfg.get("d_opt", self.d_opt[scale_idx].item()))
            jitter_std = float(cfg.get("d_opt_jitter_std", 0.0))
            jitter_range = float(cfg.get("d_opt_jitter_range", 0.0))
            jitter_seed = int(cfg.get("d_opt_jitter_seed", 1000 + scale_idx))

            if jitter_std > 0.0:
                gen = torch.Generator(device="cpu")
                gen.manual_seed(jitter_seed)
                samples = torch.normal(
                    mean=base,
                    std=jitter_std,
                    size=(n_pc,),
                    generator=gen,
                )
            else:
                samples = torch.full((n_pc,), base, dtype=torch.float32)

            if jitter_range > 0.0:
                low = base - jitter_range
                high = base + jitter_range
                samples = torch.clamp(samples, min=low, max=high)

            d_opt_blocks.append(samples.to(dtype=self.dtype, device=self.device))

        self.d_opt_per_pc = torch.cat(d_opt_blocks, dim=0)

    def compute_scale_preference_per_pc(self, proximity: float) -> torch.Tensor:
        """
        Compute per-place-cell Gaussian preference using d_opt_per_pc.
        """
        proximity_t = torch.as_tensor(proximity, dtype=self.dtype, device=self.device)
        sigma_tune_per_pc = self.expand_scale_values_to_pc(self.sigma_tune_per_scale)
        sigma2 = torch.clamp(sigma_tune_per_pc ** 2, min=1e-6)
        pref_pc = torch.exp(-((proximity_t - self.d_opt_per_pc) ** 2) / (2.0 * sigma2))
        if self.large_scale_one_sided and self.num_scales > 0:
            last_start = self.scale_boundaries[-2]
            last_end = self.scale_boundaries[-1]
            plateau = torch.as_tensor(self.large_scale_plateau, dtype=self.dtype, device=self.device)
            last_pref = pref_pc[last_start:last_end]
            last_dopt = self.d_opt_per_pc[last_start:last_end]
            pref_pc[last_start:last_end] = torch.where(
                proximity_t >= last_dopt,
                plateau,
                last_pref,
            )
        return torch.clamp(pref_pc, min=0.0, max=1.0)

    def expand_scale_values_to_pc(self, values_per_scale: torch.Tensor) -> torch.Tensor:
        """Broadcast per-scale values to per-PC vector using scale boundaries."""
        expanded = torch.zeros_like(self.place_cell_activations)
        for scale_idx in range(self.num_scales):
            start = self.scale_boundaries[scale_idx]
            end = self.scale_boundaries[scale_idx + 1]
            expanded[start:end] = values_per_scale[scale_idx]
        return expanded

    def get_last_scale_preference_per_pc(self) -> Optional[torch.Tensor]:
        """
        Return the most recently computed scale preference expanded to per-PC values.
        """
        if hasattr(self, "last_scale_preference_per_pc") and self.last_scale_preference_per_pc is not None:
            return self.last_scale_preference_per_pc
        if not hasattr(self, "last_scale_preference") or self.last_scale_preference is None:
            return None
        return self.expand_scale_values_to_pc(self.last_scale_preference)

    def compute_cross_scale_inhibition(self, proximity: float) -> torch.Tensor:
        """
        Compute Gaussian cross-scale inhibition based on boundary proximity.

        Each scale has an optimal distance from boundaries:
        - Small scale (0): optimal at d_opt^0 = 0.7m (near boundaries)
        - Medium scale (1): optimal at d_opt^1 = 2.5m (intermediate)
        - Large scale (2): optimal at d_opt^2 = 5.0m (open areas)

        Inhibition formula:
        I_cross^s(d) = Γ^cross × [1 - exp(-(d - d_opt^s)² / (2σ_tune²))] × Σ_(k≠s) Σ_j v_j^(p,k)

        Args:
            proximity: Boundary proximity in meters (minimum distance to nearest boundary)

        Returns:
            inhibition: Tensor of shape (num_pc_total,) with inhibition for each cell
        """
        # Initialize inhibition tensor
        inhibition = torch.zeros_like(self.place_cell_activations)
        scale_preference = self.compute_scale_preference(proximity)

        # For each scale, compute how inappropriate the current distance is
        for scale_idx in range(self.num_scales):
            start = self.scale_boundaries[scale_idx]
            end = self.scale_boundaries[scale_idx + 1]
            # Mismatch derived from scale preference (high far from preferred d_opt).
            mismatch = 1.0 - scale_preference[scale_idx]

            # Sum activations from OTHER scales
            other_scales_activation = 0.0
            for other_idx in range(self.num_scales):
                if other_idx != scale_idx:
                    other_start = self.scale_boundaries[other_idx]
                    other_end = self.scale_boundaries[other_idx + 1]
                    other_scales_activation += torch.sum(
                        self.place_cell_activations[other_start:other_end]
                    )

            # Apply inhibition to this scale
            gamma_cross_scale = self.gamma_cross_per_scale[scale_idx]
            inhibition[start:end] = (
                gamma_cross_scale
                * mismatch
                * other_scales_activation
            )

        return inhibition

    def get_place_cell_activations(
        self,
        distances,
        grid_activations: Optional[torch.Tensor] = None,
        hd_activations: Optional[torch.Tensor] = None,
        collided: bool = False,
        proximity: float = 5.0,  # Boundary proximity in meters
    ) -> torch.Tensor:
        """
        Compute unified place cell activations with cross-scale inhibition.

        Args:
            distances: BVC distances
            hd_activations: Head direction activations
            collided: Whether robot collided
            proximity: Minimum distance to nearest boundary (meters)

        Returns:
            place_cell_activations: Unified activation vector (num_pc_total,)
        """
        # Convert distances to torch tensor if needed
        if isinstance(distances, torch.Tensor):
            distances_torch = distances.clone().detach().to(dtype=self.dtype, device=self.device)
        else:
            distances_torch = torch.tensor(distances, dtype=self.dtype, device=self.device)

        if hd_activations is not None:
            if isinstance(hd_activations, torch.Tensor):
                hd_activations_torch = hd_activations.clone().detach().to(dtype=self.dtype, device=self.device)
            else:
                hd_activations_torch = torch.as_tensor(hd_activations, dtype=self.dtype, device=self.device)

        # Compute per-scale BVC activations and concatenate.
        bvc_blocks = []
        for bvc_layer_s in self.bvc_layers:
            bvc_blocks.append(bvc_layer_s.get_bvc_activation(distances=distances_torch))
        self.bvc_activations = torch.cat(bvc_blocks, dim=0)

        # Optional grid activations (concatenated across scales)
        if (
            grid_activations is not None
            and self.num_grid_total > 0
            and self.w_grid is not None
        ):
            if isinstance(grid_activations, torch.Tensor):
                self.grid_cell_activations = grid_activations.clone().detach().to(dtype=self.dtype, device=self.device)
            else:
                self.grid_cell_activations = torch.tensor(grid_activations, dtype=self.dtype, device=self.device)
            if int(self.grid_cell_activations.numel()) != int(self.num_grid_total):
                raise ValueError(
                    f"Unified grid activation length mismatch: "
                    f"got={int(self.grid_cell_activations.numel())}, "
                    f"expected={int(self.num_grid_total)}"
                )

        # Afferent excitation terms
        bvc_afferent_excitation = torch.matmul(self.w_in, self.bvc_activations)
        grid_afferent_excitation = torch.zeros_like(bvc_afferent_excitation)
        if self.grid_cell_activations is not None and self.w_grid is not None:
            grid_afferent_excitation = torch.matmul(self.w_grid, self.grid_cell_activations)

        # Mix BVC and Grid inputs per place cell according to scale-config grid influence
        afferent_excitation = (
            (1.0 - self.grid_influence_per_pc) * bvc_afferent_excitation
            + self.grid_influence_per_pc * grid_afferent_excitation
        )
        self._update_grid_diagnostics(
            bvc_afferent_excitation=bvc_afferent_excitation,
            grid_afferent_excitation=grid_afferent_excitation,
        )
        # Proximity-based scale gate (always computed; used differently per gate_mode).
        scale_preference = self.compute_scale_preference(proximity)
        scale_preference_per_pc = self.compute_scale_preference_per_pc(proximity)
        self.last_scale_preference = scale_preference.detach()
        self.last_scale_preference_per_pc = scale_preference_per_pc.detach()
        self.last_proximity = float(proximity)

        if self.gate_mode == "normal":
            # Standard behaviour: gate multiplies excitation and suppresses IIR bleed.
            g_excitation = scale_preference_per_pc
            g_iir        = scale_preference_per_pc
        else:
            # Both no-gate modes: remove the multiplicative gate on excitation/IIR.
            g_excitation = self._buf_ones
            g_iir        = self._buf_ones

        afferent_excitation = g_excitation * afferent_excitation

        # Scale-local BVC afferent inhibition per PC block.
        bvc_afferent_inhibition = self._buf_bvc_inh.zero_()
        for scale_idx in range(self.num_scales):
            pc_start = self.scale_boundaries[scale_idx]
            pc_end = self.scale_boundaries[scale_idx + 1]
            bvc_start = self.bvc_boundaries[scale_idx]
            bvc_end = self.bvc_boundaries[scale_idx + 1]
            bvc_sum_scale = torch.sum(self.bvc_activations[bvc_start:bvc_end])
            gamma_pb_scale = self.gamma_pb_per_pc[pc_start]
            bvc_afferent_inhibition[pc_start:pc_end] = g_excitation[pc_start:pc_end] * gamma_pb_scale * bvc_sum_scale
        grid_afferent_inhibition = self._buf_grid_inh.zero_()
        if self.grid_cell_activations is not None:
            for scale_idx in range(self.num_scales):
                pc_start = self.scale_boundaries[scale_idx]
                pc_end = self.scale_boundaries[scale_idx + 1]
                gc_start = self.grid_boundaries[scale_idx]
                gc_end = self.grid_boundaries[scale_idx + 1]
                gc_sum_scale = torch.sum(self.grid_cell_activations[gc_start:gc_end])
                gamma_pg_scale = self.gamma_pg_per_pc[pc_start]
                grid_afferent_inhibition[pc_start:pc_end] = g_excitation[pc_start:pc_end] * gamma_pg_scale * gc_sum_scale

        afferent_inhibition = (
            (1.0 - self.grid_influence_per_pc) * bvc_afferent_inhibition
            + self.grid_influence_per_pc * grid_afferent_inhibition
        )

        # Scale-local recurrent inhibition (within-scale only).
        recurrent_inhibition = self._buf_rec_inh.zero_()
        for scale_idx in range(self.num_scales):
            pc_start = self.scale_boundaries[scale_idx]
            pc_end = self.scale_boundaries[scale_idx + 1]
            scale_sum = torch.sum(self.place_cell_activations[pc_start:pc_end])
            gamma_pp_scale = self.gamma_pp_per_pc[pc_start]
            recurrent_inhibition[pc_start:pc_end] = gamma_pp_scale * scale_sum

        # Cross-scale inhibition: Gaussian boundary-based.
        # "no_gate_no_inhibition"   → zeroed out entirely.
        # "no_gate_with_inhibition" → computed normally using real proximity gate.
        # "normal"                  → computed normally.
        if self.gate_mode == "no_gate_no_inhibition":
            cross_scale_inhibition = self._buf_cross_inh.zero_()
        else:
            cross_scale_inhibition = self.compute_cross_scale_inhibition(proximity)

        # Update activation equation.
        self.activation_update += self.tau_p * (
            -self.activation_update
            + afferent_excitation
            - afferent_inhibition
            - recurrent_inhibition
            - cross_scale_inhibition
        )

        # Suppress residual IIR bleed (g_iir = ones when gate disabled).
        self.activation_update = self.activation_update * g_iir

        # Apply ReLU and tanh
        self.place_cell_activations = torch.tanh(torch.relu(self.activation_update))

        # STDP updates (unified across all scales)
        if (
            self.enable_stdp
            and torch.any(self.place_cell_activations != 0)
            and not collided
        ):
            # No gate applied to STDP traces when gate is disabled.
            gated_pc_activations = self.place_cell_activations * g_excitation
            # Update eligibility trace for place cells
            self.place_cell_trace += (self.tau / 3) * (
                gated_pc_activations - self.place_cell_trace
            )

            # Update eligibility trace for head direction cells
            hd_activations_no_nan = torch.nan_to_num(hd_activations_torch)
            hd_activations_no_nan = hd_activations_no_nan.unsqueeze(1).unsqueeze(2)
            self.hd_cell_trace += (self.tau / 3) * (
                hd_activations_no_nan - self.hd_cell_trace
            )

            # Update unified recurrent weights (cross-scale STDP)
            hd_contrib = torch.nan_to_num(hd_activations_torch).unsqueeze(-1).unsqueeze(-1)

            # Outer products: (num_pc_total x num_pc_total)
            pc_act_mat = torch.ger(gated_pc_activations, self.place_cell_trace)
            pc_trace_mat = torch.ger(self.place_cell_trace, gated_pc_activations)

            # STDP update for unified matrix
            update_rec = hd_contrib * (pc_act_mat - pc_trace_mat)

            self.w_rec_unified += update_rec.type(self.dtype)

        # Oja's rule for input weights
        if self.enable_ojas and torch.any(self.place_cell_activations != 0):
            pc_activations_col = (self.place_cell_activations * g_excitation).unsqueeze(1)
            # BVC->PC Oja update (scale-local blocks only).
            alpha_pb_col = self.alpha_pb_per_pc.unsqueeze(1)
            weight_update_bvc = torch.zeros_like(self.w_in)
            for scale_idx in range(self.num_scales):
                pc_start = self.scale_boundaries[scale_idx]
                pc_end = self.scale_boundaries[scale_idx + 1]
                bvc_start = self.bvc_boundaries[scale_idx]
                bvc_end = self.bvc_boundaries[scale_idx + 1]

                pc_col = pc_activations_col[pc_start:pc_end]
                bvc_row = self.bvc_activations[bvc_start:bvc_end].unsqueeze(0)
                w_block = self.w_in[pc_start:pc_end, bvc_start:bvc_end]
                alpha_block = alpha_pb_col[pc_start:pc_end]
                grid_mix_bvc_block = (1.0 - self.grid_influence_per_pc[pc_start:pc_end]).unsqueeze(1)
                hebbian_bvc = torch.matmul(pc_col, bvc_row)
                decay_bvc = (1.0 / torch.clamp(alpha_block, min=1e-6)) * (pc_col**2) * w_block
                update_block = self.tau * (hebbian_bvc - decay_bvc) * grid_mix_bvc_block
                weight_update_bvc[pc_start:pc_end, bvc_start:bvc_end] = update_block

            self.w_in.data += weight_update_bvc
            self.w_in.data = torch.clamp(self.w_in.data, min=0.0)

            # Grid->PC Oja update (scale-local blocks only — mirrors BVC update above).
            # Full outer-product updates would grow off-diagonal entries (no Hebbian decay
            # there since w=0), coupling each scale's PCs to the wrong GCN frequency.
            if self.grid_cell_activations is not None and self.w_grid is not None:
                alpha_pg_col = self.alpha_pg_per_pc.unsqueeze(1)
                weight_update_gc = torch.zeros_like(self.w_grid)
                for scale_idx in range(self.num_scales):
                    pc_start = self.scale_boundaries[scale_idx]
                    pc_end   = self.scale_boundaries[scale_idx + 1]
                    gc_start = self.grid_boundaries[scale_idx]
                    gc_end   = self.grid_boundaries[scale_idx + 1]
                    if gc_end <= gc_start:
                        continue
                    pc_col       = pc_activations_col[pc_start:pc_end]
                    gc_row       = self.grid_cell_activations[gc_start:gc_end].unsqueeze(0)
                    w_block      = self.w_grid[pc_start:pc_end, gc_start:gc_end]
                    alpha_block  = alpha_pg_col[pc_start:pc_end]
                    grid_mix_col = self.grid_influence_per_pc[pc_start:pc_end].unsqueeze(1)
                    hebbian_gc   = torch.matmul(pc_col, gc_row)
                    decay_gc     = (1.0 / torch.clamp(alpha_block, min=1e-6)) * (pc_col**2) * w_block
                    weight_update_gc[pc_start:pc_end, gc_start:gc_end] = \
                        self.tau * (hebbian_gc - decay_gc) * grid_mix_col
                self.w_grid.data += weight_update_gc
                self.w_grid.data *= self.w_grid_block_mask.float()  # belt-and-suspenders
                self.w_grid.data  = torch.clamp(self.w_grid.data, min=0.0)

        return self.place_cell_activations

    def _update_grid_diagnostics(
        self,
        bvc_afferent_excitation: torch.Tensor,
        grid_afferent_excitation: torch.Tensor,
    ) -> None:
        """Store lightweight diagnostics to verify GC contribution in unified mode."""
        self._diag_step_count += 1
        if self._diag_step_count % self._diag_interval != 0:
            return  # Skip to avoid per-step GPU→CPU sync

        bvc_abs_mean = float(torch.mean(torch.abs(bvc_afferent_excitation)).item())
        grid_abs_mean = float(torch.mean(torch.abs(grid_afferent_excitation)).item())
        denom = bvc_abs_mean + grid_abs_mean + 1e-12
        grid_share = float(grid_abs_mean / denom)

        if self.grid_cell_activations is not None and self.grid_cell_activations.numel() > 0:
            gc_active_frac = float((self.grid_cell_activations > 1e-6).float().mean().item())
            gc_mean = float(torch.mean(self.grid_cell_activations).item())
            gc_max = float(torch.max(self.grid_cell_activations).item())
        else:
            gc_active_frac = 0.0
            gc_mean = 0.0
            gc_max = 0.0

        per_scale = []
        for scale_idx in range(self.num_scales):
            pc_start = self.scale_boundaries[scale_idx]
            pc_end = self.scale_boundaries[scale_idx + 1]
            bvc_block = bvc_afferent_excitation[pc_start:pc_end]
            grid_block = grid_afferent_excitation[pc_start:pc_end]
            bvc_block_abs = float(torch.mean(torch.abs(bvc_block)).item())
            grid_block_abs = float(torch.mean(torch.abs(grid_block)).item())
            block_denom = bvc_block_abs + grid_block_abs + 1e-12
            per_scale.append({
                "scale_idx": int(scale_idx),
                "bvc_abs_mean": bvc_block_abs,
                "grid_abs_mean": grid_block_abs,
                "grid_share": float(grid_block_abs / block_denom),
            })

        self.last_grid_diagnostics = {
            "bvc_abs_mean": bvc_abs_mean,
            "grid_abs_mean": grid_abs_mean,
            "grid_share": grid_share,
            "gc_active_frac": gc_active_frac,
            "gc_mean": gc_mean,
            "gc_max": gc_max,
            "per_scale": per_scale,
        }

    def get_activations_per_scale(self) -> List[torch.Tensor]:
        """
        Split unified activations into per-scale lists for compatibility.

        Returns:
            List of activation tensors, one per scale
        """
        activations_per_scale = []
        for scale_idx in range(self.num_scales):
            start = self.scale_boundaries[scale_idx]
            end = self.scale_boundaries[scale_idx + 1]
            activations_per_scale.append(self.place_cell_activations[start:end])
        return activations_per_scale

    def preplay(self, direction: int, num_steps: int = 1) -> torch.Tensor:
        """
        Simple unified preplay using cross-scale recurrent weights.

        Simulates forward movement in a given direction using the unified
        W_rec matrix (1750×1750). Cross-scale connections allow information
        to flow between scales during preplay, providing richer predictions.

        Args:
            direction: Head direction index for selecting recurrent weights
            num_steps: Number of steps to simulate forward

        Returns:
            Predicted unified place cell activations (num_pc_total,)
        """
        # Clone current activations to avoid modifying the original
        place_cell_activations = self.place_cell_activations.clone()

        # Simulate forward for num_steps
        for _ in range(num_steps):
            previous_activations = place_cell_activations.clone()

            # Use unified recurrent weights for this direction
            # w_rec_unified[direction]: (num_pc_total, num_pc_total)
            # Matrix multiply: (num_pc_total, num_pc_total) @ (num_pc_total,) -> (num_pc_total,)
            updated = torch.matmul(
                self.w_rec_unified[direction],
                previous_activations
            )

            # Subtract previous activations (same as single-scale preplay)
            updated = updated - previous_activations

            # Apply ReLU then tanh
            place_cell_activations = torch.tanh(torch.relu(updated))
            if hasattr(self, "last_scale_preference") and self.last_scale_preference is not None:
                place_cell_activations = place_cell_activations * self.expand_scale_values_to_pc(
                    self.last_scale_preference
                )

        return place_cell_activations

    def preplay_from_state_batched(
        self,
        activations_batch: torch.Tensor,
        directions_batch: torch.Tensor,
        num_steps: int = 1
    ) -> torch.Tensor:
        """
        Batched preplay from arbitrary states (for stochastic sampling).

        Performs preplay for multiple trajectories in parallel, where each
        trajectory can have its own activation state and direction.

        Args:
            activations_batch: Initial activations (batch_size, num_pc_total)
            directions_batch: Direction indices (batch_size,)
            num_steps: Number of forward steps to simulate

        Returns:
            Updated activations after num_steps (batch_size, num_pc_total)
        """
        batch_size = activations_batch.shape[0]
        current_activations = activations_batch.clone()

        for _ in range(num_steps):
            previous_activations = current_activations.clone()

            # Vectorized: loop over unique directions (typically 8), not over batch items.
            # For each direction d, gather all trajectories assigned to it and do a
            # single (n_pc × n_pc) @ (n_pc × K) matmul instead of K separate mv calls.
            updated_batch = torch.zeros_like(current_activations)
            for d in torch.unique(directions_batch):
                mask = directions_batch == d            # (K,) bool
                # w_rec_unified[d]: (n_pc, n_pc),  prev[mask].T: (n_pc, K)
                updated_batch[mask] = torch.matmul(
                    self.w_rec_unified[d.item()], previous_activations[mask].T
                ).T

            # Subtract previous activations
            updated_batch = updated_batch - previous_activations

            # Apply ReLU then tanh
            current_activations = torch.tanh(torch.relu(updated_batch))

        return current_activations

    def unified_preplay_sampling(
        self,
        unified_rcn,
        n_hd: int = 8,
        num_steps: int = 3,
        discount_factor: float = 0.9,
        within_direction_beta: float = 2.0,
        num_samples: int = 10,
        sampling_strategy: str = "uniform",
        sampling_temperature: float = 1.0,
        debug: bool = False
    ) -> tuple:
        """
        Unified stochastic trajectory sampling preplay.

        Adapted from hierarchical_multiscale_preplay_sampling for unified architecture.
        Samples K trajectories per initial direction instead of exhaustive branching.

        Key differences from multi-scale version:
        - No scale loop (single unified representation)
        - No scale selection or entropy calculation
        - Direct Boltzmann distribution over directions
        - Cross-scale interactions handled automatically via W_rec_unified

        Args:
            unified_rcn: Unified reward cell network
            n_hd: Number of head directions
            num_steps: Number of preplay steps per trajectory
            discount_factor: Temporal discount factor (gamma)
            within_direction_beta: Inverse temperature for direction Boltzmann
            num_samples: Number of trajectories to sample per initial direction
            sampling_strategy: "uniform" (random turns) or "learned" (use W_rec probabilities)
            sampling_temperature: Softmax temperature for learned strategy
            debug: Whether to print debug information

        Returns:
            tuple: (final_direction_deg, expected_value, combined_vector,
                    discounted_returns_per_dir, direction_variance, direction_probs)
        """
        if debug:
            print(f"[UNIFIED-SAMPLING] Preplay: {n_hd} dirs × {num_samples} samples × {num_steps} steps")
            print(f"[UNIFIED-SAMPLING] Params: β={within_direction_beta} strategy={sampling_strategy}")
            print(f"[UNIFIED-SAMPLING] OPTIMIZED: Batched processing with batch_size={n_hd * num_samples}")

        # Build discount weights for all steps
        discount_weights = discount_factor ** torch.arange(num_steps, dtype=self.dtype, device=self.device)

        # Track sampling variance per direction
        sampling_variances = torch.zeros(n_hd, dtype=self.dtype, device=self.device)

        # ------------------------------------------------------------------
        # STAGE 1: Sample trajectories for all directions (BATCHED)
        # ------------------------------------------------------------------

        batch_size = n_hd * num_samples

        # Initialize all trajectories
        # Shape: (batch_size, num_pc_total)
        initial_activations = self.place_cell_activations.unsqueeze(0).expand(batch_size, -1).clone()

        # Initial directions: [0, 0, ..., 0, 1, 1, ..., 1, ..., 7, 7, ..., 7]
        # Shape: (batch_size,)
        initial_directions = torch.arange(n_hd, device=self.device).repeat_interleave(num_samples)
        current_directions = initial_directions.clone()

        # Initialize return and vector accumulators
        trajectory_returns = torch.zeros(batch_size, dtype=self.dtype, device=self.device)
        trajectory_vectors = torch.zeros(batch_size, 2, dtype=self.dtype, device=self.device)

        # Current activations for all trajectories
        activations_batch = initial_activations.clone()

        # Simulate num_steps forward for all trajectories in parallel
        for step in range(num_steps):
            # Sample direction for this step (if not first step)
            if step > 0:
                if sampling_strategy == "uniform":
                    # Uniform random: left, straight, right with equal probability
                    turn_options = torch.tensor([-1, 0, 1], dtype=torch.long, device=self.device)
                    chosen_turns = turn_options[torch.randint(0, 3, (batch_size,), device=self.device)]
                    current_directions = (current_directions + chosen_turns) % n_hd

                elif sampling_strategy == "learned":
                    # Use W_rec to inform turn probabilities
                    # For now, fall back to uniform (can implement learned later)
                    turn_options = torch.tensor([-1, 0, 1], dtype=torch.long, device=self.device)
                    chosen_turns = turn_options[torch.randint(0, 3, (batch_size,), device=self.device)]
                    current_directions = (current_directions + chosen_turns) % n_hd

                else:
                    raise ValueError(f"Unknown sampling_strategy: {sampling_strategy}")

            # Preplay one step for all trajectories in parallel
            activations_batch = self.preplay_from_state_batched(
                activations_batch, current_directions, num_steps=1
            )

            # Evaluate reward for all trajectories in parallel
            step_rewards = unified_rcn.compute_reward_activations_batched(activations_batch)  # (batch_size,)
            step_rewards = torch.nan_to_num(step_rewards)

            # Accumulate discounted returns
            step_weight = discount_weights[step]
            trajectory_returns += step_weight * step_rewards

            # Accumulate direction vectors
            step_angles = current_directions.float() * (2 * np.pi / n_hd)
            step_vectors = torch.stack([torch.cos(step_angles), torch.sin(step_angles)], dim=1)
            trajectory_vectors += step_vectors

        # Reshape results by (direction, sample)
        # trajectory_returns: (batch_size,) -> (n_hd, num_samples)
        returns_by_dir = trajectory_returns.view(n_hd, num_samples)
        # trajectory_vectors: (batch_size, 2) -> (n_hd, num_samples, 2)
        vectors_by_dir = trajectory_vectors.view(n_hd, num_samples, 2)

        # Compute mean and variance across samples for each direction
        macro_returns = torch.mean(returns_by_dir, dim=1)  # (n_hd,)
        macro_vectors = torch.mean(vectors_by_dir, dim=1)  # (n_hd, 2)

        # Variances: (n_hd,)
        return_variances = torch.var(returns_by_dir, dim=1)
        sampling_variances[:] = return_variances

        if debug:
            print(f"[UNIFIED-SAMPLING] Return variances: {return_variances}")

        # ------------------------------------------------------------------
        # STAGE 2: Boltzmann Distribution Over Directions
        # ------------------------------------------------------------------

        # Compute Boltzmann distribution over directions
        # P(d) = exp(beta * R_d) / Σ_d' exp(beta * R_d')
        returns_normalized = macro_returns - torch.max(macro_returns)  # Numerical stability
        boltzmann_weights = torch.exp(within_direction_beta * returns_normalized)
        direction_probs = boltzmann_weights / torch.clamp(torch.sum(boltzmann_weights), min=1e-9)

        if debug:
            print(f"[UNIFIED-SAMPLING] Direction probabilities: {direction_probs}")
            print(f"[UNIFIED-SAMPLING] Macro returns: {macro_returns}")

        # ------------------------------------------------------------------
        # STAGE 3: Compute Combined Vector and Expected Value
        # ------------------------------------------------------------------

        # Form combined movement vector: weighted sum of direction vectors
        combined_vector = torch.sum(direction_probs.unsqueeze(1) * macro_vectors, dim=0)

        # Compute expected value: weighted sum of discounted returns
        expected_value = torch.sum(direction_probs * macro_returns)

        # Robust fallback: if combined vector magnitude is near zero, use max-return direction
        combined_magnitude = torch.norm(combined_vector)
        epsilon = 1e-6
        if combined_magnitude < epsilon:
            max_idx = torch.argmax(macro_returns)
            combined_vector = macro_vectors[max_idx]
            expected_value = macro_returns[max_idx]
            if debug:
                print(f"[UNIFIED-SAMPLING] Fallback: vector near zero, using max-return direction")

        # Compute final direction angle from combined vector
        final_direction_rad = torch.atan2(combined_vector[1], combined_vector[0])

        # Convert to degrees [0, 360)
        final_direction_deg_tensor = final_direction_rad * (180.0 / np.pi)
        final_direction_deg_tensor = torch.where(
            final_direction_deg_tensor < 0,
            final_direction_deg_tensor + 360.0,
            final_direction_deg_tensor
        )

        if debug:
            print(f"[UNIFIED-SAMPLING] Final direction: {final_direction_deg_tensor.item():.1f}°")
            print(f"[UNIFIED-SAMPLING] Expected value: {expected_value.item():.3f}")
            avg_variance = torch.mean(sampling_variances).item()
            max_variance = torch.max(sampling_variances).item()
            print(f"[UNIFIED-SAMPLING] Variance: mean={avg_variance:.4f} max={max_variance:.4f}")

            if max_variance > 0.1:
                print(f"[UNIFIED-SAMPLING] WARNING: High sampling variance (max={max_variance:.4f})")
                print(f"  Suggested: Increase num_samples (current K={num_samples})")

        return (
            final_direction_deg_tensor,
            expected_value,
            combined_vector,
            macro_returns,
            sampling_variances,
            direction_probs
        )
