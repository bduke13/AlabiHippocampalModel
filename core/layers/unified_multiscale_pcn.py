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
from collections import deque
from numpy.random import default_rng
from typing import Optional, List, Dict, Union, Any

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
        learning_stdp_start_steps: int = 8000,
        enable_correlation_weighting: bool = True,
        correlation_window: int = 100,
        correlation_update_freq: int = 10,
        correlation_scaling: float = 2.0,
        min_correlation_weight: float = 0.1,
        correlation_threshold: float = 0.01,
        adjacency_learning_mode: str = "dense",
        adjacency_topk: Optional[int] = None,
        adjacency_activity_floor: float = 0.0,
        enable_adaptive_stdp: bool = False,
        adaptive_initial_lr: float = 0.1,
        adaptive_final_lr: float = 0.03,
        adaptive_decay_rate: float = 3.0,
        stdp_learning_rate: float = 0.01,
        tau_hd: float = 0.1,
        enable_connection_decay: bool = True,
        connection_decay_rate: float = 1e-5,
        cross_scale_inhibition_base_enabled: bool = True,
        use_bvc_context_modulation: bool = True,
        bvc_context_gain_floor: float = 0.15,
        bvc_context_gain_strength: float = 1.0,
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
        self.n_hd = int(n_hd)
        # BVC Modulation Parameters
        self.use_bvc_context_modulation = bool(use_bvc_context_modulation)
        self.bvc_context_gain_floor = float(min(0.95, max(0.0, bvc_context_gain_floor)))
        self.bvc_context_gain_strength = float(max(0.0, bvc_context_gain_strength))
        self.last_bvc_context_gain_per_pc = None
        self.last_bvc_context_gain_per_scale = None
        self.last_raw_bvc_abs_mean_per_scale = None
        self.last_raw_grid_abs_mean_per_scale = None
        self.last_raw_grid_share_per_scale = None
        self.last_balanced_bvc_abs_mean_per_scale = None
        self.last_balanced_grid_abs_mean_per_scale = None
        self.last_balanced_grid_share_per_scale = None
        self.last_mixed_bvc_abs_mean_per_scale = None
        self.last_mixed_grid_abs_mean_per_scale = None
        self.last_mixed_grid_share_per_scale = None
        self.last_bvc_gain_value_per_scale = None
        self.last_grid_gain_value_per_scale = None
        self.last_effective_grid_influence_mean_per_scale = None
        self.last_bvc_afferent_source_sum_per_scale = None
        self.last_grid_afferent_source_sum_per_scale = None
        self.last_bvc_afferent_inhibition_mean_per_scale = None
        self.last_grid_afferent_inhibition_mean_per_scale = None
        self.last_afferent_inhibition_mean_per_scale = None

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
        # Optional delayed plateau for the largest scale:
        # preserve the Gaussian near d_opt, then flatten only in very deep open space.
        largest_cfg = scale_configs[-1] if len(scale_configs) > 0 else {}
        self.large_scale_one_sided = bool(largest_cfg.get("large_scale_one_sided", False))
        self.large_scale_plateau = float(largest_cfg.get("large_scale_plateau", 1.0))
        self.large_scale_plateau = min(1.0, max(0.0, self.large_scale_plateau))
        self.large_scale_plateau_onset_sigma = float(
            max(0.0, largest_cfg.get("large_scale_plateau_onset_sigma", 1.0))
        )
        self.large_scale_plateau_full_sigma = float(
            max(
                self.large_scale_plateau_onset_sigma,
                largest_cfg.get("large_scale_plateau_full_sigma", 2.0),
            )
        )

        # Learning parameters
        self.enable_ojas = enable_ojas
        self.enable_stdp = enable_stdp
        assert gate_mode in ("normal", "no_gate_no_inhibition", "no_gate_with_inhibition"), \
            f"Unknown gate_mode '{gate_mode}'"
        self.gate_mode = gate_mode
        self.learning_stdp_start_steps = int(max(0, learning_stdp_start_steps))
        self.enable_correlation_weighting = bool(enable_correlation_weighting)
        self.correlation_window = int(max(10, correlation_window))
        self.correlation_update_freq = int(max(1, correlation_update_freq))
        self.correlation_scaling = float(correlation_scaling)
        self.min_correlation_weight = float(
            min(1.0, max(0.0, min_correlation_weight))
        )
        self.correlation_threshold = float(max(0.0, correlation_threshold))
        self.adjacency_learning_mode = str(adjacency_learning_mode).strip().lower()
        if self.adjacency_learning_mode not in {"dense", "topk"}:
            self.adjacency_learning_mode = "dense"
        self.adjacency_topk = (
            int(adjacency_topk) if adjacency_topk is not None else None
        )
        self.adjacency_activity_floor = float(max(0.0, adjacency_activity_floor))
        self.enable_adaptive_stdp = bool(enable_adaptive_stdp)
        self.adaptive_initial_lr = float(max(0.0, adaptive_initial_lr))
        self.adaptive_final_lr = float(max(0.0, adaptive_final_lr))
        self.adaptive_decay_rate = float(max(0.0, adaptive_decay_rate))
        self.stdp_learning_rate = float(max(0.0, stdp_learning_rate))
        self.tau_hd = float(max(1e-6, tau_hd))
        self.enable_connection_decay = bool(enable_connection_decay)
        self.connection_decay_rate = float(
            min(0.999999, max(0.0, connection_decay_rate))
        )
        self.learning_step_count = int(getattr(self, "learning_step_count", 0))
        self.cross_scale_inhibition_base_enabled = bool(
            cross_scale_inhibition_base_enabled
        )
        self.cross_scale_inhibition_lambda_base = 0.20

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
        self._configure_grid_balance_from_scale_configs()
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

                w_grid_block = self._initialize_grid_block_weights(
                    rng=rng,
                    scale_cfg=cfg,
                    num_pc=int(cfg["num_pc"]),
                    num_grid_cells=int(gc_end - gc_start),
                )
                self.w_grid[pc_start:pc_end, gc_start:gc_end] = w_grid_block

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
        self.within_scale_block_mask = None
        # Optional spatial visibility mask over recurrent place-cell connections.
        # Driver-side geometry code populates this from the active world's obstacle
        # layout so replay/preplay can suppress across-wall transitions without
        # hardcoding any environment-specific coordinates here.
        self.recurrent_visibility_mask = None

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
        self.eta_stdp = 0.3
        self.activation_history = deque(maxlen=self.correlation_window)
        self.correlation_matrix = torch.ones(
            self.num_pc_total,
            self.num_pc_total,
            dtype=dtype,
            device=device,
        ) * 0.5
        self.correlation_step_counter = 0
        self._correlation_weights_cache = torch.ones(
            self.num_pc_total,
            self.num_pc_total,
            dtype=dtype,
            device=device,
        )
        self._correlation_weights_dirty = True
        self.connection_strength_cache = torch.zeros(
            self.n_hd, dtype=dtype, device=device
        )
        self.strength_update_counter = 0
        self.strength_update_frequency = 10

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
        self.initial_w_in = torch.clone(self.w_in.data)
        if self.w_grid is not None:
            self.initial_w_grid = torch.clone(self.w_grid.data)
        else:
            self.initial_w_grid = None

        print(f"[UnifiedMultiScalePCN] Initialized with {self.num_pc_total} total cells")
        print(f"  Grid cells (total): {self.num_grid_total}")
        print(f"  BVC cells (total): {self.num_bvc_total}")
        print(f"  Scale boundaries: {self.scale_boundaries}")
        print(f"  Optimal distances: {self.d_opt.cpu().numpy()}")
        print(
            f"  Cross-scale inhibition per scale: {self.gamma_cross_per_scale.detach().cpu().numpy()}, "
            f"mean={self.gamma_cross:.4f}, sigma_tune_per_scale={self.sigma_tune_per_scale.detach().cpu().numpy()}"
        )

    def _ensure_learning_caches(self) -> None:
        """Initialize or repair learning-time state for older loaded pickles."""
        if not hasattr(self, "enable_correlation_weighting"):
            self.enable_correlation_weighting = False
        if not hasattr(self, "correlation_window"):
            self.correlation_window = 100
        if not hasattr(self, "correlation_update_freq"):
            self.correlation_update_freq = 10
        if not hasattr(self, "correlation_scaling"):
            self.correlation_scaling = 2.0
        if not hasattr(self, "min_correlation_weight"):
            self.min_correlation_weight = 0.1
        if not hasattr(self, "correlation_threshold"):
            self.correlation_threshold = 0.01
        if not hasattr(self, "adjacency_learning_mode"):
            self.adjacency_learning_mode = "dense"
        if not hasattr(self, "adjacency_topk"):
            self.adjacency_topk = None
        if not hasattr(self, "adjacency_activity_floor"):
            self.adjacency_activity_floor = 0.0
        if not hasattr(self, "enable_adaptive_stdp"):
            self.enable_adaptive_stdp = False
        if not hasattr(self, "adaptive_initial_lr"):
            self.adaptive_initial_lr = 0.1
        if not hasattr(self, "adaptive_final_lr"):
            self.adaptive_final_lr = 0.03
        if not hasattr(self, "adaptive_decay_rate"):
            self.adaptive_decay_rate = 3.0
        if not hasattr(self, "stdp_learning_rate"):
            self.stdp_learning_rate = 0.01
        if not hasattr(self, "tau_hd"):
            self.tau_hd = 0.1
        if not hasattr(self, "cross_scale_inhibition_base_enabled"):
            self.cross_scale_inhibition_base_enabled = True
        else:
            self.cross_scale_inhibition_base_enabled = bool(
                self.cross_scale_inhibition_base_enabled
            )
        if not hasattr(self, "cross_scale_inhibition_lambda_base"):
            self.cross_scale_inhibition_lambda_base = 0.20
        else:
            self.cross_scale_inhibition_lambda_base = float(
                min(1.0, max(0.0, self.cross_scale_inhibition_lambda_base))
            )


        if not isinstance(getattr(self, "activation_history", None), deque):
            existing_history = list(getattr(self, "activation_history", []))
            self.activation_history = deque(existing_history, maxlen=self.correlation_window)
        elif self.activation_history.maxlen != self.correlation_window:
            self.activation_history = deque(
                self.activation_history, maxlen=self.correlation_window
            )

        corr_shape = (self.num_pc_total, self.num_pc_total)
        if (
            not hasattr(self, "correlation_matrix")
            or self.correlation_matrix.shape != corr_shape
            or self.correlation_matrix.device != self.device
        ):
            self.correlation_matrix = torch.ones(
                corr_shape, dtype=self.dtype, device=self.device
            ) * 0.5
        else:
            self.correlation_matrix = self.correlation_matrix.to(
                device=self.device, dtype=self.dtype
            )

        if (
            not hasattr(self, "_correlation_weights_cache")
            or self._correlation_weights_cache.shape != corr_shape
            or self._correlation_weights_cache.device != self.device
        ):
            self._correlation_weights_cache = torch.ones(
                corr_shape, dtype=self.dtype, device=self.device
            )
        else:
            self._correlation_weights_cache = self._correlation_weights_cache.to(
                device=self.device, dtype=self.dtype
            )

        if not hasattr(self, "_correlation_weights_dirty"):
            self._correlation_weights_dirty = True
        if not hasattr(self, "correlation_step_counter"):
            self.correlation_step_counter = 0
        if (
            not hasattr(self, "connection_strength_cache")
            or self.connection_strength_cache.shape != (self.n_hd,)
            or self.connection_strength_cache.device != self.device
        ):
            self.connection_strength_cache = torch.zeros(
                self.n_hd, dtype=self.dtype, device=self.device
            )
        else:
            self.connection_strength_cache = self.connection_strength_cache.to(
                device=self.device, dtype=self.dtype
            )
        if not hasattr(self, "strength_update_counter"):
            self.strength_update_counter = 0
        if not hasattr(self, "strength_update_frequency"):
            self.strength_update_frequency = 10
        for attr_name in (
            "last_cross_scale_inhibition_mean_per_scale",
            "last_cross_scale_inhibition_peak_per_scale",
            "last_cross_scale_other_scale_activity_sum_per_scale",
            "last_cross_scale_effective_factor_per_scale",
            "last_raw_bvc_abs_mean_per_scale",
            "last_raw_grid_abs_mean_per_scale",
            "last_raw_grid_share_per_scale",
            "last_balanced_bvc_abs_mean_per_scale",
            "last_balanced_grid_abs_mean_per_scale",
            "last_balanced_grid_share_per_scale",
            "last_mixed_bvc_abs_mean_per_scale",
            "last_mixed_grid_abs_mean_per_scale",
            "last_mixed_grid_share_per_scale",
            "last_bvc_gain_value_per_scale",
            "last_grid_gain_value_per_scale",
            "last_effective_grid_influence_mean_per_scale",
            "last_bvc_afferent_source_sum_per_scale",
            "last_grid_afferent_source_sum_per_scale",
            "last_bvc_afferent_inhibition_mean_per_scale",
            "last_grid_afferent_inhibition_mean_per_scale",
            "last_afferent_inhibition_mean_per_scale",
        ):
            attr_value = getattr(self, attr_name, None)
            if (
                attr_value is None
                or attr_value.shape != (self.num_scales,)
                or attr_value.device != self.device
            ):
                setattr(
                    self,
                    attr_name,
                    torch.zeros(self.num_scales, dtype=self.dtype, device=self.device),
                )
            else:
                setattr(
                    self,
                    attr_name,
                    attr_value.to(device=self.device, dtype=self.dtype),
                )


    def reset_activations(self):
        """Reset activations and learning traces to match old non-unified behavior."""
        self.place_cell_activations.zero_()
        self.activation_update.zero_()
        self.place_cell_trace = None
        self._clear_cross_scale_inhibition_diagnostics()


        if getattr(self, "hd_cell_trace", None) is not None:
            self.hd_cell_trace.zero_()
        elif getattr(self, "enable_stdp", False):
            self.hd_cell_trace = torch.zeros(
                (self.n_hd, 1, 1), dtype=self.dtype, device=self.device
            )

        if hasattr(self, "connection_strength_cache"):
            self.connection_strength_cache.zero_()
            self.strength_update_counter = 0

    def update_correlation_tracking(self, pc_activations: torch.Tensor) -> None:
        """Update replay-correlation statistics used to weight STDP updates."""
        if not bool(getattr(self, "enable_correlation_weighting", False)):
            return

        self._ensure_learning_caches()
        threshold = float(getattr(self, "correlation_threshold", 0.01))
        thresholded = pc_activations * (pc_activations > threshold)
        self.activation_history.append(thresholded.clone().detach())

        self.correlation_step_counter += 1
        if (
            self.correlation_step_counter
            % int(max(1, getattr(self, "correlation_update_freq", 10)))
            == 0
            and len(self.activation_history) >= 10
        ):
            self.compute_correlation_matrix()

    def compute_correlation_matrix(self) -> None:
        """Compute the correlation matrix from recent unified PC activity."""
        self._ensure_learning_caches()
        if len(self.activation_history) < 10:
            return

        try:
            history_matrix = torch.stack(list(self.activation_history), dim=0)
            self.correlation_matrix = torch.corrcoef(history_matrix.T)
            self.correlation_matrix = torch.nan_to_num(
                self.correlation_matrix,
                nan=0.0,
                posinf=1.0,
                neginf=-1.0,
            )
            diagonal_indices = torch.arange(self.num_pc_total, device=self.device)
            self.correlation_matrix[diagonal_indices, diagonal_indices] = 1.0
            self._correlation_weights_dirty = True
        except Exception:
            self.correlation_matrix = torch.ones(
                self.num_pc_total,
                self.num_pc_total,
                dtype=self.dtype,
                device=self.device,
            ) * 0.5
            self._correlation_weights_dirty = True

    def get_correlation_weights(self) -> torch.Tensor:
        """Convert the correlation matrix into multiplicative STDP weights."""
        if not bool(getattr(self, "enable_correlation_weighting", False)):
            return torch.ones(
                self.num_pc_total,
                self.num_pc_total,
                dtype=self.dtype,
                device=self.device,
            )

        self._ensure_learning_caches()
        if not self._correlation_weights_dirty:
            return self._correlation_weights_cache

        sigmoid_corr = torch.sigmoid(
            float(getattr(self, "correlation_scaling", 2.0)) * self.correlation_matrix
        )
        min_weight = float(getattr(self, "min_correlation_weight", 0.1))
        self._correlation_weights_cache = min_weight + (
            (1.0 - min_weight) * sigmoid_corr
        )
        self._correlation_weights_dirty = False
        return self._correlation_weights_cache

    def apply_correlation_weighting_to_stdp(
        self, connection_update: torch.Tensor
    ) -> torch.Tensor:
        """Apply cached correlation weights to a unified STDP delta."""
        if not bool(getattr(self, "enable_correlation_weighting", False)):
            return connection_update
        return connection_update * self.get_correlation_weights().unsqueeze(0)

    def update_hd_eligibility_trace(
        self, hd_activations: Optional[torch.Tensor]
    ) -> None:
        """Update head-direction eligibility traces using the old v2 dynamics."""
        if hd_activations is None:
            return
        if getattr(self, "hd_cell_trace", None) is None:
            self.hd_cell_trace = torch.zeros(
                (self.n_hd, 1, 1), dtype=self.dtype, device=self.device
            )

        hd_activations_clean = torch.nan_to_num(hd_activations)
        hd_activations_expanded = hd_activations_clean.unsqueeze(1).unsqueeze(2)
        tau_hd = float(max(1e-6, getattr(self, "tau_hd", 0.1)))
        self.hd_cell_trace += (self.tau / tau_hd) * (
            hd_activations_expanded - self.hd_cell_trace
        )

    def compute_connection_strengths(
        self, direction_connections_batch: torch.Tensor
    ) -> torch.Tensor:
        """Vectorized connection-strength summary for adaptive STDP."""
        abs_connections = torch.abs(direction_connections_batch)
        significance_threshold = 0.0001
        significant_mask = abs_connections > significance_threshold
        significant_counts = significant_mask.sum(dim=(1, 2))
        significant_sums = (abs_connections * significant_mask).sum(dim=(1, 2))
        return torch.where(
            significant_counts > 0,
            significant_sums
            / torch.clamp(significant_counts.to(self.dtype), min=1.0),
            torch.zeros_like(significant_sums),
        )

    def get_adaptive_learning_rates(
        self, connection_strengths: torch.Tensor
    ) -> torch.Tensor:
        """Return one adaptive STDP learning rate per head-direction slice."""
        if not bool(getattr(self, "enable_adaptive_stdp", False)):
            return torch.full(
                (self.n_hd,),
                fill_value=float(getattr(self, "stdp_learning_rate", 0.01)),
                dtype=self.dtype,
                device=self.device,
            )

        decay_factor = torch.exp(
            -connection_strengths * float(getattr(self, "adaptive_decay_rate", 3.0))
        )
        return float(getattr(self, "adaptive_final_lr", 0.03)) + (
            float(getattr(self, "adaptive_initial_lr", 0.1))
            - float(getattr(self, "adaptive_final_lr", 0.03))
        ) * decay_factor

    def _configure_grid_balance_from_scale_configs(self) -> None:
        """
        Configure per-scale BVC/GC homeostatic gains from scale configs.

        Unified mode mixes BVC and GC drive inside one activation equation.
        Without gain balancing, whichever modality has the larger raw afferent
        magnitude dominates regardless of the configured `grid_influence`.
        """
        self.grid_balance_enabled_per_scale = [
            bool(cfg.get("grid_balance_modalities", True))
            for cfg in self.scale_configs
        ]
        self.grid_balance_ema_decay_per_scale = torch.tensor(
            [
                float(min(0.9999, max(0.0, cfg.get("grid_balance_ema", 0.95))))
                for cfg in self.scale_configs
            ],
            dtype=self.dtype,
            device=self.device,
        )
        self.grid_balance_min_gain_per_scale = torch.tensor(
            [
                float(max(1e-3, cfg.get("grid_balance_min_gain", 0.1)))
                for cfg in self.scale_configs
            ],
            dtype=self.dtype,
            device=self.device,
        )
        self.grid_balance_max_gain_per_scale = torch.tensor(
            [
                float(max(1.0, cfg.get("grid_balance_max_gain", 8.0)))
                for cfg in self.scale_configs
            ],
            dtype=self.dtype,
            device=self.device,
        )
        needs_reset = (
            not hasattr(self, "grid_balance_bvc_ema")
            or self.grid_balance_bvc_ema.numel() != self.num_scales
            or not hasattr(self, "grid_balance_gc_ema")
            or self.grid_balance_gc_ema.numel() != self.num_scales
        )
        if needs_reset:
            self.grid_balance_bvc_ema = torch.ones(
                self.num_scales, dtype=self.dtype, device=self.device
            )
            self.grid_balance_gc_ema = torch.ones(
                self.num_scales, dtype=self.dtype, device=self.device
            )
            self.grid_balance_initialized = False
        else:
            self.grid_balance_bvc_ema = self.grid_balance_bvc_ema.to(
                device=self.device, dtype=self.dtype
            )
            self.grid_balance_gc_ema = self.grid_balance_gc_ema.to(
                device=self.device, dtype=self.dtype
            )
        self.last_grid_gain_per_scale = [
            {"scale_idx": int(i), "bvc_gain": 1.0, "grid_gain": 1.0}
            for i in range(self.num_scales)
        ]

    def _is_learning_active(self) -> bool:
        """Return whether the online activation path is currently in learning mode."""
        return bool(self.enable_ojas or self.enable_stdp)

    def _is_stdp_learning_active(
        self,
        learning_active: Optional[bool] = None,
    ) -> bool:
        """
        Delay recurrent STDP until afferent fields have had time to stabilize.
        """
        if learning_active is None:
            learning_active = self._is_learning_active()
        if not learning_active:
            return True
        step = int(getattr(self, "learning_step_count", 0))
        start = int(getattr(self, "learning_stdp_start_steps", 0))
        return step >= start

    def _effective_grid_influence_per_pc(
        self,
        learning_active: Optional[bool] = None,
    ) -> torch.Tensor:
        """Return the configured grid/BVC mixing vector."""
        _ = learning_active
        return self.grid_influence_per_pc

    def _initialize_grid_block_weights(
        self,
        rng,
        scale_cfg: Dict,
        num_pc: int,
        num_grid_cells: int,
    ) -> torch.Tensor:
        """
        Initialize one scale-local GC->PC block.

        Mirrors the non-unified model's `balanced_modules` strategy so each PC
        starts with a diverse sample across GC modules rather than a purely
        Bernoulli draw over all cells.
        """
        ratio = float(scale_cfg.get("w_grid_init_ratio", 0.25))
        strategy = str(scale_cfg.get("w_grid_init_strategy", "balanced_modules")).strip().lower()
        num_modules = int(scale_cfg.get("num_modules", 0) or 0)
        cells_per_module = int(scale_cfg.get("cells_per_module", 0) or 0)
        use_balanced = (
            strategy == "balanced_modules"
            and num_modules > 0
            and cells_per_module > 0
            and (num_modules * cells_per_module) >= num_grid_cells
        )

        if not use_balanced:
            block = rng.binomial(n=1, p=ratio, size=(num_pc, num_grid_cells))
            return torch.tensor(block, dtype=self.dtype, device=self.device)

        total_gc = int(num_grid_cells)
        total_active = int(round(ratio * total_gc))
        total_active = max(0, min(total_active, total_gc))
        w_grid_np = np.zeros((num_pc, total_gc), dtype=np.int8)
        base_quota = total_active // num_modules
        remainder = total_active - (base_quota * num_modules)

        for pc_idx in range(num_pc):
            for module_idx in range(num_modules):
                module_start = module_idx * cells_per_module
                if module_start >= total_gc:
                    break
                module_size = min(cells_per_module, total_gc - module_start)
                quota = base_quota + (1 if module_idx < remainder else 0)
                quota = min(quota, module_size)
                if quota > 0:
                    chosen = rng.choice(module_size, size=quota, replace=False)
                    w_grid_np[pc_idx, module_start + chosen] = 1

        return torch.tensor(w_grid_np, dtype=self.dtype, device=self.device)

    def _balance_bvc_and_grid_drive(
        self,
        bvc_afferent_excitation: torch.Tensor,
        grid_afferent_excitation: torch.Tensor,
    ):
        """
        Homeostatically balance BVC and GC excitation per scale before mixing.

        The target magnitude is the geometric mean of the running BVC and GC
        excitation magnitudes, which keeps one modality from swamping the other
        while preserving overall scale.
        """
        balanced_bvc = bvc_afferent_excitation.clone()
        balanced_grid = grid_afferent_excitation.clone()
        bvc_gain_per_pc = torch.ones_like(self.place_cell_activations)
        grid_gain_per_pc = torch.ones_like(self.place_cell_activations)
        gain_logs = []

        if (
            self.grid_cell_activations is None
            or self.w_grid is None
            or self.num_scales <= 0
        ):
            self.last_grid_gain_per_scale = gain_logs
            return balanced_bvc, balanced_grid, bvc_gain_per_pc, grid_gain_per_pc

        initialized = bool(getattr(self, "grid_balance_initialized", False))
        for scale_idx in range(self.num_scales):
            start = self.scale_boundaries[scale_idx]
            end = self.scale_boundaries[scale_idx + 1]
            raw_bvc = torch.mean(torch.abs(bvc_afferent_excitation[start:end]))
            raw_grid = torch.mean(torch.abs(grid_afferent_excitation[start:end]))

            enabled = bool(self.grid_balance_enabled_per_scale[scale_idx])
            if enabled:
                if not initialized:
                    self.grid_balance_bvc_ema[scale_idx] = torch.clamp(raw_bvc, min=1e-6)
                    self.grid_balance_gc_ema[scale_idx] = torch.clamp(raw_grid, min=1e-6)
                else:
                    ema_decay = self.grid_balance_ema_decay_per_scale[scale_idx]
                    self.grid_balance_bvc_ema[scale_idx] = (
                        (ema_decay * self.grid_balance_bvc_ema[scale_idx])
                        + ((1.0 - ema_decay) * raw_bvc)
                    )
                    self.grid_balance_gc_ema[scale_idx] = (
                        (ema_decay * self.grid_balance_gc_ema[scale_idx])
                        + ((1.0 - ema_decay) * raw_grid)
                    )

                ema_bvc = torch.clamp(self.grid_balance_bvc_ema[scale_idx], min=1e-6)
                ema_grid = torch.clamp(self.grid_balance_gc_ema[scale_idx], min=1e-6)
                target = torch.sqrt(ema_bvc * ema_grid)
                min_gain = self.grid_balance_min_gain_per_scale[scale_idx]
                max_gain = self.grid_balance_max_gain_per_scale[scale_idx]
                bvc_gain = torch.clamp(target / ema_bvc, min=min_gain, max=max_gain)
                grid_gain = torch.clamp(target / ema_grid, min=min_gain, max=max_gain)
            else:
                bvc_gain = torch.tensor(1.0, dtype=self.dtype, device=self.device)
                grid_gain = torch.tensor(1.0, dtype=self.dtype, device=self.device)

            balanced_bvc[start:end] = balanced_bvc[start:end] * bvc_gain
            balanced_grid[start:end] = balanced_grid[start:end] * grid_gain
            bvc_gain_per_pc[start:end] = bvc_gain
            grid_gain_per_pc[start:end] = grid_gain
            gain_logs.append(
                {
                    "scale_idx": int(scale_idx),
                    "bvc_gain": float(bvc_gain.item()),
                    "grid_gain": float(grid_gain.item()),
                }
            )

        self.grid_balance_initialized = True
        self.last_grid_gain_per_scale = gain_logs
        return balanced_bvc, balanced_grid, bvc_gain_per_pc, grid_gain_per_pc

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
            pref[last_idx] = self._apply_large_scale_plateau_ramp(
                gaussian_pref=pref[last_idx],
                proximity_t=proximity_t,
                d_opt_t=self.d_opt[last_idx],
                sigma_tune_t=self.sigma_tune_per_scale[last_idx],
            )
        return torch.clamp(pref, min=0.0, max=1.0)
    
    def _compute_bvc_context_gain_per_pc(self, proximity: float, learning_active: bool = False) -> torch.Tensor:
        """
        Context-dependent gain applied only to boundary-driven (BVC) input.

        Uses the same bilateral-proximity Gaussian already used for scale preference,
        but treats it as a smooth gain rather than a hard eligibility mask.
        """
        if not self.use_bvc_context_modulation:
            return torch.ones(
                self.num_pc_total,
                dtype=self.dtype,
                device=self.device,
            )

        scale_preference = self.compute_scale_preference(proximity)
        gain_per_pc = self.expand_scale_values_to_pc(scale_preference)

        # Optional softening so it behaves like gain, not hard suppression
        floor = self.bvc_context_gain_floor
        strength = self.bvc_context_gain_strength
        gain_per_pc = floor + ((1.0 - floor) * gain_per_pc)
        gain_per_pc = 1.0 + strength * (gain_per_pc - 1.0)

        return torch.clamp(gain_per_pc, min=floor, max=1.0)

    def _apply_large_scale_plateau_ramp(
        self,
        gaussian_pref: torch.Tensor,
        proximity_t: torch.Tensor,
        d_opt_t: torch.Tensor,
        sigma_tune_t: torch.Tensor,
    ) -> torch.Tensor:
        """Blend the largest-scale Gaussian into a plateau only in deep open space."""
        plateau = torch.as_tensor(
            self.large_scale_plateau, dtype=self.dtype, device=self.device
        )
        onset = d_opt_t + self.large_scale_plateau_onset_sigma * sigma_tune_t
        full = d_opt_t + self.large_scale_plateau_full_sigma * sigma_tune_t
        denom = torch.clamp(full - onset, min=1e-6)
        alpha = torch.clamp((proximity_t - onset) / denom, min=0.0, max=1.0)
        # Smoothstep keeps the transition continuous without a hard switch at onset.
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        return (1.0 - alpha) * gaussian_pref + alpha * plateau

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
            pref_pc[last_start:last_end] = self._apply_large_scale_plateau_ramp(
                gaussian_pref=pref_pc[last_start:last_end],
                proximity_t=proximity_t,
                d_opt_t=self.d_opt_per_pc[last_start:last_end],
                sigma_tune_t=sigma_tune_per_pc[last_start:last_end],
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

    def compute_cross_scale_inhibition(
        self,
        proximity: float,
        afferent_excitation: Optional[torch.Tensor] = None,
        learning_active: Optional[bool] = None,
        from_activations: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
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
        _ref_activations = from_activations if from_activations is not None else self.place_cell_activations
        inhibition = torch.zeros_like(_ref_activations)
        other_scale_activity_sums = torch.zeros(
            self.num_scales, dtype=self.dtype, device=self.device
        )
        effective_factors = torch.zeros(
            self.num_scales, dtype=self.dtype, device=self.device
        )
        scale_preference = self.compute_scale_preference(proximity)
        effective_scale_preference = scale_preference
        # For each scale, compute how inappropriate the current distance is
        for scale_idx in range(self.num_scales):
            start = self.scale_boundaries[scale_idx]
            end = self.scale_boundaries[scale_idx + 1]
            # Mismatch derived from scale preference (high far from preferred d_opt).
            mismatch = 1.0 - effective_scale_preference[scale_idx]

            # Sum activations from OTHER scales
            other_scales_activation = 0.0
            for other_idx in range(self.num_scales):
                if other_idx != scale_idx:
                    other_start = self.scale_boundaries[other_idx]
                    other_end = self.scale_boundaries[other_idx + 1]
                    other_scales_activation += torch.sum(
                        _ref_activations[other_start:other_end]
                    )
            other_scale_activity_sums[scale_idx] = torch.nan_to_num(
                other_scales_activation,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

            # Apply inhibition to this scale
            gamma_cross_scale = self.gamma_cross_per_scale[scale_idx]
            effective_factor = self._compute_effective_cross_scale_factor(mismatch)
            effective_factors[scale_idx] = torch.nan_to_num(
                effective_factor,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            scale_inhibition = (
                gamma_cross_scale
                * effective_factor
                * other_scales_activation
            )
            inhibition[start:end] = scale_inhibition

        self.last_cross_scale_other_scale_activity_sum_per_scale = (
            other_scale_activity_sums.detach()
        )
        self.last_cross_scale_effective_factor_per_scale = (
            effective_factors.detach()
        )

        return inhibition

    def _compute_effective_cross_scale_factor(
        self,
        mismatch: torch.Tensor,
    ) -> torch.Tensor:
        """Blend in a fixed baseline cross-scale inhibition term when enabled."""
        if not bool(getattr(self, "cross_scale_inhibition_base_enabled", False)):
            return mismatch

        lambda_base = torch.as_tensor(
            getattr(self, "cross_scale_inhibition_lambda_base", 0.20),
            dtype=mismatch.dtype,
            device=mismatch.device,
        )
        return lambda_base + ((1.0 - lambda_base) * mismatch)

    def _clear_cross_scale_inhibition_diagnostics(self) -> None:
        """Reset live cross-scale inhibition telemetry for the current step."""
        zeros = torch.zeros(self.num_scales, dtype=self.dtype, device=self.device)
        self.last_cross_scale_inhibition_mean_per_scale = zeros.clone()
        self.last_cross_scale_inhibition_peak_per_scale = zeros.clone()
        self.last_cross_scale_other_scale_activity_sum_per_scale = zeros.clone()
        self.last_cross_scale_effective_factor_per_scale = zeros.clone()

    def _update_cross_scale_inhibition_applied_diagnostics(
        self,
        cross_scale_inhibition: torch.Tensor,
    ) -> None:
        """Store per-scale stats for the final live cross-scale inhibition term."""
        mean_values = torch.zeros(self.num_scales, dtype=self.dtype, device=self.device)
        peak_values = torch.zeros(self.num_scales, dtype=self.dtype, device=self.device)
        for scale_idx in range(self.num_scales):
            start = self.scale_boundaries[scale_idx]
            end = self.scale_boundaries[scale_idx + 1]
            block = cross_scale_inhibition[start:end]
            if block.numel() == 0:
                continue
            mean_values[scale_idx] = torch.nan_to_num(
                torch.mean(block),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            peak_values[scale_idx] = torch.nan_to_num(
                torch.max(torch.abs(block)),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
        self.last_cross_scale_inhibition_mean_per_scale = mean_values.detach()
        self.last_cross_scale_inhibition_peak_per_scale = peak_values.detach()

    def _apply_competition_stage(
        self,
        afferent_excitation: torch.Tensor,
        proximity: float,
        current_activations: torch.Tensor,
        activation_update_in: torch.Tensor,
        learning_active: bool = False,
        bvc_gain_per_pc: Optional[torch.Tensor] = None,
        grid_gain_per_pc: Optional[torch.Tensor] = None,
    ):
        """
        Stateless competition stage shared by normal inference and preplay.

        Takes pre-computed afferent excitation (PC-space), applies scale gating,
        afferent/recurrent/cross-scale inhibition, the IIR update, and normalization.
        Reads current_activations for inhibition terms instead of self.place_cell_activations,
        and uses activation_update_in as the IIR state instead of self.activation_update.
        Returns (new_activations, new_activation_update) without modifying any
        model state.

        Args:
            afferent_excitation: Pre-computed PC-space afferent drive (N,).
            proximity: Distance to nearest boundary in metres.
            current_activations: Imagined or real current PC activations (N,).
            activation_update_in: IIR integrator state to start from (N,).
            learning_active: Whether plasticity rules are active this step.
            bvc_gain_per_pc: Per-PC BVC gain from _balance_bvc_and_grid_drive (N,).
                Defaults to ones when not provided (preplay path).
            grid_gain_per_pc: Per-PC grid gain from _balance_bvc_and_grid_drive (N,).
                Defaults to ones when not provided (preplay path).

        Returns:
            new_activations: Reconstructed place-cell state (N,).
            new_activation_update: Updated IIR integrator state (N,).
        """
        if bvc_gain_per_pc is None:
            bvc_gain_per_pc = self._buf_ones
        if grid_gain_per_pc is None:
            grid_gain_per_pc = self._buf_ones

        effective_grid_influence_per_pc = self._effective_grid_influence_per_pc(
            learning_active=learning_active
        )
        self._clear_cross_scale_inhibition_diagnostics()

        # --- Scale gate ---
        scale_preference = self.compute_scale_preference(proximity)
        scale_preference_per_pc = self.compute_scale_preference_per_pc(proximity)
        self.last_scale_preference = scale_preference.detach()
        self.last_scale_preference_per_pc = scale_preference_per_pc.detach()
        self.last_proximity = float(proximity)

        # --- BVC afferent inhibition ---
        bvc_afferent_inhibition = self._buf_bvc_inh.zero_()
        bvc_source_sums = torch.zeros(self.num_scales, dtype=self.dtype, device=self.device)
        for scale_idx in range(self.num_scales):
            pc_start = self.scale_boundaries[scale_idx]
            pc_end = self.scale_boundaries[scale_idx + 1]
            bvc_start = self.bvc_boundaries[scale_idx]
            bvc_end = self.bvc_boundaries[scale_idx + 1]
            bvc_sum_scale = (
                torch.sum(self.bvc_activations[bvc_start:bvc_end])
                * bvc_gain_per_pc[pc_start]
            )
            bvc_source_sums[scale_idx] = bvc_sum_scale
            gamma_pb_scale = self.gamma_pb_per_pc[pc_start]
            bvc_afferent_inhibition[pc_start:pc_end] = (
                gamma_pb_scale * bvc_sum_scale
            )

        # --- Grid afferent inhibition ---
        grid_afferent_inhibition = self._buf_grid_inh.zero_()
        grid_source_sums = torch.zeros(self.num_scales, dtype=self.dtype, device=self.device)
        if self.grid_cell_activations is not None:
            for scale_idx in range(self.num_scales):
                pc_start = self.scale_boundaries[scale_idx]
                pc_end = self.scale_boundaries[scale_idx + 1]
                gc_start = self.grid_boundaries[scale_idx]
                gc_end = self.grid_boundaries[scale_idx + 1]
                gc_sum_scale = (
                    torch.sum(self.grid_cell_activations[gc_start:gc_end])
                    * grid_gain_per_pc[pc_start]
                )
                grid_source_sums[scale_idx] = gc_sum_scale
                gamma_pg_scale = self.gamma_pg_per_pc[pc_start]
                grid_afferent_inhibition[pc_start:pc_end] = (
                    gamma_pg_scale * gc_sum_scale
                )

        afferent_inhibition = (
            (1.0 - effective_grid_influence_per_pc) * bvc_afferent_inhibition
            + effective_grid_influence_per_pc * grid_afferent_inhibition
        )
        self._update_afferent_inhibition_scale_diagnostics(
            bvc_source_sums=bvc_source_sums,
            grid_source_sums=grid_source_sums,
            bvc_afferent_inhibition=bvc_afferent_inhibition,
            grid_afferent_inhibition=grid_afferent_inhibition,
            afferent_inhibition=afferent_inhibition,
        )

        # --- Within-scale recurrent inhibition (uses current_activations, not self.*) ---
        recurrent_inhibition = self._buf_rec_inh.zero_()
        for scale_idx in range(self.num_scales):
            pc_start = self.scale_boundaries[scale_idx]
            pc_end = self.scale_boundaries[scale_idx + 1]
            scale_sum = torch.sum(current_activations[pc_start:pc_end])
            gamma_pp_scale = self.gamma_pp_per_pc[pc_start]
            recurrent_inhibition[pc_start:pc_end] = gamma_pp_scale * scale_sum

        # --- Cross-scale inhibition (active in both learning and recall; ramped during learning) ---
        cross_scale_inhibition = self._buf_cross_inh.zero_()
        if self.gate_mode != "no_gate_no_inhibition":
            cross_scale_inhibition = self.compute_cross_scale_inhibition(
                proximity,
                afferent_excitation=afferent_excitation,
                learning_active=learning_active,
                from_activations=current_activations,
            )
        self._update_cross_scale_inhibition_applied_diagnostics(cross_scale_inhibition)

        # --- IIR update (stateless: operates on activation_update_in) ---
        activation_update = activation_update_in + self.tau_p * (
            -activation_update_in
            + afferent_excitation
            - afferent_inhibition
            - recurrent_inhibition
            - cross_scale_inhibition
        )

        # --- Nonlinearity ---
        new_activations = torch.tanh(torch.relu(activation_update))

        return new_activations, activation_update

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
        if self.enable_stdp:
            self.apply_connection_decay()
        if self.enable_stdp or self.enable_correlation_weighting:
            self._ensure_learning_caches()

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
        else:
            hd_activations_torch = None

        self.update_hd_eligibility_trace(hd_activations_torch)

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
        learning_active = self._is_learning_active()
        if learning_active:
            self.learning_step_count = int(getattr(self, "learning_step_count", 0)) + 1
        raw_bvc_afferent_excitation = torch.matmul(self.w_in, self.bvc_activations)
        raw_grid_afferent_excitation = torch.zeros_like(raw_bvc_afferent_excitation)
        if self.grid_cell_activations is not None and self.w_grid is not None:
            raw_grid_afferent_excitation = torch.matmul(self.w_grid, self.grid_cell_activations)

        (
            bvc_afferent_excitation,
            grid_afferent_excitation,
            bvc_gain_per_pc,
            grid_gain_per_pc,
        ) = self._balance_bvc_and_grid_drive(
            raw_bvc_afferent_excitation,
            raw_grid_afferent_excitation,
        )

        effective_grid_influence_per_pc = self._effective_grid_influence_per_pc(
            learning_active=learning_active
        )

        # NEW: context-dependent modulation on BVC drive only
        bvc_context_gain_per_pc = self._compute_bvc_context_gain_per_pc(
            proximity=proximity,
            learning_active=learning_active,
        )

        # Mix BVC and Grid inputs per place cell according to scale-config grid influence
        mixed_bvc_afferent_excitation = (
            (1.0 - effective_grid_influence_per_pc) * bvc_afferent_excitation * bvc_context_gain_per_pc
        )
        mixed_grid_afferent_excitation = (
            effective_grid_influence_per_pc * grid_afferent_excitation
        )
        afferent_excitation = mixed_bvc_afferent_excitation + mixed_grid_afferent_excitation

        # Optional debug cache
        self.last_bvc_context_gain_per_pc = bvc_context_gain_per_pc.detach()
        self.last_bvc_context_gain_per_scale = torch.stack([
            torch.mean(
                bvc_context_gain_per_pc[
                    self.scale_boundaries[s]:self.scale_boundaries[s + 1]
                ]
            )
            for s in range(self.num_scales)
        ]).detach()
        self._update_live_afferent_scale_diagnostics(
            raw_bvc_afferent_excitation=raw_bvc_afferent_excitation,
            raw_grid_afferent_excitation=raw_grid_afferent_excitation,
            balanced_bvc_afferent_excitation=bvc_afferent_excitation,
            balanced_grid_afferent_excitation=grid_afferent_excitation,
            mixed_bvc_afferent_excitation=mixed_bvc_afferent_excitation,
            mixed_grid_afferent_excitation=mixed_grid_afferent_excitation,
            bvc_gain_per_pc=bvc_gain_per_pc,
            grid_gain_per_pc=grid_gain_per_pc,
            effective_grid_influence_per_pc=effective_grid_influence_per_pc,
        )

        # Delegate to the shared stateless competition stage.
        new_activations, new_activation_update = (
            self._apply_competition_stage(
                afferent_excitation=afferent_excitation,
                proximity=proximity,
                current_activations=self.place_cell_activations,
                activation_update_in=self.activation_update,
                learning_active=learning_active,
                bvc_gain_per_pc=bvc_gain_per_pc,
                grid_gain_per_pc=grid_gain_per_pc,
            )
        )
        self.place_cell_activations = new_activations
        self.activation_update = new_activation_update
        has_place_cell_activity = bool(torch.any(self.place_cell_activations != 0).item())
        if has_place_cell_activity:
            self.update_correlation_tracking(self.place_cell_activations)
        stdp_learning_active = self._is_stdp_learning_active(
            learning_active=learning_active
        )
        self.last_learning_stdp_active = bool(
            self.enable_stdp and stdp_learning_active
        )

        # STDP updates (unified across all scales)
        if (
            self.enable_stdp
            and has_place_cell_activity
            and not collided
            and stdp_learning_active
        ):
            if self.place_cell_trace is None:
                self.place_cell_trace = torch.zeros(
                    self.num_pc_total, dtype=self.dtype, device=self.device
                )
            if self.hd_cell_trace is None:
                self.hd_cell_trace = torch.zeros(
                    (self.n_hd, 1, 1), dtype=self.dtype, device=self.device
                )
            gated_pc_activations = self.place_cell_activations
            self.last_learning_stdp_gate_mean = float(
                torch.mean(gated_pc_activations).item()
            )
            # Update eligibility trace for place cells
            self.place_cell_trace += (self.tau / 3) * (
                gated_pc_activations - self.place_cell_trace
            )

            # Update unified recurrent weights (cross-scale STDP)
            # Match the non-unified learning dynamics more closely by using the
            # HD eligibility trace rather than the instantaneous HD snapshot.
            hd_contrib = self.hd_cell_trace
            mode = str(getattr(self, "adjacency_learning_mode", "dense")).lower()
            if mode not in {"dense", "topk"}:
                mode = "dense"
            floor = float(getattr(self, "adjacency_activity_floor", 0.0))

            if bool(getattr(self, "enable_adaptive_stdp", False)):
                self.strength_update_counter += 1
                if (
                    self.strength_update_counter
                    % int(max(1, getattr(self, "strength_update_frequency", 10)))
                    == 0
                ):
                    self.connection_strength_cache.copy_(
                        self.compute_connection_strengths(self.w_rec_unified)
                    )
                lr_scale = self.get_adaptive_learning_rates(
                    self.connection_strength_cache
                ).view(self.n_hd, 1, 1)
            else:
                lr_scale = float(getattr(self, "eta_stdp", 0.3))

            recurrent_visibility_mask = self._get_recurrent_visibility_mask()
            recurrent_learning_mask = None
            if learning_active:
                recurrent_learning_mask = recurrent_visibility_mask

            active_mask = (
                gated_pc_activations > floor
                if floor > 0.0
                else gated_pc_activations > 0
            )

            if mode == "topk":
                active_indices = torch.nonzero(active_mask, as_tuple=False).flatten()
                topk = getattr(self, "adjacency_topk", None)
                if active_indices.numel() > 0 and topk is not None and int(topk) > 0:
                    k = min(int(topk), int(active_indices.numel()))
                    if k < int(active_indices.numel()):
                        active_scores = gated_pc_activations[active_indices]
                        keep_rel = torch.topk(active_scores, k=k).indices
                        sel_indices = active_indices[keep_rel]
                    else:
                        sel_indices = active_indices

                    pc_act_sel = gated_pc_activations[sel_indices]
                    pc_trace_sel = self.place_cell_trace[sel_indices]
                    delta_small = torch.ger(pc_act_sel, pc_trace_sel) - torch.ger(
                        pc_trace_sel, pc_act_sel
                    )
                    update_small = hd_contrib * delta_small.unsqueeze(0)

                    if self.enable_correlation_weighting:
                        correlation_weights = self.get_correlation_weights()
                        corr_small = correlation_weights.index_select(
                            0, sel_indices
                        ).index_select(1, sel_indices)
                        update_small = update_small * corr_small.unsqueeze(0)
                    if recurrent_learning_mask is not None:
                        learn_small = recurrent_learning_mask.index_select(
                            0, sel_indices
                        ).index_select(1, sel_indices)
                        update_small = update_small * learn_small.unsqueeze(0)
                    if recurrent_visibility_mask is not None:
                        vis_small = recurrent_visibility_mask.index_select(
                            0, sel_indices
                        ).index_select(1, sel_indices)
                        update_small = update_small * vis_small.unsqueeze(0)

                    scaled_small = (
                        update_small * lr_scale
                        if bool(getattr(self, "enable_adaptive_stdp", False))
                        else float(getattr(self, "eta_stdp", 0.3)) * update_small
                    )
                    row_idx = sel_indices.unsqueeze(1).expand(-1, sel_indices.numel())
                    col_idx = sel_indices.unsqueeze(0).expand(sel_indices.numel(), -1)
                    self.w_rec_unified[:, row_idx, col_idx] += scaled_small.type(
                        self.dtype
                    )
                    # Mark reinforced connections for decay protection
                    if hasattr(self, "_decay_protection_counter"):
                        self.mark_connections_reinforced(
                            self.w_rec_unified[:, row_idx, col_idx]
                        )
                # If no active subset, skip recurrent update.
            else:
                pc_act_for_adj = gated_pc_activations
                pc_trace_for_adj = self.place_cell_trace
                if floor > 0.0:
                    active_scale = active_mask.to(pc_act_for_adj.dtype)
                    pc_act_for_adj = pc_act_for_adj * active_scale
                    pc_trace_for_adj = pc_trace_for_adj * active_scale

                pc_act_mat = torch.ger(pc_act_for_adj, pc_trace_for_adj)
                pc_trace_mat = torch.ger(pc_trace_for_adj, pc_act_for_adj)
                update_rec = hd_contrib * (pc_act_mat - pc_trace_mat)
                update_rec = self.apply_correlation_weighting_to_stdp(update_rec)
                if recurrent_learning_mask is not None:
                    update_rec = update_rec * recurrent_learning_mask.unsqueeze(0)
                if recurrent_visibility_mask is not None:
                    update_rec = update_rec * recurrent_visibility_mask.unsqueeze(0)

                scaled_update_rec = (
                    update_rec * lr_scale
                    if bool(getattr(self, "enable_adaptive_stdp", False))
                    else float(getattr(self, "eta_stdp", 0.3)) * update_rec
                )
                self.w_rec_unified += scaled_update_rec.type(self.dtype)
                self.mark_connections_reinforced(scaled_update_rec)

            if recurrent_visibility_mask is not None:
                self.w_rec_unified *= recurrent_visibility_mask.unsqueeze(0)
        else:
            self.last_learning_stdp_gate_mean = 0.0

        # Oja's rule for input weights
        if self.enable_ojas and torch.any(self.place_cell_activations != 0):
            pc_activations_col = self.place_cell_activations.unsqueeze(1)
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
                grid_mix_bvc_block = (1.0 - effective_grid_influence_per_pc[pc_start:pc_end]).unsqueeze(1)
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
                    grid_mix_col = effective_grid_influence_per_pc[pc_start:pc_end].unsqueeze(1)
                    hebbian_gc   = torch.matmul(pc_col, gc_row)
                    decay_gc     = (1.0 / torch.clamp(alpha_block, min=1e-6)) * (pc_col**2) * w_block
                    weight_update_gc[pc_start:pc_end, gc_start:gc_end] = \
                        self.tau * (hebbian_gc - decay_gc) * grid_mix_col
                self.w_grid.data += weight_update_gc
                self.w_grid.data *= self.w_grid_block_mask.float()  # belt-and-suspenders
                self.w_grid.data  = torch.clamp(self.w_grid.data, min=0.0)

        self._update_grid_diagnostics(
            raw_bvc_afferent_excitation=raw_bvc_afferent_excitation,
            raw_grid_afferent_excitation=raw_grid_afferent_excitation,
            balanced_bvc_afferent_excitation=bvc_afferent_excitation,
            balanced_grid_afferent_excitation=grid_afferent_excitation,
            mixed_bvc_afferent_excitation=mixed_bvc_afferent_excitation,
            mixed_grid_afferent_excitation=mixed_grid_afferent_excitation,
            effective_grid_influence_per_pc=effective_grid_influence_per_pc,
            learning_active=learning_active,
        )

        return self.place_cell_activations

    def _update_grid_diagnostics(
        self,
        raw_bvc_afferent_excitation: torch.Tensor,
        raw_grid_afferent_excitation: torch.Tensor,
        balanced_bvc_afferent_excitation: torch.Tensor,
        balanced_grid_afferent_excitation: torch.Tensor,
        mixed_bvc_afferent_excitation: torch.Tensor,
        mixed_grid_afferent_excitation: torch.Tensor,
        effective_grid_influence_per_pc: torch.Tensor,
        learning_active: bool,
    ) -> None:
        """Store lightweight diagnostics to verify GC contribution in unified mode."""
        self._diag_step_count += 1
        if self._diag_step_count % self._diag_interval != 0:
            return  # Skip to avoid per-step GPU→CPU sync

        raw_bvc_abs_mean = float(torch.mean(torch.abs(raw_bvc_afferent_excitation)).item())
        raw_grid_abs_mean = float(torch.mean(torch.abs(raw_grid_afferent_excitation)).item())
        raw_denom = raw_bvc_abs_mean + raw_grid_abs_mean + 1e-12
        raw_grid_share = float(raw_grid_abs_mean / raw_denom)

        balanced_bvc_abs_mean = float(torch.mean(torch.abs(balanced_bvc_afferent_excitation)).item())
        balanced_grid_abs_mean = float(torch.mean(torch.abs(balanced_grid_afferent_excitation)).item())
        balanced_denom = balanced_bvc_abs_mean + balanced_grid_abs_mean + 1e-12
        balanced_grid_share = float(balanced_grid_abs_mean / balanced_denom)

        mixed_bvc_abs_mean = float(torch.mean(torch.abs(mixed_bvc_afferent_excitation)).item())
        mixed_grid_abs_mean = float(torch.mean(torch.abs(mixed_grid_afferent_excitation)).item())
        effective_denom = mixed_bvc_abs_mean + mixed_grid_abs_mean + 1e-12
        effective_grid_share = float(mixed_grid_abs_mean / effective_denom)

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
            raw_bvc_block = raw_bvc_afferent_excitation[pc_start:pc_end]
            raw_grid_block = raw_grid_afferent_excitation[pc_start:pc_end]
            balanced_bvc_block = balanced_bvc_afferent_excitation[pc_start:pc_end]
            balanced_grid_block = balanced_grid_afferent_excitation[pc_start:pc_end]
            mixed_bvc_block = mixed_bvc_afferent_excitation[pc_start:pc_end]
            mixed_grid_block = mixed_grid_afferent_excitation[pc_start:pc_end]
            raw_bvc_block_abs = float(torch.mean(torch.abs(raw_bvc_block)).item())
            raw_grid_block_abs = float(torch.mean(torch.abs(raw_grid_block)).item())
            balanced_bvc_block_abs = float(torch.mean(torch.abs(balanced_bvc_block)).item())
            balanced_grid_block_abs = float(torch.mean(torch.abs(balanced_grid_block)).item())
            mixed_bvc_block_abs = float(torch.mean(torch.abs(mixed_bvc_block)).item())
            mixed_grid_block_abs = float(torch.mean(torch.abs(mixed_grid_block)).item())
            raw_block_denom = raw_bvc_block_abs + raw_grid_block_abs + 1e-12
            balanced_block_denom = balanced_bvc_block_abs + balanced_grid_block_abs + 1e-12
            effective_block_denom = mixed_bvc_block_abs + mixed_grid_block_abs + 1e-12
            gains = (
                self.last_grid_gain_per_scale[scale_idx]
                if scale_idx < len(getattr(self, "last_grid_gain_per_scale", []))
                else {"bvc_gain": 1.0, "grid_gain": 1.0}
            )
            item = {
                "scale_idx": int(scale_idx),
                "raw_bvc_abs_mean": raw_bvc_block_abs,
                "raw_grid_abs_mean": raw_grid_block_abs,
                "raw_grid_share": float(raw_grid_block_abs / raw_block_denom),
                "balanced_bvc_abs_mean": balanced_bvc_block_abs,
                "balanced_grid_abs_mean": balanced_grid_block_abs,
                "balanced_grid_share": float(balanced_grid_block_abs / balanced_block_denom),
                "mixed_bvc_abs_mean": mixed_bvc_block_abs,
                "mixed_grid_abs_mean": mixed_grid_block_abs,
                "grid_share": float(mixed_grid_block_abs / effective_block_denom),
                "bvc_gain": float(gains.get("bvc_gain", 1.0)),
                "grid_gain": float(gains.get("grid_gain", 1.0)),
                "effective_grid_influence": float(
                    torch.mean(effective_grid_influence_per_pc[pc_start:pc_end]).item()
                ),
            }
            per_scale.append(item)

        self.last_grid_diagnostics = {
            "raw_bvc_abs_mean": raw_bvc_abs_mean,
            "raw_grid_abs_mean": raw_grid_abs_mean,
            "raw_grid_share": raw_grid_share,
            "balanced_bvc_abs_mean": balanced_bvc_abs_mean,
            "balanced_grid_abs_mean": balanced_grid_abs_mean,
            "balanced_grid_share": balanced_grid_share,
            "bvc_abs_mean": mixed_bvc_abs_mean,
            "grid_abs_mean": mixed_grid_abs_mean,
            "grid_share": effective_grid_share,
            "effective_grid_share": effective_grid_share,
            "effective_grid_influence": float(
                torch.mean(effective_grid_influence_per_pc).item()
            ),
            "learning_stdp_active": bool(
                getattr(self, "last_learning_stdp_active", True)
            ),
            "learning_stdp_gate_mean": float(
                getattr(self, "last_learning_stdp_gate_mean", 0.0)
            ),
            "learning_step_count": int(getattr(self, "learning_step_count", 0)),
            "learning_active": bool(learning_active),
            "gc_active_frac": gc_active_frac,
            "gc_mean": gc_mean,
            "gc_max": gc_max,
            "per_scale": per_scale,
        }

    def _update_live_afferent_scale_diagnostics(
        self,
        raw_bvc_afferent_excitation: torch.Tensor,
        raw_grid_afferent_excitation: torch.Tensor,
        balanced_bvc_afferent_excitation: torch.Tensor,
        balanced_grid_afferent_excitation: torch.Tensor,
        mixed_bvc_afferent_excitation: torch.Tensor,
        mixed_grid_afferent_excitation: torch.Tensor,
        bvc_gain_per_pc: torch.Tensor,
        grid_gain_per_pc: torch.Tensor,
        effective_grid_influence_per_pc: torch.Tensor,
    ) -> None:
        """Cache per-scale afferent-drive components for later diagnostics persistence."""
        raw_bvc_abs = torch.zeros(self.num_scales, dtype=self.dtype, device=self.device)
        raw_grid_abs = torch.zeros(self.num_scales, dtype=self.dtype, device=self.device)
        raw_grid_share = torch.zeros(self.num_scales, dtype=self.dtype, device=self.device)
        balanced_bvc_abs = torch.zeros(self.num_scales, dtype=self.dtype, device=self.device)
        balanced_grid_abs = torch.zeros(self.num_scales, dtype=self.dtype, device=self.device)
        balanced_grid_share = torch.zeros(self.num_scales, dtype=self.dtype, device=self.device)
        mixed_bvc_abs = torch.zeros(self.num_scales, dtype=self.dtype, device=self.device)
        mixed_grid_abs = torch.zeros(self.num_scales, dtype=self.dtype, device=self.device)
        mixed_grid_share = torch.zeros(self.num_scales, dtype=self.dtype, device=self.device)
        bvc_gain_values = torch.zeros(self.num_scales, dtype=self.dtype, device=self.device)
        grid_gain_values = torch.zeros(self.num_scales, dtype=self.dtype, device=self.device)
        effective_grid_influence = torch.zeros(self.num_scales, dtype=self.dtype, device=self.device)

        for scale_idx in range(self.num_scales):
            pc_start = self.scale_boundaries[scale_idx]
            pc_end = self.scale_boundaries[scale_idx + 1]

            raw_bvc_block = raw_bvc_afferent_excitation[pc_start:pc_end]
            raw_grid_block = raw_grid_afferent_excitation[pc_start:pc_end]
            balanced_bvc_block = balanced_bvc_afferent_excitation[pc_start:pc_end]
            balanced_grid_block = balanced_grid_afferent_excitation[pc_start:pc_end]
            mixed_bvc_block = mixed_bvc_afferent_excitation[pc_start:pc_end]
            mixed_grid_block = mixed_grid_afferent_excitation[pc_start:pc_end]

            raw_bvc_abs[scale_idx] = torch.mean(torch.abs(raw_bvc_block))
            raw_grid_abs[scale_idx] = torch.mean(torch.abs(raw_grid_block))
            balanced_bvc_abs[scale_idx] = torch.mean(torch.abs(balanced_bvc_block))
            balanced_grid_abs[scale_idx] = torch.mean(torch.abs(balanced_grid_block))
            mixed_bvc_abs[scale_idx] = torch.mean(torch.abs(mixed_bvc_block))
            mixed_grid_abs[scale_idx] = torch.mean(torch.abs(mixed_grid_block))

            raw_denom = raw_bvc_abs[scale_idx] + raw_grid_abs[scale_idx] + 1e-12
            balanced_denom = balanced_bvc_abs[scale_idx] + balanced_grid_abs[scale_idx] + 1e-12
            mixed_denom = mixed_bvc_abs[scale_idx] + mixed_grid_abs[scale_idx] + 1e-12
            raw_grid_share[scale_idx] = raw_grid_abs[scale_idx] / raw_denom
            balanced_grid_share[scale_idx] = balanced_grid_abs[scale_idx] / balanced_denom
            mixed_grid_share[scale_idx] = mixed_grid_abs[scale_idx] / mixed_denom

            bvc_gain_values[scale_idx] = torch.mean(bvc_gain_per_pc[pc_start:pc_end])
            grid_gain_values[scale_idx] = torch.mean(grid_gain_per_pc[pc_start:pc_end])
            effective_grid_influence[scale_idx] = torch.mean(
                effective_grid_influence_per_pc[pc_start:pc_end]
            )

        self.last_raw_bvc_abs_mean_per_scale = raw_bvc_abs.detach().clone()
        self.last_raw_grid_abs_mean_per_scale = raw_grid_abs.detach().clone()
        self.last_raw_grid_share_per_scale = raw_grid_share.detach().clone()
        self.last_balanced_bvc_abs_mean_per_scale = balanced_bvc_abs.detach().clone()
        self.last_balanced_grid_abs_mean_per_scale = balanced_grid_abs.detach().clone()
        self.last_balanced_grid_share_per_scale = balanced_grid_share.detach().clone()
        self.last_mixed_bvc_abs_mean_per_scale = mixed_bvc_abs.detach().clone()
        self.last_mixed_grid_abs_mean_per_scale = mixed_grid_abs.detach().clone()
        self.last_mixed_grid_share_per_scale = mixed_grid_share.detach().clone()
        self.last_bvc_gain_value_per_scale = bvc_gain_values.detach().clone()
        self.last_grid_gain_value_per_scale = grid_gain_values.detach().clone()
        self.last_effective_grid_influence_mean_per_scale = (
            effective_grid_influence.detach().clone()
        )

    def _update_afferent_inhibition_scale_diagnostics(
        self,
        bvc_source_sums: torch.Tensor,
        grid_source_sums: torch.Tensor,
        bvc_afferent_inhibition: torch.Tensor,
        grid_afferent_inhibition: torch.Tensor,
        afferent_inhibition: torch.Tensor,
    ) -> None:
        """Cache the exact afferent inhibition ingredients used this step."""
        bvc_inh_means = torch.zeros(self.num_scales, dtype=self.dtype, device=self.device)
        grid_inh_means = torch.zeros(self.num_scales, dtype=self.dtype, device=self.device)
        afferent_inh_means = torch.zeros(self.num_scales, dtype=self.dtype, device=self.device)

        for scale_idx in range(self.num_scales):
            pc_start = self.scale_boundaries[scale_idx]
            pc_end = self.scale_boundaries[scale_idx + 1]
            bvc_inh_means[scale_idx] = torch.mean(bvc_afferent_inhibition[pc_start:pc_end])
            grid_inh_means[scale_idx] = torch.mean(grid_afferent_inhibition[pc_start:pc_end])
            afferent_inh_means[scale_idx] = torch.mean(afferent_inhibition[pc_start:pc_end])

        self.last_bvc_afferent_source_sum_per_scale = bvc_source_sums.detach().clone()
        self.last_grid_afferent_source_sum_per_scale = grid_source_sums.detach().clone()
        self.last_bvc_afferent_inhibition_mean_per_scale = bvc_inh_means.detach().clone()
        self.last_grid_afferent_inhibition_mean_per_scale = grid_inh_means.detach().clone()
        self.last_afferent_inhibition_mean_per_scale = afferent_inh_means.detach().clone()

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

    def _get_recurrent_visibility_mask(self) -> Optional[torch.Tensor]:
        """Return recurrent visibility mask on this module's device/dtype if set."""
        mask = getattr(self, "recurrent_visibility_mask", None)
        if mask is None:
            return None
        if mask.shape != (self.num_pc_total, self.num_pc_total):
            return None
        return mask.to(device=self.device, dtype=self.dtype)

    def _get_within_scale_block_mask(self) -> torch.Tensor:
        """Return a cached block mask with ones only within the same scale."""
        mask = getattr(self, "within_scale_block_mask", None)
        if (
            mask is None
            or mask.shape != (self.num_pc_total, self.num_pc_total)
        ):
            mask = torch.zeros(
                (self.num_pc_total, self.num_pc_total),
                dtype=self.dtype,
                device=self.device,
            )
            for scale_idx in range(self.num_scales):
                start = self.scale_boundaries[scale_idx]
                end = self.scale_boundaries[scale_idx + 1]
                mask[start:end, start:end] = 1.0
            self.within_scale_block_mask = mask
        return mask.to(device=self.device, dtype=self.dtype)

    def _get_masked_recurrent_weights(self, direction: int) -> torch.Tensor:
        """
        Return the recurrent matrix for one head direction after applying the
        current spatial visibility mask, when available.
        """
        weights = self.w_rec_unified[int(direction)]
        mask = self._get_recurrent_visibility_mask()
        if mask is not None:
            weights = weights * mask
        return weights

    def apply_connection_decay(self):
        """Apply activity-dependent recurrent decay.

        Connections that were recently reinforced by STDP are protected
        from decay (synaptic tagging).  Only stale connections — those
        that have not received meaningful STDP reinforcement within the
        protection window — are decayed.
        """
        if not bool(getattr(self, "enable_connection_decay", True)):
            return

        decay_rate = float(getattr(self, "connection_decay_rate", 1e-5))
        if decay_rate <= 0.0:
            return

        # Lazy-init the reinforcement recency tracker.
        # Counts timesteps since last significant STDP update per connection.
        if not hasattr(self, "_decay_protection_counter"):
            self._decay_protection_counter = torch.zeros_like(
                self.w_rec_unified
            )
        # Increment age for all connections
        self._decay_protection_counter += 1

        # Only decay connections older than the protection window
        protection_window = int(
            getattr(self, "decay_protection_window", 500)
        )
        stale_mask = (self._decay_protection_counter > protection_window).float()
        self.w_rec_unified *= (1.0 - decay_rate * stale_mask)

    def mark_connections_reinforced(self, update: torch.Tensor):
        """Reset the decay protection counter for connections that received
        a meaningful STDP update this timestep."""
        if not hasattr(self, "_decay_protection_counter"):
            return
        # Threshold: any connection whose update magnitude exceeds a
        # small fraction of the current weight is considered reinforced.
        reinforced = (torch.abs(update) > 1e-6)
        self._decay_protection_counter[reinforced] = 0

    def _compute_scale_mass_batched(self, activations_batch: torch.Tensor) -> torch.Tensor:
        """Return per-scale activation mass for a batch of unified states."""
        if activations_batch.dim() == 1:
            activations_batch = activations_batch.unsqueeze(0)

        masses = []
        for scale_idx in range(self.num_scales):
            start = self.scale_boundaries[scale_idx]
            end = self.scale_boundaries[scale_idx + 1]
            masses.append(torch.sum(torch.abs(activations_batch[:, start:end]), dim=1))

        if not masses:
            return torch.zeros(
                (activations_batch.shape[0], 0),
                dtype=activations_batch.dtype,
                device=activations_batch.device,
            )
        return torch.stack(masses, dim=1)

    def _get_last_segmented_expression_gain_scale(self) -> Optional[torch.Tensor]:
        """
        Return the current expressed scale mass from the live activation state.

        This is a non-learning recall/preplay signal only. It does not use any
        cached Gaussian learning telemetry.
        """
        activations = getattr(self, "place_cell_activations", None)
        if activations is None or activations.numel() != self.num_pc_total:
            return None
        mass = self._compute_scale_mass_batched(
            activations.to(device=self.device, dtype=self.dtype).unsqueeze(0)
        ).squeeze(0)
        if mass.numel() != self.num_scales:
            return None
        mass_sum = torch.sum(mass)
        if float(mass_sum.item()) <= 1e-9:
            return None
        return mass / torch.clamp(mass_sum, min=1e-9)

    def get_segmented_scale_prior_batched(
        self,
        activations_batch: torch.Tensor,
        preference_mix: float = 0.5,
    ) -> torch.Tensor:
        """
        Return normalized expressed scale mass for a batch of unified states.

        `preference_mix` is retained only for API compatibility; non-learning
        segmented priors no longer blend in any cached Gaussian state.
        """
        _ = preference_mix
        if activations_batch.dim() == 1:
            activations_batch = activations_batch.unsqueeze(0)

        mass_prior = self._compute_scale_mass_batched(activations_batch)
        if mass_prior.shape[1] == 0:
            return mass_prior
        return mass_prior / torch.clamp(
            torch.sum(mass_prior, dim=1, keepdim=True), min=1e-9
        )

    def get_segmented_expression_gate_per_pc(
        self,
        activations: Optional[torch.Tensor] = None,
        preference_mix: float = 0.5,
    ) -> Optional[torch.Tensor]:
        """
        Return the per-PC segmented expression gate for replay/preplay.

        If an activation vector is supplied, infer the scale gate from that
        state's current expressed scale mass. Otherwise use the current live
        activation state's expressed scale mass.
        """
        if activations is None:
            scale_gate = self._get_last_segmented_expression_gain_scale()
            if scale_gate is None:
                return None
        else:
            act = activations.to(device=self.device, dtype=self.dtype)
            if act.dim() != 1 or act.numel() != self.num_pc_total:
                return None
            scale_gate = self.get_segmented_scale_prior_batched(
                act.unsqueeze(0),
                preference_mix=preference_mix,
            ).squeeze(0)

        scale_gate = torch.clamp(scale_gate, min=0.0, max=1.0)
        return self.expand_scale_values_to_pc(scale_gate)

    def _compute_preplay_scale_prior_batched(
        self,
        activations_batch: torch.Tensor,
        preference_mix: float = 0.5,
    ) -> torch.Tensor:
        """
        Return the current activation-mass scale prior for preplay.
        """
        return self.get_segmented_scale_prior_batched(
            activations_batch,
            preference_mix=preference_mix,
        )

    def _get_scale_block_bounds(self, scale_idx: int) -> tuple[int, int]:
        """Return [start, end) bounds for one scale block in the unified state."""
        start = int(self.scale_boundaries[int(scale_idx)])
        end = int(self.scale_boundaries[int(scale_idx) + 1])
        return start, end

    def _preplay_scale_block_from_state_batched(
        self,
        starting_activations: torch.Tensor,
        directions: torch.Tensor,
        scale_idx: int,
        num_steps: int = 1,
    ) -> torch.Tensor:
        """
        Simulate old-style preplay on one scale block only.

        This is the clean non-unified analogue inside the unified model:
        one scale proposes its own successor using only its within-scale
        recurrent block, and the imagined state for that rollout keeps all
        other scales silent.
        """
        if starting_activations.dim() == 1:
            starting_activations = starting_activations.unsqueeze(0)

        batch_size = int(starting_activations.shape[0])
        start, end = self._get_scale_block_bounds(scale_idx)
        current_block = starting_activations[:, start:end].clone()

        for _ in range(num_steps):
            previous_block = current_block.clone()
            updated_block = torch.zeros_like(previous_block)
            for d in torch.unique(directions):
                mask = directions == d
                recurrent = self._get_masked_recurrent_weights(int(d.item()))
                recurrent_block = recurrent[start:end, start:end]
                updated_block[mask] = torch.matmul(
                    recurrent_block,
                    previous_block[mask].T,
                ).T
            updated_block = updated_block - previous_block
            current_block = torch.tanh(torch.relu(updated_block))

        updated_state = torch.zeros_like(starting_activations)
        updated_state[:, start:end] = current_block
        return updated_state

    def _preplay_reconstructed_step(
        self,
        x_imagined: torch.Tensor,
        direction: int,
        proximity: float,
    ) -> torch.Tensor:
        """
        Single project → reconstruct preplay step.

        Projects x_imagined through the full cross-scale recurrent weights for
        the given head direction, then passes the proposal through a dedicated
        lighter imagined-transition reconstruction stage. The reconstruction is
        stateless: no self.place_cell_activations or self.activation_update is
        read or written.

        Args:
            x_imagined: Current imagined unified activation state (num_pc_total,).
            direction: Head-direction bin index (0..n_hd-1).
            proximity: Distance to nearest boundary in metres (use last_proximity
                from the most recent real inference step as a proxy).

        Returns:
            next_x: Reconstructed next imagined state (num_pc_total,).
        """
        # Project: full cross-scale recurrent matrix for this direction.
        recurrent = self._get_masked_recurrent_weights(direction)
        raw_projection = torch.matmul(recurrent, x_imagined)
        proposal = torch.relu(raw_projection)

        return self._preplay_reconstructed_competition_batched(
            afferent_excitation_batch=proposal.unsqueeze(0),
            proximity=proximity,
            current_activations_batch=x_imagined.unsqueeze(0),
        ).squeeze(0)

    def _build_preplay_rollout_context(self, proximity: float) -> Dict[str, Any]:
        """Precompute proximity-dependent tensors shared across one preplay call."""
        scale_preference = self.compute_scale_preference(proximity)
        effective_scale_preference = scale_preference
        return {
            "proximity": float(proximity),
            "effective_scale_preference": effective_scale_preference,
            "effective_grid_influence_per_pc": self._effective_grid_influence_per_pc(
                learning_active=False
            ),
            "preplay_afferent_inhibition_scale": float(
                max(0.0, getattr(self, "preplay_afferent_inhibition_scale", 0.0))
            ),
            "preplay_recurrent_inhibition_scale": float(
                max(0.0, getattr(self, "preplay_recurrent_inhibition_scale", 0.20))
            ),
            "preplay_cross_scale_inhibition_scale": float(
                max(0.0, getattr(self, "preplay_cross_scale_inhibition_scale", 1.0))
            ),
        }

    def _preplay_reconstructed_competition_batched(
        self,
        afferent_excitation_batch: torch.Tensor,
        proximity: float,
        current_activations_batch: torch.Tensor,
        preplay_context: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        Vectorized imagined-transition reconstruction stage for preplay states.

        Recurrent proposals still pass through a competition-aware shaping
        stage, but the preplay kernel is intentionally lighter than the online
        sensory update: afferent inhibition is removed by default and recurrent
        inhibition is reduced so imagined states do not collapse to zero.
        """
        if afferent_excitation_batch.dim() == 1:
            afferent_excitation_batch = afferent_excitation_batch.unsqueeze(0)
            current_activations_batch = current_activations_batch.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        ctx = (
            preplay_context
            if preplay_context is not None
            else self._build_preplay_rollout_context(proximity)
        )
        preplay_afferent_inhibition_scale = float(
            ctx["preplay_afferent_inhibition_scale"]
        )
        preplay_recurrent_inhibition_scale = float(
            ctx["preplay_recurrent_inhibition_scale"]
        )
        preplay_cross_scale_inhibition_scale = float(
            ctx["preplay_cross_scale_inhibition_scale"]
        )
        effective_grid_influence_per_pc = ctx["effective_grid_influence_per_pc"]
        effective_scale_preference = ctx["effective_scale_preference"]

        afferent_excitation = afferent_excitation_batch

        if preplay_afferent_inhibition_scale > 0.0:
            bvc_afferent_inhibition = torch.zeros(
                self.num_pc_total,
                dtype=self.dtype,
                device=self.device,
            )
            for scale_idx in range(self.num_scales):
                pc_start = self.scale_boundaries[scale_idx]
                pc_end = self.scale_boundaries[scale_idx + 1]
                bvc_start = self.bvc_boundaries[scale_idx]
                bvc_end = self.bvc_boundaries[scale_idx + 1]
                bvc_sum_scale = torch.sum(self.bvc_activations[bvc_start:bvc_end])
                gamma_pb_scale = self.gamma_pb_per_pc[pc_start]
                bvc_afferent_inhibition[pc_start:pc_end] = (
                    gamma_pb_scale * bvc_sum_scale
                )

            grid_afferent_inhibition = torch.zeros(
                self.num_pc_total,
                dtype=self.dtype,
                device=self.device,
            )
            if self.grid_cell_activations is not None:
                for scale_idx in range(self.num_scales):
                    pc_start = self.scale_boundaries[scale_idx]
                    pc_end = self.scale_boundaries[scale_idx + 1]
                    gc_start = self.grid_boundaries[scale_idx]
                    gc_end = self.grid_boundaries[scale_idx + 1]
                    gc_sum_scale = torch.sum(self.grid_cell_activations[gc_start:gc_end])
                    gamma_pg_scale = self.gamma_pg_per_pc[pc_start]
                    grid_afferent_inhibition[pc_start:pc_end] = (
                        gamma_pg_scale * gc_sum_scale
                    )

            afferent_inhibition = (
                preplay_afferent_inhibition_scale
                * (
                    (1.0 - effective_grid_influence_per_pc) * bvc_afferent_inhibition
                    + effective_grid_influence_per_pc * grid_afferent_inhibition
                )
            ).unsqueeze(0)
        else:
            afferent_inhibition = torch.zeros_like(afferent_excitation)

        recurrent_inhibition = torch.zeros_like(current_activations_batch)
        scale_sums = []
        for scale_idx in range(self.num_scales):
            pc_start = self.scale_boundaries[scale_idx]
            pc_end = self.scale_boundaries[scale_idx + 1]
            scale_sum = torch.sum(
                current_activations_batch[:, pc_start:pc_end],
                dim=1,
                keepdim=True,
            )
            scale_sums.append(scale_sum)
            if preplay_recurrent_inhibition_scale > 0.0:
                gamma_pp_scale = self.gamma_pp_per_pc[pc_start]
                recurrent_inhibition[:, pc_start:pc_end] = (
                    preplay_recurrent_inhibition_scale
                    * gamma_pp_scale
                    * scale_sum
                )

        cross_scale_inhibition = torch.zeros_like(current_activations_batch)
        if (
            preplay_cross_scale_inhibition_scale > 0.0
            and self.gate_mode != "no_gate_no_inhibition"
        ):
            scale_sums_tensor = torch.cat(scale_sums, dim=1)
            total_scale_sum = torch.sum(scale_sums_tensor, dim=1, keepdim=True)
            for scale_idx in range(self.num_scales):
                pc_start = self.scale_boundaries[scale_idx]
                pc_end = self.scale_boundaries[scale_idx + 1]
                mismatch = 1.0 - effective_scale_preference[scale_idx]
                effective_factor = self._compute_effective_cross_scale_factor(
                    mismatch
                )
                other_scales_activation = (
                    total_scale_sum - scale_sums_tensor[:, scale_idx : scale_idx + 1]
                )
                scale_inhibition = (
                    preplay_cross_scale_inhibition_scale
                    * self.gamma_cross_per_scale[scale_idx]
                    * effective_factor
                    * other_scales_activation
                )
                cross_scale_inhibition[:, pc_start:pc_end] = scale_inhibition

        activation_update = self.tau_p * (
            afferent_excitation
            - afferent_inhibition
            - recurrent_inhibition
            - cross_scale_inhibition
        )

        new_activations = torch.tanh(torch.relu(activation_update))
        if squeeze_output:
            return new_activations.squeeze(0)
        return new_activations

    def _preplay_reconstructed_step_batched(
        self,
        x_imagined_batch: torch.Tensor,
        directions: torch.Tensor,
        proximity: float,
        preplay_context: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        Batched wrapper around _preplay_reconstructed_step.

        Each sample is advanced with the same reconstructed imagined transition
        used by the single-state helper so exploit/preplay rollouts are scored
        under the shared competition-aware dynamics instead of raw recurrent
        projection alone.
        """
        if x_imagined_batch.dim() == 1:
            return self._preplay_reconstructed_step(
                x_imagined=x_imagined_batch,
                direction=int(directions.item()),
                proximity=proximity,
            )

        proposal_batch = torch.zeros_like(x_imagined_batch)
        for direction in range(int(self.n_hd)):
            batch_indices = torch.nonzero(
                directions == direction,
                as_tuple=False,
            ).squeeze(1)
            if batch_indices.numel() == 0:
                continue
            recurrent = self._get_masked_recurrent_weights(direction)
            raw_projection = torch.matmul(recurrent, x_imagined_batch[batch_indices].T).T
            proposal_batch[batch_indices] = torch.relu(raw_projection)
        return self._preplay_reconstructed_competition_batched(
            afferent_excitation_batch=proposal_batch,
            proximity=proximity,
            current_activations_batch=x_imagined_batch,
            preplay_context=preplay_context,
        )

    def _compute_scale_block_turn_probabilities_batched(
        self,
        activations_batch: torch.Tensor,
        current_directions: torch.Tensor,
        scale_idx: int,
        temperature: float = 1.0,
        unified_rcn=None,
        reward_vector: Optional[torch.Tensor] = None,
        gaussian_weights: Optional[torch.Tensor] = None,
        preplay_context: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        State-evaluation turn probabilities using reconstructed imagined steps.

        For each candidate turn direction, projects the full unified activation
        through the same reconstructed competition-aware transition used by the
        sampled rollout path, then scores the predicted next state via the same
        normalized reward contribution used by the rollout return accumulator.
        Raw dot products are kept only as a compatibility fallback when no
        reward model is provided.
        """
        if activations_batch.dim() == 1:
            activations_batch = activations_batch.unsqueeze(0)

        batch_size = int(activations_batch.shape[0])
        proximity = float(getattr(self, "last_proximity", 5.0))
        turn_options = torch.tensor([-1, 0, 1], dtype=torch.long, device=self.device)
        next_dirs = (current_directions.unsqueeze(1) + turn_options.unsqueeze(0)) % self.n_hd
        expanded_batch = activations_batch.unsqueeze(1).expand(
            batch_size, 3, activations_batch.shape[1]
        ).reshape(batch_size * 3, activations_batch.shape[1])
        flat_dirs = next_dirs.reshape(-1)
        x_next_all = self._preplay_reconstructed_step_batched(
            x_imagined_batch=expanded_batch,
            directions=flat_dirs,
            proximity=proximity,
            preplay_context=preplay_context,
        )
        if unified_rcn is not None:
            turn_logits = unified_rcn.compute_reward_contribution_for_scale_batched(
                x_next_all,
                scale_idx=int(scale_idx),
            ).view(batch_size, 3)
        elif reward_vector is not None:
            turn_logits = torch.matmul(x_next_all, reward_vector).view(batch_size, 3)
        elif gaussian_weights is not None:
            turn_logits = torch.matmul(x_next_all, gaussian_weights).view(batch_size, 3)
        else:
            turn_logits = torch.sum(x_next_all, dim=1).view(batch_size, 3)

        safe_temp = float(max(1e-6, temperature))
        turn_logits = turn_logits.to(
            dtype=self.dtype,
            device=self.device,
        ) / safe_temp

        return torch.softmax(turn_logits, dim=1)

    def unified_preplay_sampling(
        self,
        unified_rcn,
        n_hd: int = 8,
        num_steps: int = 3,
        discount_factor: float = 0.9,
        within_direction_beta: float = 2.0,
        scale_selection_beta: float = 2.0,
        ema_lambda: float = 0.25,
        prev_scale_entropies: Optional[torch.Tensor] = None,
        scale_reliability: Optional[torch.Tensor] = None,
        entropy_exponent: float = 2.0,
        reliability_exponent: float = 2.0,
        variance_lambda: float = 1.0,
        use_entropy_ema: bool = True,
        enable_scale_arbitration: bool = True,
        scale_prior_mix: float = 0.0,
        num_samples: int = 10,
        sampling_strategy: str = "uniform",
        sampling_temperature: float = 1.0,
        sample_aggregation: str = "mean",
        pc_centers: Optional[torch.Tensor] = None,
        pc_center_mask: Optional[torch.Tensor] = None,
        debug: bool = False
    ) -> tuple:
        """
        Clean hierarchical preplay for the unified model.

        Each scale still rolls out its own trajectories using only its own
        recurrent block, but unified preplay no longer performs a second
        entropy/reliability arbitration pass across scales. Cross-scale mixing
        is driven only by the segmented scale prior, which already reflects the
        current environmental regime.
        """
        sample_aggregation = str(sample_aggregation).strip().lower()
        if sample_aggregation not in {"mean", "max"}:
            raise ValueError(
                f"Unknown sample_aggregation: {sample_aggregation}"
            )

        eps = 1e-9
        discount_weights = discount_factor ** torch.arange(
            num_steps, dtype=self.dtype, device=self.device
        )
        scale_macro_returns = torch.zeros(
            (self.num_scales, n_hd),
            dtype=self.dtype,
            device=self.device,
        )
        scale_macro_vectors = torch.zeros(
            (self.num_scales, n_hd, 2),
            dtype=self.dtype,
            device=self.device,
        )
        scale_sampling_variances = torch.zeros(
            (self.num_scales, n_hd),
            dtype=self.dtype,
            device=self.device,
        )

        preplay_scale_gate = self.get_segmented_scale_prior_batched(
            self.place_cell_activations.unsqueeze(0),
            preference_mix=0.0,
        ).squeeze(0)
        scale_gate_weights = torch.clamp(preplay_scale_gate, min=0.0)
        gate_sum = torch.sum(scale_gate_weights)
        if (not torch.isfinite(gate_sum)) or float(gate_sum.item()) <= eps:
            scale_gate_weights = torch.full(
                (self.num_scales,),
                1.0 / float(max(1, self.num_scales)),
                dtype=self.dtype,
                device=self.device,
            )
        else:
            scale_gate_weights = scale_gate_weights / torch.clamp(gate_sum, min=eps)

        active_scale_indices = list(range(self.num_scales))
        rollout_proximity = float(getattr(self, "last_proximity", 5.0))
        preplay_context = self._build_preplay_rollout_context(rollout_proximity)
        turn_options = torch.tensor(
            [-1, 0, 1],
            dtype=torch.long,
            device=self.device,
        )

        for scale_idx in active_scale_indices:
            batch_size = int(n_hd * num_samples)
            # Initialise every trajectory from the full unified state (all scales),
            # not just the current scale's block.
            initial_activations = self.place_cell_activations.unsqueeze(0).expand(
                batch_size, -1
            ).clone()
            initial_directions = torch.arange(
                n_hd,
                device=self.device,
            ).repeat_interleave(num_samples)
            current_directions = initial_directions.clone()
            trajectory_returns = torch.zeros(batch_size, dtype=self.dtype, device=self.device)
            trajectory_vectors = torch.zeros(batch_size, 2, dtype=self.dtype, device=self.device)
            activations_batch = initial_activations.clone()

            for step in range(num_steps):
                if step > 0:
                    if sampling_strategy == "uniform":
                        chosen_turns = turn_options[
                            torch.randint(0, 3, (batch_size,), device=self.device)
                        ]
                        current_directions = (current_directions + chosen_turns) % n_hd
                    elif sampling_strategy == "learned":
                        turn_probs_batch = self._compute_scale_block_turn_probabilities_batched(
                            activations_batch=activations_batch,
                            current_directions=current_directions,
                            scale_idx=scale_idx,
                            temperature=sampling_temperature,
                            unified_rcn=unified_rcn,
                            preplay_context=preplay_context,
                        )
                        turn_indices = torch.multinomial(
                            turn_probs_batch,
                            num_samples=1,
                        ).squeeze(1)
                        chosen_turns = turn_options[turn_indices]
                        current_directions = (current_directions + chosen_turns) % n_hd
                    else:
                        raise ValueError(f"Unknown sampling_strategy: {sampling_strategy}")

                activations_batch = self._preplay_reconstructed_step_batched(
                    x_imagined_batch=activations_batch,
                    directions=current_directions,
                    proximity=rollout_proximity,
                    preplay_context=preplay_context,
                )

                step_scale_rewards = unified_rcn.compute_reward_contribution_for_scale_batched(
                    activations_batch,
                    scale_idx=int(scale_idx),
                )
                step_scale_rewards = torch.nan_to_num(step_scale_rewards)
                trajectory_returns += discount_weights[step] * step_scale_rewards

                step_angles = -current_directions.to(dtype=self.dtype) * (
                    2.0 * np.pi / float(n_hd)
                )
                step_vectors = torch.stack(
                    [torch.cos(step_angles), torch.sin(step_angles)],
                    dim=1,
                )
                trajectory_vectors += step_vectors

            returns_by_dir = trajectory_returns.view(n_hd, num_samples)
            vectors_by_dir = trajectory_vectors.view(n_hd, num_samples, 2)
            if sample_aggregation == "max":
                best_sample_idx = torch.argmax(returns_by_dir, dim=1)
                scale_macro_returns[scale_idx] = torch.max(returns_by_dir, dim=1).values
                scale_macro_vectors[scale_idx] = vectors_by_dir[
                    torch.arange(n_hd, device=self.device),
                    best_sample_idx,
                ]
            else:
                scale_macro_returns[scale_idx] = torch.mean(returns_by_dir, dim=1)
                scale_macro_vectors[scale_idx] = torch.mean(vectors_by_dir, dim=1)
            scale_sampling_variances[scale_idx] = torch.var(returns_by_dir, dim=1)


        scale_direction_probs = []

        for scale_idx in range(self.num_scales):
            scale_returns = scale_macro_returns[scale_idx]
            normalized_returns = scale_returns - torch.max(scale_returns)
            boltzmann_weights = torch.exp(within_direction_beta * normalized_returns)
            direction_probs_scale = boltzmann_weights / torch.clamp(
                torch.sum(boltzmann_weights),
                min=eps,
            )
            scale_direction_probs.append(direction_probs_scale)
        scale_weights = scale_gate_weights
        scale_entropies_smoothed = torch.zeros(
            self.num_scales,
            dtype=self.dtype,
            device=self.device,
        )

        direction_prob_matrix = torch.stack(scale_direction_probs, dim=0)
        joint_prob_matrix = scale_weights.unsqueeze(1) * direction_prob_matrix
        direction_probs = torch.sum(joint_prob_matrix, dim=0)
        direction_denoms = torch.clamp(direction_probs, min=eps)

        macro_returns = torch.sum(joint_prob_matrix * scale_macro_returns, dim=0) / direction_denoms
        macro_vectors = torch.sum(
            joint_prob_matrix.unsqueeze(2) * scale_macro_vectors,
            dim=0,
        ) / direction_denoms.unsqueeze(1)
        sampling_variances = torch.sum(
            joint_prob_matrix * scale_sampling_variances,
            dim=0,
        ) / direction_denoms
        direction_probs = torch.nan_to_num(
            direction_probs,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        direction_probs = torch.clamp(direction_probs, min=0.0)
        direction_prob_sum = torch.sum(direction_probs)
        direction_prob_sum_valid = bool(torch.isfinite(direction_prob_sum).item())
        if (not direction_prob_sum_valid) or float(direction_prob_sum.item()) <= eps:
            direction_probs = torch.full(
                (n_hd,),
                1.0 / float(max(1, n_hd)),
                dtype=self.dtype,
                device=self.device,
            )
        else:
            direction_probs = direction_probs / torch.clamp(direction_prob_sum, min=eps)
        macro_returns = torch.nan_to_num(
            macro_returns,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        macro_vectors = torch.nan_to_num(
            macro_vectors,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        sampling_variances = torch.nan_to_num(
            sampling_variances,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        joint_direction_scores = torch.nan_to_num(
            direction_probs * macro_returns,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        best_dir_idx = torch.argmax(joint_direction_scores)
        expected_value = torch.sum(direction_probs * macro_returns)
        combined_vector = torch.nan_to_num(
            torch.sum(
                direction_probs.unsqueeze(1) * macro_vectors,
                dim=0,
            ),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        if float(torch.norm(combined_vector).item()) < 1e-6:
            best_angle = -best_dir_idx.to(dtype=self.dtype) * (
                2.0 * np.pi / float(n_hd)
            )
            combined_vector = torch.stack(
                [torch.cos(best_angle), torch.sin(best_angle)]
            )

        final_direction_rad = torch.atan2(combined_vector[1], combined_vector[0])
        final_direction_deg_tensor = final_direction_rad * (180.0 / np.pi)
        final_direction_deg_tensor = torch.where(
            final_direction_deg_tensor < 0,
            final_direction_deg_tensor + 360.0,
            final_direction_deg_tensor,
        )
        self.last_preplay_commit_dir = int(best_dir_idx.item())
        self.last_preplay_action_deg = float(final_direction_deg_tensor.item())
        self.last_preplay_expected_value = float(expected_value.item())

        return (
            final_direction_deg_tensor,
            expected_value,
            combined_vector,
            macro_returns,
            macro_vectors,
            sampling_variances,
            direction_probs,
            scale_weights,
            scale_entropies_smoothed,
        )
