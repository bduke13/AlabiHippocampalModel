import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from core.layers.grid_cell_layer import GridCellLayer
from core.layers.multiscale_bvc import BoundaryVectorCellLayer


torch.manual_seed(5)


class UnifiedMultiScalePCN:
    """Unified multiscale PCN for open obstacle worlds."""

    def __init__(
        self,
        scale_configs: Sequence[Dict],
        bvc_layers: Optional[List[BoundaryVectorCellLayer]] = None,
        timestep: int = 96,
        n_hd: int = 8,
        n_res: int = 720,
        max_dist: float = 30.0,
        world_name: Optional[str] = None,
        enable_ojas: bool = False,
        enable_stdp: bool = False,
        w_in_init_ratio: float = 0.25,
        grid_influence: float = 0.3,
        gamma_pp: float = 0.5,
        gamma_cross: Optional[float] = None,
        gamma_pb: float = 0.3,
        gamma_pg: float = 0.3,
        alpha_pb: float = math.sqrt(0.5),
        alpha_pg: float = math.sqrt(0.5),
        grid_inhibition_mode: str = "sum",
        sigma_tune: float = 1.5,
        learning_stdp_start_steps: int = 0,
        eta_stdp: Union[float, Sequence[float]] = 0.3,
        tau_hd: float = 0.1,
        connection_decay_rate: Union[float, Sequence[float]] = 1e-4,
        bvc_context_modulation_mode: str = "bvc_context",
        bvc_context_gain_floor: float = 0.0,
        bvc_context_gain_strength: float = 1.0,
        bvc_excitation_modulation_floor: float = 0.0,
        enforce_grid_structural_mask: bool = False,
        stdp_rectified_scale_centering: bool = False,
        stdp_rectified_hd_gate: bool = False,
        stdp_winner_hd_gate: bool = False,
        stdp_min_input_mass: float = 0.0,
        enable_live_diagnostics: bool = False,
        random_seed: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.device = device or torch.device("cpu")
        self.dtype = dtype
        self.random_seed = None if random_seed is None else int(random_seed)
        self.n_hd = int(n_hd)
        self.n_res = int(n_res)
        self.max_dist = float(max_dist)
        self.world_name = world_name
        self.tau_p = 0.5
        self.tau = float(timestep) / 1000.0

        self.enable_ojas = bool(enable_ojas)
        self.enable_stdp = bool(enable_stdp)
        self.learning_stdp_start_steps = int(max(0, learning_stdp_start_steps))
        self.learning_step_count = 0

        self.tau_hd = float(max(1e-6, tau_hd))
        self.bvc_context_modulation_mode = self._normalize_bvc_context_modulation_mode(
            bvc_context_modulation_mode
        )
        self.bvc_context_gain_floor = float(max(0.0, bvc_context_gain_floor))
        self.bvc_context_gain_strength = float(max(0.0, bvc_context_gain_strength))
        self.bvc_excitation_modulation_floor = float(min(1.0, max(0.0, bvc_excitation_modulation_floor)))
        self.enforce_grid_structural_mask = bool(enforce_grid_structural_mask)
        self.clamp_afferent_weights_nonnegative = False
        self.stdp_rectified_scale_centering = bool(stdp_rectified_scale_centering)
        self.stdp_rectified_hd_gate = bool(stdp_rectified_hd_gate)
        self.stdp_winner_hd_gate = bool(stdp_winner_hd_gate)
        self.stdp_min_input_mass = float(max(0.0, stdp_min_input_mass))
        self.enable_live_diagnostics = bool(enable_live_diagnostics)
        self.grid_inhibition_mode = self._normalize_grid_inhibition_mode(grid_inhibition_mode)
        self.disable_cross_scale_inhibition = False

        gamma_cross_fallback = float(gamma_pp if gamma_cross is None else gamma_cross)
        self.scale_configs = [dict(cfg) for cfg in scale_configs]
        for cfg in self.scale_configs:
            cfg.setdefault("gamma_cross", cfg.get("gamma_pp", gamma_cross_fallback))
        self.num_scales = len(self.scale_configs)
        self.num_pc_per_scale = [int(cfg["num_pc"]) for cfg in self.scale_configs]
        self.num_pc_total = int(sum(self.num_pc_per_scale))
        self.num_place_cells_total = self.num_pc_total

        self.scale_boundaries = [0]
        for n_pc in self.num_pc_per_scale:
            self.scale_boundaries.append(self.scale_boundaries[-1] + int(n_pc))

        self.d_opt = torch.tensor(
            [float(cfg.get("d_opt", sigma_tune)) for cfg in self.scale_configs],
            dtype=self.dtype,
            device=self.device,
        )
        self.sigma_r_per_scale = torch.tensor(
            [float(cfg.get("sigma_r", 1.0)) for cfg in self.scale_configs],
            dtype=self.dtype,
            device=self.device,
        )
        sigma_tune_k = torch.tensor(
            [float(cfg.get("sigma_tune_k", 1.0)) for cfg in self.scale_configs],
            dtype=self.dtype,
            device=self.device,
        )
        self.sigma_tune_per_scale = torch.clamp(
            sigma_tune_k * self.sigma_r_per_scale,
            min=1e-3,
        )
        self.sigma_tune = float(torch.mean(self.sigma_tune_per_scale).item()) if self.num_scales else float(sigma_tune)

        self.gamma_pp_per_pc = self._expand_cfg_value("gamma_pp", gamma_pp)
        self.gamma_cross_per_pc = self._expand_cfg_value("gamma_cross", gamma_cross_fallback)
        self.gamma_pb_per_pc = self._expand_cfg_value("gamma_pb", gamma_pb)
        self.gamma_pg_per_pc = self._expand_cfg_value("gamma_pg", gamma_pg)
        self.grid_influence_per_pc = self._expand_cfg_value("grid_influence", grid_influence)
        self.alpha_pb_per_pc = self._expand_cfg_value("alpha_pb", alpha_pb)
        self.alpha_pg_per_pc = self._expand_cfg_value("alpha_pg", alpha_pg)
        self.eta_ojas_per_pc = self._expand_cfg_value("eta_ojas", 1.0)
        eta_ojas_gc_values = [
            float(cfg.get("eta_ojas_gc", cfg.get("eta_ojas", 1.0)))
            for cfg in self.scale_configs
        ]
        self.eta_ojas_gc_per_pc = self.expand_scale_values_to_pc(
            torch.tensor(eta_ojas_gc_values, dtype=self.dtype, device=self.device)
        ) if eta_ojas_gc_values else torch.zeros(0, dtype=self.dtype, device=self.device)

        self.gamma_pp = float(torch.mean(self.gamma_pp_per_pc).item()) if self.num_pc_total else float(gamma_pp)
        self.gamma_cross = float(torch.mean(self.gamma_cross_per_pc).item()) if self.num_pc_total else gamma_cross_fallback
        self.gamma_pb = float(torch.mean(self.gamma_pb_per_pc).item()) if self.num_pc_total else float(gamma_pb)
        self.gamma_pg = float(torch.mean(self.gamma_pg_per_pc).item()) if self.num_pc_total else float(gamma_pg)
        self.grid_influence = float(torch.mean(self.grid_influence_per_pc).item()) if self.num_pc_total else float(grid_influence)
        self.alpha_pb = float(torch.mean(self.alpha_pb_per_pc).item()) if self.num_pc_total else math.sqrt(0.5)
        self.alpha_pg = float(torch.mean(self.alpha_pg_per_pc).item()) if self.num_pc_total else math.sqrt(0.5)
        self.eta_ojas = float(torch.mean(self.eta_ojas_per_pc).item()) if self.num_pc_total else 1.0
        self.eta_ojas_gc = float(torch.mean(self.eta_ojas_gc_per_pc).item()) if self.num_pc_total else 1.0

        self.bvc_layers = bvc_layers or self._build_bvc_layers()
        self.num_bvc_per_scale = [int(layer.num_bvc) for layer in self.bvc_layers]
        self.num_bvc_total = int(sum(self.num_bvc_per_scale))
        self.bvc_boundaries = [0]
        for n_bvc in self.num_bvc_per_scale:
            self.bvc_boundaries.append(self.bvc_boundaries[-1] + int(n_bvc))

        self.grid_layers = self._build_grid_layers()
        self.num_grid_per_scale = [int(layer.total_grid_cells) for layer in self.grid_layers]
        self.num_grid_total = int(sum(self.num_grid_per_scale))
        self.grid_boundaries = [0]
        for n_grid in self.num_grid_per_scale:
            self.grid_boundaries.append(self.grid_boundaries[-1] + int(n_grid))

        rng = np.random.default_rng(self.random_seed)
        self.w_in = torch.zeros(
            (self.num_pc_total, self.num_bvc_total),
            dtype=self.dtype,
            device=self.device,
        )
        for scale_idx, cfg in enumerate(self.scale_configs):
            pc_start, pc_end = self.scale_boundaries[scale_idx:scale_idx + 2]
            bvc_start, bvc_end = self.bvc_boundaries[scale_idx:scale_idx + 2]
            ratio = float(cfg.get("w_in_init_ratio", w_in_init_ratio))
            block = rng.binomial(1, ratio, size=(pc_end - pc_start, bvc_end - bvc_start))
            self.w_in[pc_start:pc_end, bvc_start:bvc_end] = torch.tensor(
                block,
                dtype=self.dtype,
                device=self.device,
            )
        self.initial_w_in = self.w_in.clone()

        self.w_grid = torch.zeros(
            (self.num_pc_total, self.num_grid_total),
            dtype=self.dtype,
            device=self.device,
        )
        for scale_idx, cfg in enumerate(self.scale_configs):
            pc_start, pc_end = self.scale_boundaries[scale_idx:scale_idx + 2]
            grid_start, grid_end = self.grid_boundaries[scale_idx:scale_idx + 2]
            if grid_end <= grid_start:
                continue
            self.w_grid[pc_start:pc_end, grid_start:grid_end] = self._initialize_grid_block_weights(
                rng=rng,
                scale_cfg=cfg,
                num_pc=pc_end - pc_start,
                num_grid_cells=grid_end - grid_start,
            )
        self.initial_w_grid = self.w_grid.clone()

        self.place_cell_activations = torch.zeros(self.num_pc_total, dtype=self.dtype, device=self.device)
        self.prev_place_cell_activations = self.place_cell_activations.clone()
        self.activation_update = torch.zeros_like(self.place_cell_activations)
        self.bvc_activations = torch.zeros(self.num_bvc_total, dtype=self.dtype, device=self.device)
        self.raw_bvc_activations = self.bvc_activations.clone()
        self.grid_cell_activations = torch.zeros(self.num_grid_total, dtype=self.dtype, device=self.device)
        self.grid_cell_activations_effective = self.grid_cell_activations.clone()
        self.w_rec_unified = torch.zeros(
            (self.n_hd, self.num_pc_total, self.num_pc_total),
            dtype=self.dtype,
            device=self.device,
        )
        self.place_cell_trace = torch.zeros(self.num_pc_total, dtype=self.dtype, device=self.device)
        self.hd_cell_trace = torch.zeros((self.n_hd, 1, 1), dtype=self.dtype, device=self.device)
        self.configure_eta_stdp(eta_stdp)
        self.configure_connection_decay_rates(connection_decay_rate)

        self.last_grid_activations: List[torch.Tensor] = []
        self.last_grid_diagnostics = None
        self.last_scale_preference = torch.ones(self.num_scales, dtype=self.dtype, device=self.device)
        self.last_bvc_context_gain_per_scale = torch.ones(self.num_scales, dtype=self.dtype, device=self.device)
        self.last_bvc_excitation_modulation_per_scale = torch.ones(self.num_scales, dtype=self.dtype, device=self.device)
        self.last_bvc_plasticity_gate_per_scale = torch.ones(self.num_scales, dtype=self.dtype, device=self.device)
        self.last_oja_bvc_update_norm_per_scale = torch.zeros(self.num_scales, dtype=self.dtype, device=self.device)
        self.last_oja_gc_update_norm_per_scale = torch.zeros(self.num_scales, dtype=self.dtype, device=self.device)
        self.last_stdp_input_mass_per_scale = torch.zeros(self.num_scales, dtype=self.dtype, device=self.device)
        self.last_stdp_input_active_fraction_per_scale = torch.zeros(self.num_scales, dtype=self.dtype, device=self.device)
        self.last_stdp_hd_gate = torch.zeros(self.n_hd, dtype=self.dtype, device=self.device)
        self.last_competition_diagnostics = None
        self.last_learning_stdp_active = False
        self.last_stdp_transition_eligibility = 1.0
        self.last_connection_decay_scale = 0.0
        self.cache_preplay_transitions = False
        self._preplay_transitions_cache = None
        self.preplay_normalize_transitions = True
        self.preplay_global_score_normalization = True

        print(f"[UnifiedMultiScalePCN] Initialized with {self.num_pc_total} total cells")
        print(f"  Grid cells total: {self.num_grid_total}")
        print(f"  BVC cells total: {self.num_bvc_total}")
        print(f"  Scale boundaries: {self.scale_boundaries}")

    def reset_runtime_state(self, reset_traces: bool = True) -> None:
        """Clear transient neural state while preserving learned weights."""
        self.place_cell_activations = torch.zeros(self.num_pc_total, dtype=self.dtype, device=self.device)
        self.prev_place_cell_activations = self.place_cell_activations.clone()
        self.activation_update = torch.zeros_like(self.place_cell_activations)
        self.bvc_activations = torch.zeros(self.num_bvc_total, dtype=self.dtype, device=self.device)
        self.raw_bvc_activations = self.bvc_activations.clone()
        self.grid_cell_activations = torch.zeros(self.num_grid_total, dtype=self.dtype, device=self.device)
        self.grid_cell_activations_effective = self.grid_cell_activations.clone()
        if reset_traces:
            self.place_cell_trace = torch.zeros(self.num_pc_total, dtype=self.dtype, device=self.device)
            self.hd_cell_trace = torch.zeros((self.n_hd, 1, 1), dtype=self.dtype, device=self.device)
        self.last_grid_activations = [
            torch.zeros(self.grid_boundaries[i + 1] - self.grid_boundaries[i], dtype=self.dtype, device=self.device)
            for i in range(self.num_scales)
        ]
        self.last_grid_diagnostics = None
        self.last_scale_preference = torch.ones(self.num_scales, dtype=self.dtype, device=self.device)
        self.last_bvc_context_gain_per_scale = torch.ones(self.num_scales, dtype=self.dtype, device=self.device)
        self.last_bvc_excitation_modulation_per_scale = torch.ones(self.num_scales, dtype=self.dtype, device=self.device)
        self.last_bvc_plasticity_gate_per_scale = torch.ones(self.num_scales, dtype=self.dtype, device=self.device)
        self.last_oja_bvc_update_norm_per_scale = torch.zeros(self.num_scales, dtype=self.dtype, device=self.device)
        self.last_oja_gc_update_norm_per_scale = torch.zeros(self.num_scales, dtype=self.dtype, device=self.device)
        self._reset_stdp_input_diagnostics()
        self.last_competition_diagnostics = None
        self.last_learning_stdp_active = False
        self.last_stdp_transition_eligibility = 1.0
        self.last_connection_decay_scale = 0.0
        for layer in getattr(self, "bvc_layers", []):
            if hasattr(layer, "bvc_activations"):
                layer.bvc_activations = None

    def _invalidate_preplay_transition_cache(self) -> None:
        self._preplay_transitions_cache = None

    def _build_bvc_layers(self) -> List[BoundaryVectorCellLayer]:
        return [
            BoundaryVectorCellLayer(
                n_res=self.n_res,
                n_hd=self.n_hd,
                sigma_theta=float(cfg.get("sigma_theta", 8.0)),
                sigma_r=float(cfg.get("sigma_r", 1.0)),
                max_dist=float(cfg.get("max_dist", self.max_dist)),
                num_bvc_per_dir=int(cfg.get("num_bvc_per_dir", 50)),
                dtype=self.dtype,
                device=self.device,
            )
            for cfg in self.scale_configs
        ]

    def _build_grid_layers(self) -> List[GridCellLayer]:
        layers = []
        for cfg in self.scale_configs:
            num_modules = int(cfg.get("num_modules", cfg.get("num_grid_modules", 1)))
            cells_per_module = int(cfg.get("cells_per_module", 1))
            layers.append(
                GridCellLayer(
                    num_modules=num_modules,
                    cells_per_module=cells_per_module,
                    spread_range=tuple(cfg.get("spread_range", cfg.get("grid_spread_range", (1.2, 1.2)))),
                    scale_multiplier=float(cfg.get("scale_multiplier", cfg.get("grid_scale_multiplier", 1.0))),
                    translation_scale=float(cfg.get("translation_scale", cfg.get("grid_translation_scale", 1.0))),
                    threshold=float(cfg.get("threshold", cfg.get("grid_threshold", 0.7))),
                    threshold_type=str(cfg.get("threshold_type", "soft")),
                    normalization=str(cfg.get("normalization", "per-cell")),
                    world_name=str(cfg.get("world_name", self.world_name)) if cfg.get("world_name", self.world_name) else None,
                    mask_resolution=int(cfg.get("mask_resolution", 128)),
                    wall_split_thresh=float(cfg.get("wall_split_thresh", 0.2)),
                    smooth_sigma=float(cfg.get("smooth_sigma", 1.5)),
                    device=self.device.type,
                    dtype=self.dtype,
                )
            )
        return layers

    def _expand_cfg_value(self, key: str, fallback: float) -> torch.Tensor:
        blocks = [
            torch.full(
                (int(cfg["num_pc"]),),
                float(cfg.get(key, fallback)),
                dtype=self.dtype,
                device=self.device,
            )
            for cfg in self.scale_configs
        ]
        return torch.cat(blocks) if blocks else torch.zeros(0, dtype=self.dtype, device=self.device)

    def _initialize_grid_block_weights(
        self,
        rng: np.random.Generator,
        scale_cfg: Dict,
        num_pc: int,
        num_grid_cells: int,
    ) -> torch.Tensor:
        ratio = float(scale_cfg.get("w_grid_init_ratio", 0.25))
        num_modules = int(scale_cfg.get("num_modules", 0) or 0)
        cells_per_module = int(scale_cfg.get("cells_per_module", 0) or 0)
        if num_modules <= 0 or cells_per_module <= 0 or (num_modules * cells_per_module) < num_grid_cells:
            raise ValueError(
                "Grid weight initialization requires valid num_modules and cells_per_module "
                f"for {num_grid_cells} grid cells"
            )

        total_active = max(0, min(num_grid_cells, int(round(ratio * num_grid_cells))))
        base_quota = total_active // num_modules
        remainder = total_active - (base_quota * num_modules)
        weights = np.zeros((num_pc, num_grid_cells), dtype=np.int8)
        for pc_idx in range(num_pc):
            for module_idx in range(num_modules):
                module_start = module_idx * cells_per_module
                if module_start >= num_grid_cells:
                    break
                module_size = min(cells_per_module, num_grid_cells - module_start)
                quota = min(module_size, base_quota + (1 if module_idx < remainder else 0))
                if quota > 0:
                    chosen = rng.choice(module_size, size=quota, replace=False)
                    weights[pc_idx, module_start + chosen] = 1
        return torch.tensor(weights, dtype=self.dtype, device=self.device)

    def configure_eta_stdp(self, eta_stdp: Union[float, Sequence[float]]) -> None:
        if isinstance(eta_stdp, (list, tuple, np.ndarray, torch.Tensor)):
            values = [float(v) for v in eta_stdp]
        else:
            values = [float(cfg.get("eta_stdp", eta_stdp)) for cfg in self.scale_configs]
        if len(values) != self.num_scales:
            base = float(eta_stdp if not isinstance(eta_stdp, (list, tuple, np.ndarray, torch.Tensor)) else 0.3)
            values = [base] * self.num_scales
        self.eta_stdp_per_scale = torch.tensor(values, dtype=self.dtype, device=self.device)
        self.eta_stdp_per_pc = torch.cat([
            torch.full((int(cfg["num_pc"]),), max(0.0, values[idx]), dtype=self.dtype, device=self.device)
            for idx, cfg in enumerate(self.scale_configs)
        ]) if self.scale_configs else torch.zeros(0, dtype=self.dtype, device=self.device)
        self.eta_stdp_pair = torch.sqrt(torch.clamp(torch.outer(self.eta_stdp_per_pc, self.eta_stdp_per_pc), min=0.0))
        self.eta_stdp = float(np.mean(values)) if values else 0.0

    def configure_connection_decay_rates(self, connection_decay_rate: Union[float, Sequence[float]]) -> None:
        if isinstance(connection_decay_rate, (list, tuple, np.ndarray, torch.Tensor)):
            values = [float(v) for v in connection_decay_rate]
        else:
            values = [float(cfg.get("connection_decay_rate", connection_decay_rate)) for cfg in self.scale_configs]
        if len(values) != self.num_scales:
            base = float(connection_decay_rate if not isinstance(connection_decay_rate, (list, tuple, np.ndarray, torch.Tensor)) else 1e-5)
            values = [base] * self.num_scales
        values = [min(0.999999, max(0.0, v)) for v in values]
        self.connection_decay_rate_per_scale = torch.tensor(values, dtype=self.dtype, device=self.device)
        self.connection_decay_rate_per_pc = torch.cat([
            torch.full((int(cfg["num_pc"]),), values[idx], dtype=self.dtype, device=self.device)
            for idx, cfg in enumerate(self.scale_configs)
        ]) if self.scale_configs else torch.zeros(0, dtype=self.dtype, device=self.device)
        self.connection_decay_rate_pair = torch.sqrt(
            torch.clamp(torch.outer(self.connection_decay_rate_per_pc, self.connection_decay_rate_per_pc), min=0.0)
        )
        self.connection_decay_factor = 1.0 - self.connection_decay_rate_pair.unsqueeze(0)
        self.connection_decay_rate = float(np.mean(values)) if values else 0.0

    def _coerce_tensor(self, value, *, dtype: torch.dtype) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.detach().to(device=self.device, dtype=dtype)
        return torch.as_tensor(value, dtype=dtype, device=self.device)

    def expand_scale_values_to_pc(self, values_per_scale: torch.Tensor) -> torch.Tensor:
        expanded = torch.zeros(self.num_pc_total, dtype=self.dtype, device=self.device)
        for scale_idx in range(self.num_scales):
            start, end = self.scale_boundaries[scale_idx:scale_idx + 2]
            expanded[start:end] = values_per_scale[scale_idx]
        return expanded

    @staticmethod
    def _normalize_bvc_context_modulation_mode(mode: str) -> str:
        mode_key = str(mode or "bvc_context").strip().lower().replace("-", "_")
        aliases = {
            "bvc_context": "bvc_context",
            "pcn_excitation": "bvc_context",
            "excitation": "bvc_context",
            "excitation_only": "bvc_context",
            "none": "none",
            "off": "none",
            "disabled": "none",
            "disable": "none",
            "no_context": "none",
            "no_modulation": "none",
            "no_context_modulation": "none",
            "plasticity": "bvc_plasticity_gate",
            "plasticity_only": "bvc_plasticity_gate",
            "plasticity_gate": "bvc_plasticity_gate",
            "context_plasticity": "bvc_plasticity_gate",
            "context_plasticity_gate": "bvc_plasticity_gate",
            "learning_gate": "bvc_plasticity_gate",
            "bvc_plasticity": "bvc_plasticity_gate",
            "bvc_plasticity_gate": "bvc_plasticity_gate",
        }
        return aliases.get(mode_key, "bvc_context")

    @staticmethod
    def _normalize_grid_inhibition_mode(mode: str) -> str:
        mode_key = str(mode or "sum").strip().lower().replace("-", "_")
        aliases = {
            "sum": "sum",
            "summed": "sum",
            "summed_grid": "sum",
            "grid_sum": "sum",
            "activation_sum": "sum",
            "grid_activation_sum": "sum",
            "weighted_module": "weighted_module_baseline",
            "module_baseline": "weighted_module_baseline",
            "weighted_module_baseline": "weighted_module_baseline",
        }
        return aliases.get(mode_key, "sum")

    def compute_scale_preference(self, proximity: float) -> torch.Tensor:
        proximity_t = torch.as_tensor(proximity, dtype=self.dtype, device=self.device)
        raw_pref = torch.zeros(self.num_scales, dtype=self.dtype, device=self.device)
        for scale_idx, cfg in enumerate(self.scale_configs):
            scale_name = str(cfg.get("name", "")).strip().lower()
            sigma_r_value = float(
                cfg.get(
                    "sigma_r",
                    self.sigma_r_per_scale[scale_idx].item() if scale_idx < int(self.sigma_r_per_scale.numel()) else 1.0,
                )
            )

            if scale_name == "small":
                center_value = float(cfg.get("mod_d_opt", 0.0))
                sigma_value = max(1e-3, float(cfg.get("mod_sigma", sigma_r_value)))
                sigma_sq = torch.as_tensor(sigma_value * sigma_value, dtype=self.dtype, device=self.device)
                raw_pref[scale_idx] = torch.exp(-((proximity_t - center_value) ** 2) / (2.0 * sigma_sq))
                continue

            if scale_name == "medium":
                center_value = float(cfg.get("mod_d_opt", sigma_r_value))
                sigma_value = max(1e-3, float(cfg.get("mod_sigma", sigma_r_value)))
                sigma_sq = torch.as_tensor(sigma_value * sigma_value, dtype=self.dtype, device=self.device)
                raw_pref[scale_idx] = torch.exp(
                    -((proximity_t - center_value) ** 2) / (2.0 * sigma_sq)
                )
                continue

            if scale_name == "large":
                center_value = float(cfg.get("mod_d_opt", 0.0))
                sigma_value = max(1e-3, float(cfg.get("mod_sigma", sigma_r_value)))
                sigma_sq = torch.as_tensor(sigma_value * sigma_value, dtype=self.dtype, device=self.device)
                raw_pref[scale_idx] = 1.0 - torch.exp(
                    -((proximity_t - center_value) ** 2) / (2.0 * sigma_sq)
                )
                continue

            default_center = float(
                cfg.get(
                    "d_opt",
                    self.d_opt[scale_idx].item() if scale_idx < int(self.d_opt.numel()) else self.sigma_tune,
                )
            )
            default_sigma = float(
                cfg.get(
                    "sigma_tune_k",
                    1.0,
                )
            ) * sigma_r_value
            center_value = float(cfg.get("mod_d_opt", default_center))
            sigma_value = max(1e-3, float(cfg.get("mod_sigma", default_sigma)))
            sigma_sq = torch.as_tensor(sigma_value * sigma_value, dtype=self.dtype, device=self.device)
            raw_pref[scale_idx] = torch.exp(
                -((proximity_t - center_value) ** 2) / (2.0 * sigma_sq)
            )

        raw_pref = torch.clamp(raw_pref, min=0.0, max=1.0)
        return raw_pref

    def _compute_bvc_context_gain_per_scale(
        self,
        proximity: float,
        scale_preference: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pref = scale_preference if scale_preference is not None else self.compute_scale_preference(proximity)
        floor = float(max(0.0, getattr(self, "bvc_context_gain_floor", 0.0)))
        strength = float(max(0.0, getattr(self, "bvc_context_gain_strength", 1.0)))
        return torch.clamp(floor + (strength * pref), min=0.0, max=1.0)

    def _compute_bvc_context_gain_per_pc(
        self,
        proximity: float,
        gain_per_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        gain = gain_per_scale if gain_per_scale is not None else self._compute_bvc_context_gain_per_scale(proximity)
        return self.expand_scale_values_to_pc(gain)

    def _compute_bvc_excitation_modulation_per_pc(
        self,
        bvc_context_gain_per_pc: torch.Tensor,
    ) -> torch.Tensor:
        bvc_share = torch.clamp(1.0 - self.grid_influence_per_pc, min=0.0, max=1.0)
        if getattr(self, "bvc_context_modulation_mode", "bvc_context") != "bvc_context":
            return bvc_share
        floor = float(min(1.0, max(0.0, getattr(self, "bvc_excitation_modulation_floor", 0.0))))
        modulation = torch.clamp(bvc_share * bvc_context_gain_per_pc, min=0.0, max=1.0)
        return torch.clamp(floor + ((1.0 - floor) * modulation), min=0.0, max=1.0)

    def _grid_excitation_signal(self) -> torch.Tensor:
        return self.w_grid @ self.grid_cell_activations_effective

    def _is_stdp_learning_active(self, learning_active: bool) -> bool:
        if not learning_active:
            return True
        return int(self.learning_step_count) >= int(self.learning_stdp_start_steps)


    def _grid_activations_for_position(self, position: Optional[Sequence[float]]) -> torch.Tensor:
        if position is None:
            return torch.zeros(self.num_grid_total, dtype=self.dtype, device=self.device)
        blocks = []
        for layer in self.grid_layers:
            act = layer.get_grid_cell_activations(position).to(device=self.device, dtype=self.dtype)
            blocks.append(act)
        return torch.cat(blocks) if blocks else torch.zeros(0, dtype=self.dtype, device=self.device)

    def get_place_cell_activations(
        self,
        distances,
        position: Optional[Sequence[float]] = None,
        grid_activations: Optional[torch.Tensor] = None,
        hd_activations: Optional[torch.Tensor] = None,
        collided: bool = False,
        proximity: Optional[float] = None,
        learn: Optional[bool] = None,
        transition_eligibility: Optional[float] = None,
    ) -> torch.Tensor:
        learning_active = bool(learn) if learn is not None else bool(self.enable_ojas or self.enable_stdp)
        learning_active = learning_active and bool(self.enable_ojas or self.enable_stdp)
        self.last_oja_bvc_update_norm_per_scale = torch.zeros(self.num_scales, dtype=self.dtype, device=self.device)
        self.last_oja_gc_update_norm_per_scale = torch.zeros(self.num_scales, dtype=self.dtype, device=self.device)
        if transition_eligibility is None:
            stdp_transition_eligibility = 1.0
        else:
            stdp_transition_eligibility = min(1.0, max(0.0, float(transition_eligibility)))
        self.last_stdp_transition_eligibility = float(stdp_transition_eligibility)
        self.last_connection_decay_scale = 0.0
        if self.enable_stdp and learning_active and stdp_transition_eligibility > 1e-6:
            self.apply_connection_decay(decay_scale=stdp_transition_eligibility)
            self.last_connection_decay_scale = float(stdp_transition_eligibility)

        distances_t = self._coerce_tensor(distances, dtype=self.dtype).flatten()
        if proximity is None:
            proximity = float(torch.clamp(torch.min(distances_t), min=0.0, max=self.max_dist).item())

        self.prev_place_cell_activations = self.place_cell_activations.detach().clone()
        if hd_activations is not None:
            self.update_hd_trace(self._coerce_tensor(hd_activations, dtype=self.dtype))

        scale_preference = self.compute_scale_preference(float(proximity))
        self.last_scale_preference = scale_preference.detach()

        raw_bvc_blocks = [layer.get_bvc_activation(distances_t) for layer in self.bvc_layers]
        self.raw_bvc_activations = torch.cat(raw_bvc_blocks) if raw_bvc_blocks else torch.zeros(0, dtype=self.dtype, device=self.device)
        bvc_gain_scale = self._compute_bvc_context_gain_per_scale(float(proximity), scale_preference=scale_preference)
        self.bvc_activations = self.raw_bvc_activations

        if grid_activations is None:
            self.grid_cell_activations = self._grid_activations_for_position(position)
        else:
            self.grid_cell_activations = self._coerce_tensor(grid_activations, dtype=self.dtype).flatten()
        if int(self.grid_cell_activations.numel()) != int(self.num_grid_total):
            raise ValueError(
                f"Grid activation length mismatch: got {int(self.grid_cell_activations.numel())}, expected {self.num_grid_total}"
            )
        self.grid_cell_activations_effective = self.grid_cell_activations

        if learning_active:
            self.learning_step_count += 1

        raw_bvc_drive = self.w_in @ self.bvc_activations
        raw_grid_drive = self._grid_excitation_signal()
        grid_mix = self.grid_influence_per_pc
        bvc_context_gain_per_pc = self._compute_bvc_context_gain_per_pc(float(proximity), gain_per_scale=bvc_gain_scale)
        bvc_excitation_modulation_per_pc = self._compute_bvc_excitation_modulation_per_pc(
            bvc_context_gain_per_pc=bvc_context_gain_per_pc,
        )
        if getattr(self, "bvc_context_modulation_mode", "bvc_context") == "bvc_plasticity_gate":
            bvc_plasticity_gate_per_pc = self.expand_scale_values_to_pc(bvc_gain_scale)
        else:
            bvc_plasticity_gate_per_pc = torch.ones_like(self.place_cell_activations)
        self.last_bvc_context_gain_per_scale = bvc_gain_scale.detach()
        self.last_bvc_excitation_modulation_per_scale = torch.stack([
            torch.mean(bvc_excitation_modulation_per_pc[self.scale_boundaries[i]:self.scale_boundaries[i + 1]])
            for i in range(self.num_scales)
        ]).detach() if self.num_scales else torch.zeros(0, dtype=self.dtype, device=self.device)
        self.last_bvc_plasticity_gate_per_scale = torch.stack([
            torch.mean(bvc_plasticity_gate_per_pc[self.scale_boundaries[i]:self.scale_boundaries[i + 1]])
            for i in range(self.num_scales)
        ]).detach() if self.num_scales else torch.zeros(0, dtype=self.dtype, device=self.device)
        afferent_excitation = (raw_bvc_drive * bvc_excitation_modulation_per_pc) + (grid_mix * raw_grid_drive)

        new_activations, new_update = self._apply_competition_stage(
            afferent_excitation=afferent_excitation,
            current_activations=self.place_cell_activations,
            activation_update_in=self.activation_update,
        )
        self.place_cell_activations = torch.nan_to_num(new_activations)
        self.activation_update = torch.nan_to_num(new_update)

        has_activity = bool(torch.any(self.place_cell_activations > 0).item())

        stdp_active_candidate = bool(
            self.enable_stdp
            and learning_active
            and not collided
            and has_activity
            and stdp_transition_eligibility > 1e-6
            and self._is_stdp_learning_active(learning_active)
        )
        if stdp_active_candidate:
            self.last_learning_stdp_active = self._stdp_update(update_scale=stdp_transition_eligibility)
        else:
            self.last_learning_stdp_active = False
            self._reset_stdp_input_diagnostics()

        if bool(self.enable_ojas and learning_active and has_activity):
            self._oja_update(
                bvc_plasticity_gate_per_pc=bvc_plasticity_gate_per_pc,
            )

        self.last_grid_activations = [
            self.grid_cell_activations[self.grid_boundaries[i]:self.grid_boundaries[i + 1]].detach()
            for i in range(self.num_scales)
        ]
        self._update_grid_diagnostics(raw_bvc_drive, raw_grid_drive, afferent_excitation, grid_mix)
        return self.place_cell_activations

    def _apply_competition_stage(
        self,
        afferent_excitation: torch.Tensor,
        current_activations: torch.Tensor,
        activation_update_in: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bvc_inhibition = torch.zeros_like(current_activations)
        grid_inhibition = torch.zeros_like(current_activations)
        for scale_idx in range(self.num_scales):
            pc_start, pc_end = self.scale_boundaries[scale_idx:scale_idx + 2]
            bvc_start, bvc_end = self.bvc_boundaries[scale_idx:scale_idx + 2]
            grid_start, grid_end = self.grid_boundaries[scale_idx:scale_idx + 2]
            bvc_sum = torch.sum(self.bvc_activations[bvc_start:bvc_end])
            bvc_inhibition[pc_start:pc_end] = self.gamma_pb_per_pc[pc_start] * bvc_sum
            grid_inhibition[pc_start:pc_end] = self.gamma_pg_per_pc[pc_start:pc_end] * self._grid_inhibition_signal_for_scale(
                scale_idx,
                pc_start,
                pc_end,
                grid_start,
                grid_end,
            )

        grid_mix = self.grid_influence_per_pc
        afferent_inhibition = ((1.0 - grid_mix) * bvc_inhibition) + (grid_mix * grid_inhibition)

        total_activity = torch.sum(current_activations)
        within_scale_recurrent_inhibition = torch.zeros_like(current_activations)
        cross_scale_inhibition = torch.zeros_like(current_activations)
        recurrent_inhibition = torch.zeros_like(current_activations)
        for scale_idx in range(self.num_scales):
            pc_start, pc_end = self.scale_boundaries[scale_idx:scale_idx + 2]
            scale_activity = torch.sum(current_activations[pc_start:pc_end])
            cross_scale_activity = total_activity - scale_activity
            within_scale_recurrent_inhibition[pc_start:pc_end] = (
                self.gamma_pp_per_pc[pc_start:pc_end] * scale_activity
            )
            if not bool(getattr(self, "disable_cross_scale_inhibition", False)):
                cross_scale_inhibition[pc_start:pc_end] = (
                    self.gamma_cross_per_pc[pc_start:pc_end] * cross_scale_activity
                )
            recurrent_inhibition[pc_start:pc_end] = (
                within_scale_recurrent_inhibition[pc_start:pc_end]
                + cross_scale_inhibition[pc_start:pc_end]
            )

        activation_update = activation_update_in + self.tau_p * (
            -activation_update_in
            + afferent_excitation
            - afferent_inhibition
            - recurrent_inhibition
        )
        new_activations = torch.tanh(torch.relu(activation_update))
        self._update_competition_diagnostics(
            afferent_excitation=afferent_excitation,
            bvc_inhibition=bvc_inhibition,
            grid_inhibition=grid_inhibition,
            afferent_inhibition=afferent_inhibition,
            within_scale_recurrent_inhibition=within_scale_recurrent_inhibition,
            cross_scale_inhibition=cross_scale_inhibition,
            recurrent_inhibition=recurrent_inhibition,
            current_activations=current_activations,
            activation_update=activation_update,
            new_activations=new_activations,
        )
        return new_activations, activation_update

    def _block_stats(self, values: torch.Tensor) -> Dict[str, float]:
        block = torch.nan_to_num(values.detach()).flatten()
        if int(block.numel()) == 0:
            return {"mean": 0.0, "abs_mean": 0.0, "sum": 0.0, "max": 0.0, "std": 0.0}
        return {
            "mean": float(torch.mean(block).item()),
            "abs_mean": float(torch.mean(torch.abs(block)).item()),
            "sum": float(torch.sum(block).item()),
            "max": float(torch.max(block).item()),
            "std": float(torch.std(block, unbiased=False).item()) if int(block.numel()) > 1 else 0.0,
        }

    def _update_competition_diagnostics(
        self,
        afferent_excitation: torch.Tensor,
        bvc_inhibition: torch.Tensor,
        grid_inhibition: torch.Tensor,
        afferent_inhibition: torch.Tensor,
        within_scale_recurrent_inhibition: torch.Tensor,
        cross_scale_inhibition: torch.Tensor,
        recurrent_inhibition: torch.Tensor,
        current_activations: torch.Tensor,
        activation_update: torch.Tensor,
        new_activations: torch.Tensor,
    ) -> None:
        if not self.enable_live_diagnostics:
            return
        per_scale = []
        total_activity = float(torch.sum(torch.nan_to_num(current_activations)).item())
        for scale_idx in range(self.num_scales):
            start, end = self.scale_boundaries[scale_idx:scale_idx + 2]
            new_block = torch.nan_to_num(new_activations[start:end])
            current_block = torch.nan_to_num(current_activations[start:end])
            update_block = torch.nan_to_num(activation_update[start:end])
            afferent_block = torch.nan_to_num(afferent_excitation[start:end])
            afferent_inh_block = torch.nan_to_num(afferent_inhibition[start:end])
            within_recurrent_block = torch.nan_to_num(within_scale_recurrent_inhibition[start:end])
            cross_inhibition_block = torch.nan_to_num(cross_scale_inhibition[start:end])
            recurrent_block = torch.nan_to_num(recurrent_inhibition[start:end])
            net_drive = afferent_block - afferent_inh_block - recurrent_block
            afferent_stats = self._block_stats(afferent_block)
            bvc_inhibition_stats = self._block_stats(bvc_inhibition[start:end])
            grid_inhibition_stats = self._block_stats(grid_inhibition[start:end])
            afferent_inhibition_stats = self._block_stats(afferent_inh_block)
            within_recurrent_inhibition_stats = self._block_stats(within_recurrent_block)
            cross_scale_inhibition_stats = self._block_stats(cross_inhibition_block)
            recurrent_inhibition_stats = self._block_stats(recurrent_block)
            net_drive_stats = self._block_stats(net_drive)
            activation_update_stats = self._block_stats(update_block)
            total_inhibition_abs_mean = (
                afferent_inhibition_stats["abs_mean"] + recurrent_inhibition_stats["abs_mean"]
            )
            feedforward_to_total_inhibition_ratio = afferent_stats["abs_mean"] / max(total_inhibition_abs_mean, 1e-12)
            feedforward_to_afferent_inhibition_ratio = (
                afferent_stats["abs_mean"] / max(afferent_inhibition_stats["abs_mean"], 1e-12)
            )
            feedforward_to_recurrent_inhibition_ratio = (
                afferent_stats["abs_mean"] / max(recurrent_inhibition_stats["abs_mean"], 1e-12)
            )
            denom = max(1, int(new_block.numel()))
            active_gt0 = torch.count_nonzero(new_block > 0.0)
            active_gt01 = torch.count_nonzero(new_block > 0.01)
            peak = float(torch.max(new_block).item()) if int(new_block.numel()) else 0.0
            peak_gate = 0.10 * peak if peak > 0.0 else float("inf")
            active_gt_10pct_peak = (
                torch.count_nonzero(new_block >= peak_gate) if peak > 0.0 else torch.tensor(0, device=self.device)
            )
            per_scale.append(
                {
                    "scale_idx": int(scale_idx),
                    "current_mass": float(torch.sum(torch.abs(current_block)).item()),
                    "new_mass": float(torch.sum(torch.abs(new_block)).item()),
                    "new_mean": float(torch.mean(new_block).item()) if int(new_block.numel()) else 0.0,
                    "new_max": peak,
                    "active_count": int(active_gt0.item()),
                    "active_fraction": float(active_gt0.item() / denom),
                    "active_gt_0p01_fraction": float(active_gt01.item() / denom),
                    "active_gt_10pct_peak_fraction": float(active_gt_10pct_peak.item() / denom),
                    "afferent": afferent_stats,
                    "bvc_inhibition": bvc_inhibition_stats,
                    "grid_inhibition": grid_inhibition_stats,
                    "afferent_inhibition": afferent_inhibition_stats,
                    "within_scale_recurrent_inhibition": within_recurrent_inhibition_stats,
                    "cross_scale_inhibition": cross_scale_inhibition_stats,
                    "recurrent_inhibition": recurrent_inhibition_stats,
                    "total_inhibition_abs_mean": float(total_inhibition_abs_mean),
                    "excitation_inhibition_ratio": float(feedforward_to_total_inhibition_ratio),
                    "excitation_afferent_inhibition_ratio": float(feedforward_to_afferent_inhibition_ratio),
                    "excitation_recurrent_inhibition_ratio": float(feedforward_to_recurrent_inhibition_ratio),
                    "net_drive": net_drive_stats,
                    "activation_update": activation_update_stats,
                }
            )
        self.last_competition_diagnostics = {
            "total_activity": total_activity,
            "per_scale": per_scale,
        }

    def _grid_inhibition_signal_for_scale(
        self,
        scale_idx: int,
        pc_start: int,
        pc_end: int,
        grid_start: int,
        grid_end: int,
    ) -> torch.Tensor:
        grid_block = self.grid_cell_activations_effective[grid_start:grid_end]
        if int(grid_block.numel()) == 0:
            return torch.zeros(pc_end - pc_start, dtype=self.dtype, device=self.device)
        mode = self._normalize_grid_inhibition_mode(getattr(self, "grid_inhibition_mode", "sum"))
        if mode == "sum":
            grid_sum = torch.sum(grid_block)
            return grid_sum.expand(pc_end - pc_start).clone()
        return self._grid_module_feedforward_baseline(scale_idx, pc_start, pc_end, grid_start, grid_end)

    def _grid_module_feedforward_baseline(
        self,
        scale_idx: int,
        pc_start: int,
        pc_end: int,
        grid_start: int,
        grid_end: int,
    ) -> torch.Tensor:
        grid_block = self.grid_cell_activations_effective[grid_start:grid_end]
        if int(grid_block.numel()) == 0:
            return torch.zeros(pc_end - pc_start, dtype=self.dtype, device=self.device)

        cfg = self.scale_configs[scale_idx] if scale_idx < len(self.scale_configs) else {}
        num_modules = max(1, int(cfg.get("num_modules", 1)))
        cells_per_module = max(1, int(cfg.get("cells_per_module", int(grid_block.numel()))))
        w_block = self.w_grid[pc_start:pc_end, grid_start:grid_end]
        baseline = torch.zeros(pc_end - pc_start, dtype=self.dtype, device=self.device)

        for module_idx in range(num_modules):
            start = module_idx * cells_per_module
            if start >= int(grid_block.numel()):
                break
            end = min(start + cells_per_module, int(grid_block.numel()))
            module_mean = torch.mean(grid_block[start:end])
            baseline += torch.sum(w_block[:, start:end], dim=1) * module_mean
        return baseline

    def update_hd_trace(self, hd_activations: torch.Tensor) -> None:
        hd = torch.nan_to_num(hd_activations).view(-1)
        if int(hd.numel()) != self.n_hd:
            return
        self.current_hd_gate_source = hd.detach().clone()
        self.hd_cell_trace += (self.tau / self.tau_hd) * (
            hd.view(self.n_hd, 1, 1) - self.hd_cell_trace
        )

    def _reset_stdp_input_diagnostics(self) -> None:
        self.last_stdp_input_mass_per_scale = torch.zeros(self.num_scales, dtype=self.dtype, device=self.device)
        self.last_stdp_input_active_fraction_per_scale = torch.zeros(self.num_scales, dtype=self.dtype, device=self.device)
        self.last_stdp_hd_gate = torch.zeros(self.n_hd, dtype=self.dtype, device=self.device)

    def _rectified_scale_center_for_stdp(self, values: torch.Tensor) -> torch.Tensor:
        source = torch.nan_to_num(values).flatten()
        if not bool(getattr(self, "stdp_rectified_scale_centering", False)):
            return source
        centered = torch.zeros_like(source)
        for scale_idx in range(self.num_scales):
            start, end = self.scale_boundaries[scale_idx:scale_idx + 2]
            block = source[start:end]
            if int(block.numel()) == 0:
                continue
            centered[start:end] = torch.relu(block - torch.mean(block))
        return centered

    def _update_stdp_input_diagnostics(self, pc_act: torch.Tensor) -> None:
        mass = torch.zeros(self.num_scales, dtype=self.dtype, device=self.device)
        active_fraction = torch.zeros(self.num_scales, dtype=self.dtype, device=self.device)
        for scale_idx in range(self.num_scales):
            start, end = self.scale_boundaries[scale_idx:scale_idx + 2]
            block = pc_act[start:end]
            if int(block.numel()) == 0:
                continue
            mass[scale_idx] = torch.sum(block)
            active_fraction[scale_idx] = torch.mean((block > 0).to(dtype=self.dtype))
        self.last_stdp_input_mass_per_scale = mass.detach()
        self.last_stdp_input_active_fraction_per_scale = active_fraction.detach()

    def _hd_gate_for_stdp(self) -> torch.Tensor:
        if bool(getattr(self, "stdp_winner_hd_gate", False)):
            source = torch.nan_to_num(
                getattr(self, "current_hd_gate_source", self.hd_cell_trace.reshape(-1))
            ).view(-1)
            gate = torch.zeros(self.n_hd, dtype=self.dtype, device=self.device)
            if int(source.numel()) == self.n_hd:
                gate[int(torch.argmax(source).item())] = 1.0
            return gate.view(self.n_hd, 1, 1)
        hd = torch.nan_to_num(self.hd_cell_trace)
        if not bool(getattr(self, "stdp_rectified_hd_gate", False)):
            return hd
        return torch.relu(hd)

    def _stdp_update(self, update_scale: float = 1.0) -> bool:
        self.place_cell_trace += (self.tau / 3.0) * (
            self.place_cell_activations - self.place_cell_trace
        )
        update_scale = min(1.0, max(0.0, float(update_scale)))
        if update_scale <= 1e-6:
            self._reset_stdp_input_diagnostics()
            return False
        pc_act = self._rectified_scale_center_for_stdp(self.place_cell_activations)
        pc_trace = self._rectified_scale_center_for_stdp(self.place_cell_trace)
        self._update_stdp_input_diagnostics(pc_act)
        hd_gate = self._hd_gate_for_stdp()
        self.last_stdp_hd_gate = hd_gate.reshape(-1).detach()

        min_input_mass = float(max(0.0, getattr(self, "stdp_min_input_mass", 0.0)))
        if float(torch.sum(pc_act).item()) < min_input_mass:
            return False

        pair_update = torch.outer(pc_act, pc_trace)
        pair_update.sub_(torch.outer(pc_trace, pc_act))
        pair_update.mul_(self.eta_stdp_pair)
        pair_update.mul_(update_scale)
        self.w_rec_unified.addcmul_(hd_gate, pair_update.unsqueeze(0))
        self._invalidate_preplay_transition_cache()
        return True

    def _oja_update(
        self,
        bvc_plasticity_gate_per_pc: Optional[torch.Tensor] = None,
    ) -> None:
        pc_col_all = self.place_cell_activations.unsqueeze(1)
        if bvc_plasticity_gate_per_pc is None:
            bvc_plasticity_gate_per_pc = torch.ones_like(self.place_cell_activations)
        grid_mix = self.grid_influence_per_pc
        bvc_update_norms = torch.zeros(self.num_scales, dtype=self.dtype, device=self.device)
        gc_update_norms = torch.zeros(self.num_scales, dtype=self.dtype, device=self.device)
        for scale_idx in range(self.num_scales):
            pc_start, pc_end = self.scale_boundaries[scale_idx:scale_idx + 2]
            bvc_start, bvc_end = self.bvc_boundaries[scale_idx:scale_idx + 2]
            grid_start, grid_end = self.grid_boundaries[scale_idx:scale_idx + 2]

            pc_col = pc_col_all[pc_start:pc_end]
            bvc_row = self.bvc_activations[bvc_start:bvc_end].unsqueeze(0)
            w_block = self.w_in[pc_start:pc_end, bvc_start:bvc_end]
            alpha_block = torch.clamp(self.alpha_pb_per_pc[pc_start:pc_end].unsqueeze(1), min=1e-6)
            eta_block = torch.clamp(self.eta_ojas_per_pc[pc_start:pc_end].unsqueeze(1), min=0.0)
            plasticity_gate = torch.clamp(
                torch.nan_to_num(bvc_plasticity_gate_per_pc[pc_start:pc_end]).unsqueeze(1),
                min=0.0,
                max=1.0,
            )
            pc_sq = pc_col.square()
            hebbian_bvc = pc_col @ bvc_row
            decay_bvc = pc_sq * w_block / alpha_block
            # Apply the Gaussian/plasticity gate only to the BVC Hebbian drive.
            hebbian_bvc.mul_(plasticity_gate)
            delta_bvc = hebbian_bvc - decay_bvc
            delta_bvc.mul_(eta_block)
            delta_bvc.mul_(self.tau)
            bvc_mix = (1.0 - grid_mix[pc_start:pc_end]).unsqueeze(1)
            delta_bvc.mul_(bvc_mix)
            bvc_update_norms[scale_idx] = torch.linalg.vector_norm(delta_bvc.reshape(-1), ord=2)
            w_block.add_(delta_bvc)
            if bool(getattr(self, "clamp_afferent_weights_nonnegative", False)):
                w_block.clamp_(min=0.0)

            if grid_end > grid_start:
                grid_row = self.grid_cell_activations_effective[grid_start:grid_end].unsqueeze(0)
                wg_block = self.w_grid[pc_start:pc_end, grid_start:grid_end]
                alpha_grid = torch.clamp(self.alpha_pg_per_pc[pc_start:pc_end].unsqueeze(1), min=1e-6)
                eta_grid = torch.clamp(self.eta_ojas_gc_per_pc[pc_start:pc_end].unsqueeze(1), min=0.0)
                delta_grid = pc_col @ grid_row
                delta_grid.sub_(pc_sq * wg_block / alpha_grid)
                delta_grid.mul_(eta_grid)
                delta_grid.mul_(self.tau)
                grid_mix_col = grid_mix[pc_start:pc_end].unsqueeze(1)
                delta_grid.mul_(grid_mix_col)
                gc_update_norms[scale_idx] = torch.linalg.vector_norm(delta_grid.reshape(-1), ord=2)
                wg_block.add_(delta_grid)
                if bool(getattr(self, "clamp_afferent_weights_nonnegative", False)):
                    wg_block.clamp_(min=0.0)
                if bool(getattr(self, "enforce_grid_structural_mask", True)):
                    initial_w_grid = getattr(self, "initial_w_grid", None)
                    if (
                        initial_w_grid is not None
                        and tuple(getattr(initial_w_grid, "shape", ())) == tuple(self.w_grid.shape)
                    ):
                        mask_block = initial_w_grid[pc_start:pc_end, grid_start:grid_end] > 0
                        wg_block.mul_(mask_block.to(dtype=wg_block.dtype, device=wg_block.device))
        self.last_oja_bvc_update_norm_per_scale = bvc_update_norms.detach()
        self.last_oja_gc_update_norm_per_scale = gc_update_norms.detach()

    def apply_connection_decay(self, decay_scale: float = 1.0) -> None:
        decay_scale = min(1.0, max(0.0, float(decay_scale)))
        if decay_scale <= 1e-6:
            return
        decay_factor = getattr(self, "connection_decay_factor", None)
        if decay_factor is None or decay_factor.numel() != self.num_pc_total * self.num_pc_total:
            decay_rates = getattr(self, "connection_decay_rate_per_scale", None)
            if decay_rates is None:
                decay_rates = getattr(self, "connection_decay_rate", 1e-5)
            self.configure_connection_decay_rates(decay_rates)
            decay_factor = self.connection_decay_factor
        if decay_factor.numel() != self.num_pc_total * self.num_pc_total:
            return
        if decay_scale >= 1.0 - 1e-6:
            self.w_rec_unified.mul_(decay_factor)
        else:
            scaled_decay_factor = 1.0 - (decay_scale * (1.0 - decay_factor))
            self.w_rec_unified.mul_(scaled_decay_factor)
        self._invalidate_preplay_transition_cache()

    def _get_preplay_recurrent_weights(self, direction: int) -> torch.Tensor:
        return self.w_rec_unified[int(direction) % self.n_hd]

    def _compute_preplay_transitions_v2(self) -> torch.Tensor:
        cached = getattr(self, "_preplay_transitions_cache", None)
        if bool(getattr(self, "cache_preplay_transitions", False)) and cached is not None:
            return cached

        recurrent_weights = torch.clamp(torch.nan_to_num(self.w_rec_unified), min=0.0)
        if bool(getattr(self, "preplay_normalize_transitions", True)):
            column_sums = torch.sum(recurrent_weights, dim=1, keepdim=True)
            transitions = recurrent_weights / torch.clamp(column_sums, min=1e-12)
        else:
            transitions = recurrent_weights
        if bool(getattr(self, "cache_preplay_transitions", False)):
            self._preplay_transitions_cache = transitions
        return transitions

    def _one_step_preplay_v2(
        self,
        states: torch.Tensor,
        dirs: torch.Tensor,
        transitions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        states_t = self._coerce_tensor(states, dtype=self.dtype)
        if states_t.dim() != 2:
            raise ValueError(
                f"states must have shape [batch, num_pc], got {tuple(states_t.shape)}"
            )
        if states_t.shape[1] != self.num_pc_total:
            raise ValueError(
                f"State width mismatch: got {states_t.shape[1]}, expected {self.num_pc_total}"
            )

        dirs_t = self._coerce_tensor(dirs, dtype=torch.long).view(-1) % self.n_hd
        batch_size = int(states_t.shape[0])
        if int(dirs_t.numel()) == 1 and batch_size > 1:
            dirs_t = dirs_t.expand(batch_size)
        elif int(dirs_t.numel()) != batch_size:
            raise ValueError(
                f"dirs must contain 1 or {batch_size} entries, got {int(dirs_t.numel())}"
            )

        transitions_t = (
            self._compute_preplay_transitions_v2()
            if transitions is None
            else self._coerce_tensor(transitions, dtype=self.dtype)
        )
        recurrent = torch.zeros_like(states_t)
        for direction_idx in range(self.n_hd):
            mask = dirs_t == direction_idx
            recurrent[mask] = states_t[mask] @ transitions_t[direction_idx].transpose(0, 1)
        return torch.tanh(torch.relu(recurrent))

    def preplay_from_state_batched(
        self,
        starting_activations: torch.Tensor,
        directions: torch.Tensor,
        num_steps: int = 1,
    ) -> torch.Tensor:
        states = self._coerce_tensor(starting_activations, dtype=self.dtype)
        if states.dim() == 1:
            states = states.unsqueeze(0)
        dirs = self._coerce_tensor(directions, dtype=torch.long).view(-1)
        transitions = self._compute_preplay_transitions_v2()
        for _ in range(max(1, int(num_steps))):
            states = self._one_step_preplay_v2(states, dirs, transitions=transitions)
        return states

    def _normalize_preplay_state_by_scale(self, states: torch.Tensor) -> torch.Tensor:
        """Normalize imagined PC activity independently per scale before scoring."""
        states_t = self._coerce_tensor(states, dtype=self.dtype)
        squeeze_output = False
        if states_t.dim() == 1:
            states_t = states_t.unsqueeze(0)
            squeeze_output = True
        if states_t.dim() != 2:
            raise ValueError(
                f"states must have shape [batch, num_pc], got {tuple(states_t.shape)}"
            )
        if states_t.shape[1] != self.num_pc_total:
            raise ValueError(
                f"State width mismatch: got {states_t.shape[1]}, expected {self.num_pc_total}"
            )

        normalized = torch.zeros_like(states_t)
        for scale_idx in range(self.num_scales):
            start, end = self.scale_boundaries[scale_idx:scale_idx + 2]
            block = torch.clamp(torch.nan_to_num(states_t[:, start:end]), min=0.0)
            mass = torch.sum(block, dim=1, keepdim=True)
            normalized[:, start:end] = torch.where(
                mass > 1e-6,
                block / torch.clamp(mass, min=1e-6),
                block,
            )

        return normalized.squeeze(0) if squeeze_output else normalized

    def _preplay_scale_masses(self, states: torch.Tensor) -> torch.Tensor:
        """Return raw nonnegative PC activity mass per scale for each imagined state."""
        states_t = self._coerce_tensor(states, dtype=self.dtype)
        if states_t.dim() == 1:
            states_t = states_t.unsqueeze(0)
        masses = torch.zeros(
            (states_t.shape[0], self.num_scales),
            dtype=self.dtype,
            device=self.device,
        )
        for scale_idx in range(self.num_scales):
            start, end = self.scale_boundaries[scale_idx:scale_idx + 2]
            block = torch.clamp(torch.nan_to_num(states_t[:, start:end]), min=0.0)
            masses[:, scale_idx] = torch.sum(block, dim=1)
        return masses

    def _preplay_reward_diagnostics(self, unified_rcn, states: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Score imagined states with and without per-scale normalization for debug logs."""
        reward_fn = getattr(unified_rcn, "compute_reward_activations_batched", None)
        if not callable(reward_fn):
            raise AttributeError(
                "unified_rcn must provide compute_reward_activations_batched(states)"
            )

        states_t = self._coerce_tensor(states, dtype=self.dtype)
        if states_t.dim() == 1:
            states_t = states_t.unsqueeze(0)
        normalized_states = self._normalize_preplay_state_by_scale(states_t)

        raw_scores = self._coerce_tensor(reward_fn(states_t), dtype=self.dtype).view(-1)
        normalized_scores = self._coerce_tensor(
            reward_fn(normalized_states),
            dtype=self.dtype,
        ).view(-1)

        scale_scores = torch.zeros(
            (states_t.shape[0], self.num_scales),
            dtype=self.dtype,
            device=self.device,
        )
        weights = getattr(unified_rcn, "w_in_effective", None)
        if weights is not None:
            weights_t = self._coerce_tensor(weights, dtype=self.dtype).view(-1)
            if int(weights_t.numel()) == self.num_pc_total:
                for scale_idx in range(self.num_scales):
                    start, end = self.scale_boundaries[scale_idx:scale_idx + 2]
                    scale_scores[:, scale_idx] = torch.sum(
                        normalized_states[:, start:end] * weights_t[start:end],
                        dim=1,
                    )
            else:
                weights = None

        if weights is None:
            for scale_idx in range(self.num_scales):
                start, end = self.scale_boundaries[scale_idx:scale_idx + 2]
                masked = torch.zeros_like(normalized_states)
                masked[:, start:end] = normalized_states[:, start:end]
                scale_scores[:, scale_idx] = self._coerce_tensor(
                    reward_fn(masked),
                    dtype=self.dtype,
                ).view(-1)

        return {
            "raw_scores": torch.clamp(
                torch.nan_to_num(raw_scores, nan=0.0, posinf=0.0, neginf=0.0),
                min=0.0,
            ),
            "normalized_scores": torch.clamp(
                torch.nan_to_num(normalized_scores, nan=0.0, posinf=0.0, neginf=0.0),
                min=0.0,
            ),
            "scale_scores": torch.clamp(
                torch.nan_to_num(scale_scores, nan=0.0, posinf=0.0, neginf=0.0),
                min=0.0,
            ),
            "scale_masses": self._preplay_scale_masses(states_t),
        }

    def unified_preplay_sampling(
        self,
        unified_rcn,
        n_hd: int = 8,
        num_steps: int = 3,
        discount_factor: float = 0.9,
        within_direction_beta: float = 2.0,
        num_samples: int = 10,
        sampling_temperature: float = 1.0,
        turn_offsets: Optional[Sequence[int]] = None,
        return_diagnostics: bool = False,
        trajectory_blocked_fn: Optional[Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]] = None,
        trajectory_step_distance: Optional[float] = None,
        blocked_return_penalty: float = -1.0,
    ) -> tuple:
        """Stochastic delta-preplay with a fixed first step along the macro direction."""
        n_hd = int(n_hd)
        num_steps = int(max(1, num_steps))
        num_samples = int(max(1, num_samples))
        turn_values = list(turn_offsets) if turn_offsets is not None else [-1, 0, 1]
        if len(turn_values) == 0:
            raise ValueError("turn_offsets must contain at least one candidate turn")
        turn_options = torch.as_tensor(turn_values, dtype=torch.long, device=self.device).view(-1)
        num_turn_options = int(turn_options.numel())
        discount = discount_factor ** torch.arange(num_steps, dtype=self.dtype, device=self.device)
        reward_fn = getattr(unified_rcn, "compute_reward_activations_batched", None)
        if not callable(reward_fn):
            raise AttributeError(
                "unified_rcn must provide compute_reward_activations_batched(states)"
            )

        def _score_states(batch_states: torch.Tensor) -> torch.Tensor:
            # Score a sanitized view; keep rollout dynamics raw.
            states_t = self._coerce_tensor(batch_states, dtype=self.dtype)
            if states_t.dim() == 1:
                states_t = states_t.unsqueeze(0)
            states_t = torch.clamp(torch.nan_to_num(states_t), min=0.0)
            if bool(getattr(self, "preplay_global_score_normalization", True)):
                mass = torch.sum(states_t, dim=1, keepdim=True)
                scoring_states = states_t / torch.clamp(mass, min=1e-12)
                try:
                    raw_scores = reward_fn(scoring_states, normalize=True)
                except TypeError:
                    raw_scores = reward_fn(scoring_states)
            else:
                scoring_states = states_t
                try:
                    raw_scores = reward_fn(scoring_states, normalize=False)
                except TypeError:
                    raw_scores = reward_fn(scoring_states)
            scores = self._coerce_tensor(raw_scores, dtype=self.dtype).view(-1)
            if int(scores.numel()) != int(batch_states.shape[0]):
                raise ValueError(
                    f"Reward scorer returned {int(scores.numel())} values for batch size {int(batch_states.shape[0])}"
                )
            return torch.clamp(
                torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0),
                min=0.0,
            )

        batch_size = n_hd * num_samples
        states = self.place_cell_activations.unsqueeze(0).expand(batch_size, -1).clone()
        dirs = torch.arange(n_hd, device=self.device).repeat_interleave(num_samples)
        returns = torch.zeros(batch_size, dtype=self.dtype, device=self.device)
        vectors = torch.zeros((batch_size, 2), dtype=self.dtype, device=self.device)
        temperature = max(1e-6, float(sampling_temperature))
        transitions = self._compute_preplay_transitions_v2()
        current_reward = _score_states(self.place_cell_activations.unsqueeze(0)).view(1)[0]
        trajectory_step = (
            float(trajectory_step_distance)
            if trajectory_step_distance is not None
            else 0.0
        )
        use_trajectory_blocking = callable(trajectory_blocked_fn) and trajectory_step > 0.0
        blocked_penalty = float(blocked_return_penalty)
        sim_positions = torch.zeros((batch_size, 2), dtype=self.dtype, device=self.device)
        trajectory_active = torch.ones(batch_size, dtype=torch.bool, device=self.device)
        trajectory_blocked_counts = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        trajectory_dirs = torch.full(
            (batch_size, num_steps),
            -1,
            dtype=torch.long,
            device=self.device,
        )

        def _check_trajectory_blocked(
            positions: torch.Tensor,
            direction_ids: torch.Tensor,
            step_idx: int,
        ) -> torch.Tensor:
            if not use_trajectory_blocking:
                return torch.zeros(
                    int(direction_ids.numel()),
                    dtype=torch.bool,
                    device=self.device,
                )
            blocked = trajectory_blocked_fn(positions, direction_ids, int(step_idx))
            blocked_t = torch.as_tensor(blocked, dtype=torch.bool, device=self.device).view(-1)
            if int(blocked_t.numel()) != int(direction_ids.numel()):
                raise ValueError(
                    "trajectory_blocked_fn returned "
                    f"{int(blocked_t.numel())} values for {int(direction_ids.numel())} directions"
                )
            return blocked_t

        def _advance_sim_positions(
            positions: torch.Tensor,
            direction_ids: torch.Tensor,
            active_mask: torch.Tensor,
        ) -> torch.Tensor:
            if not use_trajectory_blocking:
                return positions
            angles = direction_ids.to(dtype=self.dtype) * (2.0 * math.pi / float(n_hd))
            deltas = trajectory_step * torch.stack(
                [torch.cos(angles), torch.sin(angles)],
                dim=1,
            )
            return torch.where(active_mask.view(-1, 1), positions + deltas, positions)

        collect_diagnostics = bool(return_diagnostics)
        if collect_diagnostics:
            raw_returns = torch.zeros(batch_size, dtype=self.dtype, device=self.device)
            scale_returns = torch.zeros(
                (batch_size, self.num_scales),
                dtype=self.dtype,
                device=self.device,
            )
            current_mass_by_scale = self._preplay_scale_masses(self.place_cell_activations).squeeze(0)
            first_raw_scores_by_dir = None
            first_normalized_scores_by_dir = None
            first_planner_scores_by_dir = None
            first_scale_scores_by_dir = None
            first_predicted_mass_by_scale = None

        first_blocked = _check_trajectory_blocked(sim_positions, dirs, 0)
        trajectory_active = torch.logical_and(trajectory_active, torch.logical_not(first_blocked))
        trajectory_blocked_counts += first_blocked.to(dtype=torch.long)
        trajectory_dirs[:, 0] = dirs

        states = self._one_step_preplay_v2(states, dirs, transitions=transitions)
        first_rewards = _score_states(states)
        first_advantages = torch.where(
            trajectory_active,
            first_rewards - current_reward,
            torch.full_like(first_rewards, blocked_penalty),
        )
        returns += discount[0] * first_advantages
        sim_positions = _advance_sim_positions(sim_positions, dirs, trajectory_active)
        if collect_diagnostics:
            first_diag = self._preplay_reward_diagnostics(unified_rcn, states)
            raw_returns += discount[0] * first_diag["raw_scores"]
            scale_returns += discount[0] * first_diag["scale_scores"]
            first_raw_scores_by_dir = first_diag["raw_scores"].view(n_hd, num_samples)[:, 0]
            first_normalized_scores_by_dir = first_diag["normalized_scores"].view(n_hd, num_samples)[:, 0]
            first_planner_scores_by_dir = first_rewards.view(n_hd, num_samples)[:, 0]
            first_scale_scores_by_dir = first_diag["scale_scores"].view(n_hd, num_samples, self.num_scales)[:, 0, :]
            first_predicted_mass_by_scale = first_diag["scale_masses"].view(n_hd, num_samples, self.num_scales)[:, 0, :]
        first_angles = dirs.to(dtype=self.dtype) * (2.0 * math.pi / float(n_hd))
        vectors += torch.stack([torch.cos(first_angles), torch.sin(first_angles)], dim=1)

        for step_idx in range(1, num_steps):
            candidate_dirs = ((dirs.unsqueeze(1) + turn_options.unsqueeze(0)) % n_hd).reshape(-1)
            candidate_states_in = states.repeat_interleave(num_turn_options, dim=0)
            candidate_positions = sim_positions.repeat_interleave(num_turn_options, dim=0)
            candidate_states = self._one_step_preplay_v2(
                candidate_states_in,
                candidate_dirs,
                transitions=transitions,
            )
            candidate_rewards = _score_states(candidate_states).view(batch_size, num_turn_options)
            candidate_advantages = candidate_rewards - current_reward
            if use_trajectory_blocking:
                candidate_blocked = _check_trajectory_blocked(
                    candidate_positions,
                    candidate_dirs,
                    step_idx,
                ).view(batch_size, num_turn_options)
                active_options = trajectory_active.view(-1, 1).expand(-1, num_turn_options)
                candidate_valid = torch.logical_and(active_options, torch.logical_not(candidate_blocked))
                has_valid = torch.any(candidate_valid, dim=1)
                selection_logits = candidate_advantages / temperature
                selection_logits = selection_logits.masked_fill(~candidate_valid, -1.0e9)
                selection_logits = torch.where(
                    has_valid.view(-1, 1),
                    selection_logits,
                    torch.zeros_like(selection_logits),
                )
            else:
                candidate_valid = torch.ones_like(candidate_advantages, dtype=torch.bool)
                has_valid = torch.ones(batch_size, dtype=torch.bool, device=self.device)
                selection_logits = candidate_advantages / temperature
            probs = torch.softmax(selection_logits, dim=1)
            sampled_turn_idx = torch.multinomial(probs, num_samples=1).squeeze(1)
            turns = turn_options.index_select(0, sampled_turn_idx)
            selected_flat = (
                torch.arange(batch_size, dtype=torch.long, device=self.device) * num_turn_options
                + sampled_turn_idx
            )
            if collect_diagnostics:
                candidate_diag = self._preplay_reward_diagnostics(unified_rcn, candidate_states)
                raw_returns += discount[step_idx] * candidate_diag["raw_scores"].index_select(0, selected_flat)
                scale_returns += discount[step_idx] * candidate_diag["scale_scores"].index_select(0, selected_flat)
            dirs = (dirs + turns) % n_hd
            states = candidate_states.index_select(0, selected_flat)
            selected_valid = torch.logical_and(
                has_valid,
                candidate_valid.gather(1, sampled_turn_idx.view(-1, 1)).squeeze(1),
            )
            advantages = candidate_advantages.gather(1, sampled_turn_idx.view(-1, 1)).squeeze(1)
            advantages = torch.where(
                selected_valid,
                advantages,
                torch.full_like(advantages, blocked_penalty),
            )
            returns += discount[step_idx] * advantages
            trajectory_active = selected_valid
            trajectory_blocked_counts += torch.logical_not(selected_valid).to(dtype=torch.long)
            trajectory_dirs[:, step_idx] = dirs
            sim_positions = _advance_sim_positions(sim_positions, dirs, trajectory_active)
            angles = dirs.to(dtype=self.dtype) * (2.0 * math.pi / float(n_hd))
            vectors += torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)

        returns_by_dir = returns.view(n_hd, num_samples)
        vectors_by_dir = vectors.view(n_hd, num_samples, 2)
        best_sample_idx = torch.argmax(returns_by_dir, dim=1)
        macro_returns = returns_by_dir.gather(1, best_sample_idx.unsqueeze(1)).squeeze(1)
        macro_vectors = vectors_by_dir[
            torch.arange(n_hd, dtype=torch.long, device=self.device),
            best_sample_idx,
        ]
        if num_samples > 1:
            sampling_variances = torch.var(returns_by_dir, dim=1, unbiased=False)
        else:
            sampling_variances = torch.zeros(n_hd, dtype=self.dtype, device=self.device)
        diagnostics = None
        if collect_diagnostics:
            final_mass_by_scale = self._preplay_scale_masses(states).view(n_hd, num_samples, self.num_scales)
            raw_returns_by_dir = raw_returns.view(n_hd, num_samples)
            scale_returns_by_dir = scale_returns.view(n_hd, num_samples, self.num_scales)
            row_idx = torch.arange(n_hd, dtype=torch.long, device=self.device)
            best_raw_returns = raw_returns_by_dir.gather(1, best_sample_idx.unsqueeze(1)).squeeze(1)
            best_scale_returns = scale_returns_by_dir[row_idx, best_sample_idx]
            best_predicted_mass_by_scale = final_mass_by_scale[row_idx, best_sample_idx]
            best_predicted_states = states.reshape(n_hd, num_samples, self.num_pc_total)[
                row_idx,
                best_sample_idx,
            ]
            best_trajectory_blocked_counts = trajectory_blocked_counts.view(n_hd, num_samples)[
                row_idx,
                best_sample_idx,
            ]
            best_trajectory_dirs = trajectory_dirs.view(n_hd, num_samples, num_steps)[
                row_idx,
                best_sample_idx,
            ]
            scale_names = [
                str(cfg.get("name", f"scale_{scale_idx}"))
                for scale_idx, cfg in enumerate(self.scale_configs)
            ]
            diagnostics = {
                "scale_names": scale_names,
                "num_steps": int(num_steps),
                "num_samples": int(num_samples),
                "discount_factor": float(discount_factor),
                "return_mode": "sum",
                "planner_scoring_mode": (
                    (
                        "global_norm_rcn_reward_advantage"
                        if bool(getattr(self, "preplay_global_score_normalization", True))
                        else "raw_rcn_reward_advantage"
                    )
                    + (
                        "+normalized_transitions"
                        if bool(getattr(self, "preplay_normalize_transitions", True))
                        else "+raw_transitions"
                    )
                    + ("+executable_microtrajectories" if use_trajectory_blocking else "")
                    + "+sum_return"
                ),
                "current_reward": float(current_reward.detach().cpu().item()),
                "current_mass_by_scale": current_mass_by_scale.detach().cpu(),
                "first_raw_scores": first_raw_scores_by_dir.detach().cpu(),
                "first_normalized_scores": first_normalized_scores_by_dir.detach().cpu(),
                "first_planner_scores": first_planner_scores_by_dir.detach().cpu(),
                "first_scale_scores": first_scale_scores_by_dir.detach().cpu(),
                "first_predicted_mass_by_scale": first_predicted_mass_by_scale.detach().cpu(),
                "raw_returns": best_raw_returns.detach().cpu(),
                "planner_returns": macro_returns.detach().cpu(),
                "scale_returns": best_scale_returns.detach().cpu(),
                "predicted_mass_by_scale": best_predicted_mass_by_scale.detach().cpu(),
                "predicted_states": best_predicted_states.detach().cpu(),
                "best_sample_idx": best_sample_idx.detach().cpu(),
                "trajectory_blocked_counts": best_trajectory_blocked_counts.detach().cpu(),
                "trajectory_dirs": best_trajectory_dirs.detach().cpu(),
                "trajectory_step_distance": float(trajectory_step if use_trajectory_blocking else 0.0),
                "trajectory_blocked_penalty": float(blocked_penalty),
            }
        centered = macro_returns - torch.max(macro_returns)
        probs = torch.softmax(within_direction_beta * centered, dim=0)
        candidate_angles = torch.arange(n_hd, dtype=self.dtype, device=self.device) * (
            2.0 * math.pi / float(n_hd)
        )
        candidate_vectors = torch.stack(
            [torch.cos(candidate_angles), torch.sin(candidate_angles)],
            dim=1,
        )
        combined_vector = torch.sum(probs.unsqueeze(1) * candidate_vectors, dim=0)
        best_dir = torch.argmax(macro_returns)
        if float(torch.norm(combined_vector).item()) < 1e-6:
            angle = best_dir.to(dtype=self.dtype) * (2.0 * math.pi / float(n_hd))
            combined_vector = torch.stack([torch.cos(angle), torch.sin(angle)])
        final_direction = torch.atan2(combined_vector[1], combined_vector[0]) * (180.0 / math.pi)
        final_direction = torch.where(final_direction < 0, final_direction + 360.0, final_direction)
        expected_value = torch.sum(probs * macro_returns)
        if diagnostics is not None:
            diagnostics["action_vector_mode"] = "initial_candidate_hd_vectors"
        if diagnostics is not None:
            return (
                final_direction,
                expected_value,
                combined_vector,
                macro_returns,
                macro_vectors,
                sampling_variances,
                probs,
                diagnostics,
            )
        return final_direction, expected_value, combined_vector, macro_returns, macro_vectors, sampling_variances, probs

    def get_activations_per_scale(self) -> List[torch.Tensor]:
        return [
            self.place_cell_activations[self.scale_boundaries[i]:self.scale_boundaries[i + 1]]
            for i in range(self.num_scales)
        ]

    def _update_grid_diagnostics(
        self,
        raw_bvc_drive: torch.Tensor,
        raw_grid_drive: torch.Tensor,
        afferent_excitation: torch.Tensor,
        grid_mix: torch.Tensor,
    ) -> None:
        if not self.enable_live_diagnostics:
            return
        per_scale = []
        stdp_mass = getattr(self, "last_stdp_input_mass_per_scale", None)
        stdp_active_fraction = getattr(self, "last_stdp_input_active_fraction_per_scale", None)
        for scale_idx in range(self.num_scales):
            start, end = self.scale_boundaries[scale_idx:scale_idx + 2]
            raw_bvc = torch.mean(torch.abs(raw_bvc_drive[start:end]))
            raw_grid = torch.mean(torch.abs(raw_grid_drive[start:end]))
            raw_bvc_block = torch.nan_to_num(raw_bvc_drive[start:end])
            raw_grid_block = torch.nan_to_num(raw_grid_drive[start:end])
            afferent_block = torch.nan_to_num(afferent_excitation[start:end])
            denom = torch.clamp(raw_bvc + raw_grid, min=1e-12)
            per_scale.append(
                {
                    "scale_idx": int(scale_idx),
                    "raw_bvc_abs_mean": float(raw_bvc.item()),
                    "raw_bvc_sum": float(torch.sum(raw_bvc_block).item()),
                    "raw_bvc_max": float(torch.max(raw_bvc_block).item()) if int(raw_bvc_block.numel()) else 0.0,
                    "raw_bvc_std": float(torch.std(raw_bvc_block, unbiased=False).item()) if int(raw_bvc_block.numel()) > 1 else 0.0,
                    "raw_bvc_positive_fraction": float(torch.mean((raw_bvc_block > 0.0).to(dtype=self.dtype)).item()) if int(raw_bvc_block.numel()) else 0.0,
                    "raw_grid_abs_mean": float(raw_grid.item()),
                    "raw_grid_sum": float(torch.sum(raw_grid_block).item()),
                    "raw_grid_max": float(torch.max(raw_grid_block).item()) if int(raw_grid_block.numel()) else 0.0,
                    "raw_grid_std": float(torch.std(raw_grid_block, unbiased=False).item()) if int(raw_grid_block.numel()) > 1 else 0.0,
                    "raw_grid_positive_fraction": float(torch.mean((raw_grid_block > 0.0).to(dtype=self.dtype)).item()) if int(raw_grid_block.numel()) else 0.0,
                    "raw_grid_share": float((raw_grid / denom).item()),
                    "effective_grid_influence": float(torch.mean(grid_mix[start:end]).item()),
                    "bvc_context_modulation_mode": str(
                        getattr(self, "bvc_context_modulation_mode", "bvc_context")
                    ),
                    "grid_inhibition_mode": str(getattr(self, "grid_inhibition_mode", "sum")),
                    "bvc_excitation_modulation": float(
                        torch.mean(self.last_bvc_excitation_modulation_per_scale[scale_idx]).item()
                        if int(self.last_bvc_excitation_modulation_per_scale.numel()) > scale_idx
                        else 1.0
                    ),
                    "bvc_plasticity_gate": float(
                        torch.mean(self.last_bvc_plasticity_gate_per_scale[scale_idx]).item()
                        if int(getattr(self, "last_bvc_plasticity_gate_per_scale", torch.empty(0)).numel()) > scale_idx
                        else 1.0
                    ),
                    "stdp_input_mass": float(
                        stdp_mass[scale_idx].item()
                        if stdp_mass is not None and int(stdp_mass.numel()) > scale_idx
                        else 0.0
                    ),
                    "stdp_input_active_fraction": float(
                        stdp_active_fraction[scale_idx].item()
                        if stdp_active_fraction is not None and int(stdp_active_fraction.numel()) > scale_idx
                        else 0.0
                    ),
                    "afferent_mean": float(torch.mean(afferent_block).item()) if int(afferent_block.numel()) else 0.0,
                    "afferent_sum": float(torch.sum(afferent_block).item()),
                    "afferent_max": float(torch.max(afferent_block).item()) if int(afferent_block.numel()) else 0.0,
                    "afferent_std": float(torch.std(afferent_block, unbiased=False).item()) if int(afferent_block.numel()) > 1 else 0.0,
                    "afferent_positive_fraction": float(torch.mean((afferent_block > 0.0).to(dtype=self.dtype)).item()) if int(afferent_block.numel()) else 0.0,
                }
            )
        self.last_grid_diagnostics = {"per_scale": per_scale}
