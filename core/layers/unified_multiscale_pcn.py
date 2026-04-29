import math
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from core.layers.grid_cell_layer_v13 import GridCellLayer
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
        gamma_pb: float = 0.3,
        gamma_pg: float = 0.3,
        gamma_cross: Union[float, Sequence[float]] = 0.35,
        sigma_tune: float = 1.5,
        gate_mode: str = "normal",
        recurrent_inhibition_mode: str = "all_scales",
        learning_stdp_start_steps: int = 8000,
        eta_stdp: Union[float, Sequence[float]] = 0.3,
        tau_hd: float = 0.1,
        enable_connection_decay: bool = True,
        connection_decay_rate: Union[float, Sequence[float]] = 1e-4,
        cross_scale_inhibition_base_enabled: bool = True,
        cross_scale_inhibition_gaussian_enabled: bool = True,
        cross_scale_inhibition_lambda_base: float = 0.40,
        use_bvc_context_modulation: bool = True,
        bvc_context_modulation_mode: str = "pcn_excitation",
        bvc_context_gain_floor: float = 0.10,
        bvc_context_gain_strength: float = 1.0,
        enable_live_diagnostics: bool = False,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.device = device or torch.device("cpu")
        self.dtype = dtype
        self.n_hd = int(n_hd)
        self.n_res = int(n_res)
        self.max_dist = float(max_dist)
        self.world_name = world_name
        self.tau_p = 0.5
        self.tau = float(timestep) / 1000.0

        self.enable_ojas = bool(enable_ojas)
        self.enable_stdp = bool(enable_stdp)
        self.gate_mode = str(gate_mode).strip().lower()
        if self.gate_mode not in {"normal", "no_gate_no_inhibition", "no_gate_with_inhibition"}:
            self.gate_mode = "normal"
        self.recurrent_inhibition_mode = self._normalize_recurrent_inhibition_mode(
            recurrent_inhibition_mode
        )
        self.learning_stdp_start_steps = int(max(0, learning_stdp_start_steps))
        self.learning_step_count = 0

        self.tau_hd = float(max(1e-6, tau_hd))
        self.enable_connection_decay = bool(enable_connection_decay)
        self.cross_scale_inhibition_base_enabled = bool(cross_scale_inhibition_base_enabled)
        self.cross_scale_inhibition_gaussian_enabled = bool(cross_scale_inhibition_gaussian_enabled)
        self.cross_scale_inhibition_lambda_base = float(
            min(1.0, max(0.0, cross_scale_inhibition_lambda_base))
        )

        self.use_bvc_context_modulation = bool(use_bvc_context_modulation)
        self.bvc_context_modulation_mode = self._normalize_bvc_context_modulation_mode(
            bvc_context_modulation_mode
        )
        self.bvc_context_gain_floor = float(min(0.95, max(0.0, bvc_context_gain_floor)))
        self.bvc_context_gain_strength = float(max(0.0, bvc_context_gain_strength))
        self.enable_live_diagnostics = bool(enable_live_diagnostics)

        self.scale_configs = [dict(cfg) for cfg in scale_configs]
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
        self._build_d_opt_per_pc()
        self.sigma_tune_per_pc = self.expand_scale_values_to_pc(self.sigma_tune_per_scale)

        largest_cfg = self.scale_configs[-1] if self.scale_configs else {}
        self.large_scale_one_sided = bool(largest_cfg.get("large_scale_one_sided", False))
        self.large_scale_plateau = float(min(1.0, max(0.0, largest_cfg.get("large_scale_plateau", 1.0))))
        self.large_scale_plateau_onset_sigma = float(
            max(0.0, largest_cfg.get("large_scale_plateau_onset_sigma", 1.0))
        )
        self.large_scale_plateau_full_sigma = float(
            max(
                self.large_scale_plateau_onset_sigma,
                largest_cfg.get("large_scale_plateau_full_sigma", 2.0),
            )
        )

        self.gamma_pp_per_pc = self._expand_cfg_value("gamma_pp", gamma_pp)
        self.gamma_pb_per_pc = self._expand_cfg_value("gamma_pb", gamma_pb)
        self.gamma_pg_per_pc = self._expand_cfg_value("gamma_pg", gamma_pg)
        self.grid_influence_per_pc = self._expand_cfg_value("grid_influence", grid_influence)
        self.alpha_pb_per_pc = self._expand_cfg_value("alpha_pb", math.sqrt(0.5))
        self.alpha_pg_per_pc = self._expand_cfg_value("alpha_pg", math.sqrt(0.5))
        self.gc_plasticity_gate_floor_per_pc = self._expand_cfg_value("gc_plasticity_gate_floor", 0.0)

        self.gamma_pp = float(torch.mean(self.gamma_pp_per_pc).item()) if self.num_pc_total else float(gamma_pp)
        self.gamma_pb = float(torch.mean(self.gamma_pb_per_pc).item()) if self.num_pc_total else float(gamma_pb)
        self.gamma_pg = float(torch.mean(self.gamma_pg_per_pc).item()) if self.num_pc_total else float(gamma_pg)
        self.grid_influence = float(torch.mean(self.grid_influence_per_pc).item()) if self.num_pc_total else float(grid_influence)
        self.alpha_pb = float(torch.mean(self.alpha_pb_per_pc).item()) if self.num_pc_total else math.sqrt(0.5)
        self.alpha_pg = float(torch.mean(self.alpha_pg_per_pc).item()) if self.num_pc_total else math.sqrt(0.5)

        if isinstance(gamma_cross, (list, tuple, np.ndarray, torch.Tensor)):
            gamma_cross_values = [float(v) for v in gamma_cross]
        else:
            gamma_cross_values = [
                float(cfg.get("gamma_cross", gamma_cross)) for cfg in self.scale_configs
            ]
        if len(gamma_cross_values) != self.num_scales:
            gamma_cross_values = [float(gamma_cross)] * self.num_scales
        self.gamma_cross_per_scale = torch.tensor(
            gamma_cross_values,
            dtype=self.dtype,
            device=self.device,
        )
        self.gamma_cross = float(torch.mean(self.gamma_cross_per_scale).item()) if self.num_scales else float(gamma_cross)

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

        rng = np.random.default_rng()
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
        self.w_grid_block_mask = torch.zeros_like(self.w_grid, dtype=torch.bool)
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
            self.w_grid_block_mask[pc_start:pc_end, grid_start:grid_end] = True
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
        self.recurrent_visibility_mask = None
        self.within_scale_block_mask = None

        self.place_cell_trace = torch.zeros(self.num_pc_total, dtype=self.dtype, device=self.device)
        self.hd_cell_trace = torch.zeros((self.n_hd, 1, 1), dtype=self.dtype, device=self.device)
        self.configure_eta_stdp(eta_stdp)
        self.configure_connection_decay_rates(connection_decay_rate)
        self._configure_grid_balance_from_scale_configs()

        self.last_scale_activations: List[torch.Tensor] = []
        self.last_grid_activations: List[torch.Tensor] = []
        self.last_bvc_activations: List[torch.Tensor] = []
        self.last_grid_diagnostics = None
        self.last_scale_preference = torch.ones(self.num_scales, dtype=self.dtype, device=self.device)
        self.last_scale_preference_per_pc = torch.ones(self.num_pc_total, dtype=self.dtype, device=self.device)
        self.last_bvc_context_gain_per_scale = torch.ones(self.num_scales, dtype=self.dtype, device=self.device)
        self.last_bvc_context_gain_per_pc = torch.ones(self.num_pc_total, dtype=self.dtype, device=self.device)
        self.last_learning_stdp_active = False
        self.last_preplay_action_deg = 0.0
        self.last_preplay_expected_value = 0.0

        print(f"[UnifiedMultiScalePCN] Initialized with {self.num_pc_total} total cells")
        print(f"  Grid cells total: {self.num_grid_total}")
        print(f"  BVC cells total: {self.num_bvc_total}")
        print(f"  Scale boundaries: {self.scale_boundaries}")
        print(f"  Recurrent inhibition mode: {self.recurrent_inhibition_mode}")

    @staticmethod
    def _normalize_bvc_context_modulation_mode(mode: str) -> str:
        mode_key = str(mode or "pcn_excitation").strip().lower()
        aliases = {
            "pcn": "pcn_excitation",
            "pc": "pcn_excitation",
            "afferent": "pcn_excitation",
            "pcn_afferent": "pcn_excitation",
            "pcn_excitation": "pcn_excitation",
            "excitation": "pcn_excitation",
            "excitation_only": "pcn_excitation",
            "bvc": "bvc_activation",
            "bvc_activation": "bvc_activation",
            "bvc_activations": "bvc_activation",
            "effective_bvc": "bvc_activation",
            "plasticity": "bvc_plasticity_gate",
            "plasticity_only": "bvc_plasticity_gate",
            "plasticity_gate": "bvc_plasticity_gate",
            "context_plasticity": "bvc_plasticity_gate",
            "learning_gate": "bvc_plasticity_gate",
            "bvc_plasticity": "bvc_plasticity_gate",
            "bvc_plasticity_gate": "bvc_plasticity_gate",
            "bvc_activation_plasticity": "bvc_activation_plasticity_gate",
            "bvc_activation_plasticity_gate": "bvc_activation_plasticity_gate",
        }
        return aliases.get(mode_key, "pcn_excitation")

    @staticmethod
    def _normalize_recurrent_inhibition_mode(mode: str) -> str:
        mode_key = str(mode or "within_scale").strip().lower()
        aliases = {
            "within_scale": "within_scale",
            "per_scale": "within_scale",
            "same_scale": "within_scale",
            "local": "within_scale",
            "all_scales": "all_scales",
            "global": "all_scales",
            "integrated": "all_scales",
            "integrated_cross_scale": "all_scales",
            "cross_scale_as_recurrent": "all_scales",
        }
        return aliases.get(mode_key, "within_scale")

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
                    module_scale_ratio=float(cfg.get("module_scale_ratio", cfg.get("grid_module_scale_ratio", 1.0))),
                    translation_scale=float(cfg.get("translation_scale", cfg.get("grid_translation_scale", 1.0))),
                    threshold=float(cfg.get("threshold", cfg.get("grid_threshold", 0.7))),
                    threshold_type=str(cfg.get("threshold_type", cfg.get("grid_threshold_type", "soft"))),
                    world_name=str(cfg.get("world_name", self.world_name)) if cfg.get("world_name", self.world_name) else None,
                    mask_resolution=int(cfg.get("mask_resolution", 128)),
                    smooth_sigma=float(cfg.get("smooth_sigma", 1.5)),
                    activation_cache_size=int(cfg.get("activation_cache_size", 0)),
                    activation_cache_quantization=cfg.get("activation_cache_quantization", None),
                    device=self.device,
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

    def _build_d_opt_per_pc(self) -> None:
        blocks = []
        for scale_idx, cfg in enumerate(self.scale_configs):
            n_pc = int(cfg["num_pc"])
            base = float(cfg.get("d_opt", self.d_opt[scale_idx].item()))
            jitter_std = float(cfg.get("d_opt_jitter_std", 0.0))
            jitter_range = float(cfg.get("d_opt_jitter_range", 0.0))
            jitter_seed = int(cfg.get("d_opt_jitter_seed", 1000 + scale_idx))
            if jitter_std > 0.0:
                gen = torch.Generator(device="cpu")
                gen.manual_seed(jitter_seed)
                values = torch.normal(base, jitter_std, size=(n_pc,), generator=gen)
            else:
                values = torch.full((n_pc,), base, dtype=torch.float32)
            if jitter_range > 0.0:
                values = torch.clamp(values, min=base - jitter_range, max=base + jitter_range)
            blocks.append(values.to(dtype=self.dtype, device=self.device))
        self.d_opt_per_pc = torch.cat(blocks) if blocks else torch.zeros(0, dtype=self.dtype, device=self.device)

    def _initialize_grid_block_weights(
        self,
        rng: np.random.Generator,
        scale_cfg: Dict,
        num_pc: int,
        num_grid_cells: int,
    ) -> torch.Tensor:
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
            block = rng.binomial(1, ratio, size=(num_pc, num_grid_cells))
            return torch.tensor(block, dtype=self.dtype, device=self.device)

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
            base = float(connection_decay_rate if not isinstance(connection_decay_rate, (list, tuple, np.ndarray, torch.Tensor)) else 1e-4)
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
        self.connection_decay_rate = float(np.mean(values)) if values else 0.0

    def _configure_grid_balance_from_scale_configs(self) -> None:
        self.grid_balance_enabled_per_scale = [
            bool(cfg.get("grid_balance_modalities", True)) for cfg in self.scale_configs
        ]
        self.grid_balance_ema_decay_per_scale = torch.tensor(
            [float(min(0.9999, max(0.0, cfg.get("grid_balance_ema", 0.95)))) for cfg in self.scale_configs],
            dtype=self.dtype,
            device=self.device,
        )
        self.grid_balance_min_gain_per_scale = torch.tensor(
            [float(max(1e-3, cfg.get("grid_balance_min_gain", 0.1))) for cfg in self.scale_configs],
            dtype=self.dtype,
            device=self.device,
        )
        self.grid_balance_max_gain_per_scale = torch.tensor(
            [float(max(1.0, cfg.get("grid_balance_max_gain", 8.0))) for cfg in self.scale_configs],
            dtype=self.dtype,
            device=self.device,
        )
        self.grid_balance_bvc_ema = torch.ones(self.num_scales, dtype=self.dtype, device=self.device)
        self.grid_balance_gc_ema = torch.ones(self.num_scales, dtype=self.dtype, device=self.device)
        self.grid_balance_initialized = False
        self.last_grid_gain_per_scale = []

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

    def compute_scale_preference(self, proximity: float) -> torch.Tensor:
        proximity_t = torch.as_tensor(proximity, dtype=self.dtype, device=self.device)
        pref = torch.exp(-((proximity_t - self.d_opt) ** 2) / (2.0 * torch.clamp(self.sigma_tune_per_scale ** 2, min=1e-6)))
        if self.large_scale_one_sided and self.num_scales > 0:
            last_idx = self.num_scales - 1
            pref[last_idx] = self._apply_large_scale_plateau_ramp(
                pref[last_idx],
                proximity_t,
                self.d_opt[last_idx],
                self.sigma_tune_per_scale[last_idx],
            )
        return torch.clamp(pref, min=0.0, max=1.0)

    def compute_scale_preference_per_pc(self, proximity: float) -> torch.Tensor:
        proximity_t = torch.as_tensor(proximity, dtype=self.dtype, device=self.device)
        pref = torch.exp(-((proximity_t - self.d_opt_per_pc) ** 2) / (2.0 * torch.clamp(self.sigma_tune_per_pc ** 2, min=1e-6)))
        if self.large_scale_one_sided and self.num_scales > 0:
            start, end = self.scale_boundaries[-2], self.scale_boundaries[-1]
            pref[start:end] = self._apply_large_scale_plateau_ramp(
                pref[start:end],
                proximity_t,
                self.d_opt_per_pc[start:end],
                self.sigma_tune_per_pc[start:end],
            )
        return torch.clamp(pref, min=0.0, max=1.0)

    def _apply_large_scale_plateau_ramp(
        self,
        gaussian_pref: torch.Tensor,
        proximity_t: torch.Tensor,
        d_opt_t: torch.Tensor,
        sigma_tune_t: torch.Tensor,
    ) -> torch.Tensor:
        plateau = torch.as_tensor(self.large_scale_plateau, dtype=self.dtype, device=self.device)
        onset = d_opt_t + self.large_scale_plateau_onset_sigma * sigma_tune_t
        full = d_opt_t + self.large_scale_plateau_full_sigma * sigma_tune_t
        alpha = torch.clamp((proximity_t - onset) / torch.clamp(full - onset, min=1e-6), min=0.0, max=1.0)
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        return (1.0 - alpha) * gaussian_pref + alpha * plateau

    def _compute_bvc_context_gain_per_scale(
        self,
        proximity: float,
        scale_preference: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self.use_bvc_context_modulation:
            return torch.ones(self.num_scales, dtype=self.dtype, device=self.device)
        pref = scale_preference if scale_preference is not None else self.compute_scale_preference(proximity)
        gain = self.bvc_context_gain_floor + ((1.0 - self.bvc_context_gain_floor) * pref)
        gain = 1.0 + self.bvc_context_gain_strength * (gain - 1.0)
        return torch.clamp(gain, min=self.bvc_context_gain_floor, max=1.0)

    def _compute_bvc_context_gain_per_pc(
        self,
        proximity: float,
        gain_per_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        gain = gain_per_scale if gain_per_scale is not None else self._compute_bvc_context_gain_per_scale(proximity)
        return self.expand_scale_values_to_pc(gain)

    def _uses_bvc_activation_context(self) -> bool:
        return self.bvc_context_modulation_mode in {"bvc_activation", "bvc_activation_plasticity_gate"}

    def _uses_bvc_excitation_context(self) -> bool:
        return self.bvc_context_modulation_mode == "pcn_excitation"

    def _uses_bvc_plasticity_gate(self) -> bool:
        return self.bvc_context_modulation_mode in {"bvc_plasticity_gate", "bvc_activation_plasticity_gate"}

    def _is_stdp_learning_active(self, learning_active: bool) -> bool:
        if not learning_active:
            return True
        return int(self.learning_step_count) >= int(self.learning_stdp_start_steps)

    def _effective_grid_influence_per_pc(self) -> torch.Tensor:
        return self.grid_influence_per_pc

    def _balance_bvc_and_grid_drive(
        self,
        bvc_drive: torch.Tensor,
        grid_drive: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        balanced_bvc = bvc_drive.clone()
        balanced_grid = grid_drive.clone()
        bvc_gain_per_pc = torch.ones_like(self.place_cell_activations)
        grid_gain_per_pc = torch.ones_like(self.place_cell_activations)
        gain_logs = []
        initialized = bool(getattr(self, "grid_balance_initialized", False))
        for scale_idx in range(self.num_scales):
            start, end = self.scale_boundaries[scale_idx:scale_idx + 2]
            raw_bvc = torch.mean(torch.abs(bvc_drive[start:end]))
            raw_grid = torch.mean(torch.abs(grid_drive[start:end]))
            if bool(self.grid_balance_enabled_per_scale[scale_idx]):
                if not initialized:
                    self.grid_balance_bvc_ema[scale_idx] = torch.clamp(raw_bvc, min=1e-6)
                    self.grid_balance_gc_ema[scale_idx] = torch.clamp(raw_grid, min=1e-6)
                else:
                    decay = self.grid_balance_ema_decay_per_scale[scale_idx]
                    self.grid_balance_bvc_ema[scale_idx] = decay * self.grid_balance_bvc_ema[scale_idx] + (1.0 - decay) * raw_bvc
                    self.grid_balance_gc_ema[scale_idx] = decay * self.grid_balance_gc_ema[scale_idx] + (1.0 - decay) * raw_grid
                target = torch.sqrt(torch.clamp(self.grid_balance_bvc_ema[scale_idx] * self.grid_balance_gc_ema[scale_idx], min=1e-12))
                bvc_gain = torch.clamp(
                    target / torch.clamp(self.grid_balance_bvc_ema[scale_idx], min=1e-6),
                    min=self.grid_balance_min_gain_per_scale[scale_idx],
                    max=self.grid_balance_max_gain_per_scale[scale_idx],
                )
                grid_gain = torch.clamp(
                    target / torch.clamp(self.grid_balance_gc_ema[scale_idx], min=1e-6),
                    min=self.grid_balance_min_gain_per_scale[scale_idx],
                    max=self.grid_balance_max_gain_per_scale[scale_idx],
                )
            else:
                bvc_gain = torch.tensor(1.0, dtype=self.dtype, device=self.device)
                grid_gain = torch.tensor(1.0, dtype=self.dtype, device=self.device)
            balanced_bvc[start:end] *= bvc_gain
            balanced_grid[start:end] *= grid_gain
            bvc_gain_per_pc[start:end] = bvc_gain
            grid_gain_per_pc[start:end] = grid_gain
            gain_logs.append({"scale_idx": scale_idx, "bvc_gain": float(bvc_gain.item()), "grid_gain": float(grid_gain.item())})
        self.grid_balance_initialized = True
        self.last_grid_gain_per_scale = gain_logs
        return balanced_bvc, balanced_grid, bvc_gain_per_pc, grid_gain_per_pc

    def _grid_activations_for_position(self, position: Optional[Sequence[float]]) -> torch.Tensor:
        if position is None:
            return torch.zeros(self.num_grid_total, dtype=self.dtype, device=self.device)
        blocks = []
        for scale_idx, (cfg, layer) in enumerate(zip(self.scale_configs, self.grid_layers)):
            act = layer.get_grid_cell_activations(
                position,
                use_mask=bool(cfg.get("grid_use_mask", True)),
            ).to(device=self.device, dtype=self.dtype)
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
    ) -> torch.Tensor:
        learning_active = bool(learn) if learn is not None else bool(self.enable_ojas or self.enable_stdp)
        learning_active = learning_active and bool(self.enable_ojas or self.enable_stdp)
        if self.enable_stdp:
            self.apply_connection_decay()

        distances_t = self._coerce_tensor(distances, dtype=self.dtype).flatten()
        if proximity is None:
            proximity = float(torch.clamp(torch.min(distances_t), min=0.0, max=self.max_dist).item())

        self.prev_place_cell_activations = self.place_cell_activations.detach().clone()
        if hd_activations is not None:
            self.update_hd_trace(self._coerce_tensor(hd_activations, dtype=self.dtype))

        scale_preference = self.compute_scale_preference(float(proximity))
        scale_preference_per_pc = self.compute_scale_preference_per_pc(float(proximity))
        self.last_scale_preference = scale_preference.detach()
        self.last_scale_preference_per_pc = scale_preference_per_pc.detach()

        raw_bvc_blocks = [layer.get_bvc_activation(distances_t) for layer in self.bvc_layers]
        self.raw_bvc_activations = torch.cat(raw_bvc_blocks) if raw_bvc_blocks else torch.zeros(0, dtype=self.dtype, device=self.device)
        bvc_gain_scale = self._compute_bvc_context_gain_per_scale(float(proximity), scale_preference=scale_preference)
        if self._uses_bvc_activation_context():
            bvc_blocks = [raw_bvc_blocks[i] * bvc_gain_scale[i] for i in range(self.num_scales)]
            self.bvc_activations = torch.cat(bvc_blocks)
        else:
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
        raw_grid_drive = self.w_grid @ self.grid_cell_activations_effective
        bvc_drive, grid_drive, bvc_gain_per_pc, grid_gain_per_pc = self._balance_bvc_and_grid_drive(raw_bvc_drive, raw_grid_drive)

        grid_mix = self._effective_grid_influence_per_pc()
        bvc_context_gain_per_pc = self._compute_bvc_context_gain_per_pc(float(proximity), gain_per_scale=bvc_gain_scale)
        self.last_bvc_context_gain_per_scale = bvc_gain_scale.detach()
        self.last_bvc_context_gain_per_pc = bvc_context_gain_per_pc.detach()
        if self._uses_bvc_excitation_context():
            bvc_excitation_gain = bvc_context_gain_per_pc
        else:
            bvc_excitation_gain = torch.ones_like(bvc_context_gain_per_pc)
        afferent_excitation = ((1.0 - grid_mix) * bvc_drive * bvc_excitation_gain) + (grid_mix * grid_drive)

        new_activations, new_update = self._apply_competition_stage(
            afferent_excitation=afferent_excitation,
            proximity=float(proximity),
            current_activations=self.place_cell_activations,
            activation_update_in=self.activation_update,
            bvc_gain_per_pc=bvc_gain_per_pc,
            grid_gain_per_pc=grid_gain_per_pc,
        )
        self.place_cell_activations = torch.nan_to_num(new_activations)
        self.activation_update = torch.nan_to_num(new_update)

        has_activity = bool(torch.any(self.place_cell_activations > 0).item())

        stdp_active = bool(self.enable_stdp and learning_active and not collided and has_activity and self._is_stdp_learning_active(learning_active))
        self.last_learning_stdp_active = stdp_active
        if stdp_active:
            self._stdp_update()

        if bool(self.enable_ojas and learning_active and has_activity):
            if self._uses_bvc_plasticity_gate():
                denom = torch.clamp(torch.sum(scale_preference), min=1e-6)
                bvc_plasticity_gate = torch.clamp(scale_preference_per_pc / denom, min=0.0, max=1.0)
            else:
                bvc_plasticity_gate = torch.ones_like(self.place_cell_activations)
            self._oja_update(bvc_plasticity_gate=bvc_plasticity_gate, grid_mix=grid_mix)

        self.last_scale_activations = [a.detach().clone() for a in self.get_activations_per_scale()]
        self.last_grid_activations = [
            self.grid_cell_activations[self.grid_boundaries[i]:self.grid_boundaries[i + 1]].detach().clone()
            for i in range(self.num_scales)
        ]
        self.last_bvc_activations = [
            self.bvc_activations[self.bvc_boundaries[i]:self.bvc_boundaries[i + 1]].detach().clone()
            for i in range(self.num_scales)
        ]
        self._update_grid_diagnostics(raw_bvc_drive, raw_grid_drive, bvc_drive, grid_drive, afferent_excitation, grid_mix)
        return self.place_cell_activations

    def _apply_competition_stage(
        self,
        afferent_excitation: torch.Tensor,
        proximity: float,
        current_activations: torch.Tensor,
        activation_update_in: torch.Tensor,
        bvc_gain_per_pc: torch.Tensor,
        grid_gain_per_pc: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bvc_inhibition = torch.zeros_like(current_activations)
        grid_inhibition = torch.zeros_like(current_activations)
        for scale_idx in range(self.num_scales):
            pc_start, pc_end = self.scale_boundaries[scale_idx:scale_idx + 2]
            bvc_start, bvc_end = self.bvc_boundaries[scale_idx:scale_idx + 2]
            grid_start, grid_end = self.grid_boundaries[scale_idx:scale_idx + 2]
            bvc_sum = torch.sum(self.bvc_activations[bvc_start:bvc_end]) * bvc_gain_per_pc[pc_start]
            grid_sum = torch.sum(self.grid_cell_activations_effective[grid_start:grid_end]) * grid_gain_per_pc[pc_start]
            bvc_inhibition[pc_start:pc_end] = self.gamma_pb_per_pc[pc_start] * bvc_sum
            grid_inhibition[pc_start:pc_end] = self.gamma_pg_per_pc[pc_start] * grid_sum

        grid_mix = self._effective_grid_influence_per_pc()
        afferent_inhibition = ((1.0 - grid_mix) * bvc_inhibition) + (grid_mix * grid_inhibition)

        recurrent_inhibition = torch.zeros_like(current_activations)
        use_all_scale = self.recurrent_inhibition_mode == "all_scales" and self.gate_mode != "no_gate_no_inhibition"
        total_activity = torch.sum(current_activations)
        for scale_idx in range(self.num_scales):
            pc_start, pc_end = self.scale_boundaries[scale_idx:scale_idx + 2]
            activity_sum = total_activity if use_all_scale else torch.sum(current_activations[pc_start:pc_end])
            recurrent_inhibition[pc_start:pc_end] = self.gamma_pp_per_pc[pc_start] * activity_sum

        cross_scale_inhibition = torch.zeros_like(current_activations)
        if not use_all_scale and self.gate_mode != "no_gate_no_inhibition":
            cross_scale_inhibition = self.compute_cross_scale_inhibition(proximity, from_activations=current_activations)

        activation_update = activation_update_in + self.tau_p * (
            -activation_update_in
            + afferent_excitation
            - afferent_inhibition
            - recurrent_inhibition
            - cross_scale_inhibition
        )
        return torch.tanh(torch.relu(activation_update)), activation_update

    def compute_cross_scale_inhibition(
        self,
        proximity: float,
        from_activations: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        activations = from_activations if from_activations is not None else self.place_cell_activations
        total_activity = torch.sum(activations)
        inhibition = torch.zeros_like(activations)
        scale_pref = self.compute_scale_preference(proximity)
        for scale_idx in range(self.num_scales):
            start, end = self.scale_boundaries[scale_idx:scale_idx + 2]
            other_activity = total_activity - torch.sum(activations[start:end])
            factor = torch.tensor(1.0, dtype=self.dtype, device=self.device)
            if self.cross_scale_inhibition_gaussian_enabled:
                factor = 1.0 - scale_pref[scale_idx]
            if self.cross_scale_inhibition_base_enabled:
                factor = self.cross_scale_inhibition_lambda_base + (1.0 - self.cross_scale_inhibition_lambda_base) * factor
            inhibition[start:end] = self.gamma_cross_per_scale[scale_idx] * factor * other_activity
        return inhibition

    def update_hd_trace(self, hd_activations: torch.Tensor) -> None:
        hd = torch.nan_to_num(hd_activations).view(-1)
        if int(hd.numel()) != self.n_hd:
            return
        self.hd_cell_trace += (self.tau / self.tau_hd) * (
            hd.view(self.n_hd, 1, 1) - self.hd_cell_trace
        )

    def _stdp_update(self) -> None:
        self.place_cell_trace += (self.tau / 3.0) * (
            self.place_cell_activations - self.place_cell_trace
        )
        pc_act = self.place_cell_activations
        pc_trace = self.place_cell_trace

        update = self.hd_cell_trace * (torch.outer(pc_act, pc_trace) - torch.outer(pc_trace, pc_act)).unsqueeze(0)
        visibility = self._get_recurrent_visibility_mask()
        if visibility is not None:
            update = update * visibility.unsqueeze(0)

        self.w_rec_unified += (update * self.eta_stdp_pair.unsqueeze(0)).to(self.dtype)
        if visibility is not None:
            self.w_rec_unified *= visibility.unsqueeze(0)

    def _oja_update(self, bvc_plasticity_gate: torch.Tensor, grid_mix: torch.Tensor) -> None:
        pc_col_all = self.place_cell_activations.unsqueeze(1)
        for scale_idx in range(self.num_scales):
            pc_start, pc_end = self.scale_boundaries[scale_idx:scale_idx + 2]
            bvc_start, bvc_end = self.bvc_boundaries[scale_idx:scale_idx + 2]
            grid_start, grid_end = self.grid_boundaries[scale_idx:scale_idx + 2]

            pc_col = pc_col_all[pc_start:pc_end]
            bvc_row = self.bvc_activations[bvc_start:bvc_end].unsqueeze(0)
            w_block = self.w_in[pc_start:pc_end, bvc_start:bvc_end]
            alpha_block = torch.clamp(self.alpha_pb_per_pc[pc_start:pc_end].unsqueeze(1), min=1e-6)
            bvc_gate = bvc_plasticity_gate[pc_start:pc_end].unsqueeze(1)
            bvc_mix = (1.0 - grid_mix[pc_start:pc_end]).unsqueeze(1)
            self.w_in[pc_start:pc_end, bvc_start:bvc_end] = torch.clamp(
                w_block + self.tau * ((bvc_gate * (pc_col @ bvc_row)) - ((pc_col ** 2) * w_block / alpha_block)) * bvc_mix,
                min=0.0,
            )

            if grid_end > grid_start:
                grid_row = self.grid_cell_activations_effective[grid_start:grid_end].unsqueeze(0)
                wg_block = self.w_grid[pc_start:pc_end, grid_start:grid_end]
                alpha_grid = torch.clamp(self.alpha_pg_per_pc[pc_start:pc_end].unsqueeze(1), min=1e-6)
                grid_mix_col = grid_mix[pc_start:pc_end].unsqueeze(1)
                self.w_grid[pc_start:pc_end, grid_start:grid_end] = torch.clamp(
                    wg_block + self.tau * ((pc_col @ grid_row) - ((pc_col ** 2) * wg_block / alpha_grid)) * grid_mix_col,
                    min=0.0,
                )
        self.w_grid *= self.w_grid_block_mask.to(dtype=self.dtype, device=self.device)

    def apply_connection_decay(self) -> None:
        if not self.enable_connection_decay:
            return
        if self.connection_decay_rate_pair.numel() != self.num_pc_total * self.num_pc_total:
            return
        self.w_rec_unified *= (1.0 - self.connection_decay_rate_pair.unsqueeze(0))

    def _get_recurrent_visibility_mask(self) -> Optional[torch.Tensor]:
        mask = getattr(self, "recurrent_visibility_mask", None)
        if mask is None or tuple(mask.shape) != (self.num_pc_total, self.num_pc_total):
            return None
        return mask.to(device=self.device, dtype=self.dtype)

    def _get_preplay_recurrent_weights(self, direction: int) -> torch.Tensor:
        return self.w_rec_unified[int(direction) % self.n_hd]

    def preplay_step(self, activations: torch.Tensor, hd_idx: int) -> torch.Tensor:
        v = self._coerce_tensor(activations, dtype=self.dtype).view(-1)
        return torch.tanh(torch.relu((self._get_preplay_recurrent_weights(int(hd_idx)) @ v) - v))

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
        if int(dirs.numel()) == 1 and states.shape[0] > 1:
            dirs = dirs.expand(states.shape[0])
        for _ in range(max(1, int(num_steps))):
            next_states = torch.zeros_like(states)
            for hd in range(self.n_hd):
                idx = torch.nonzero(dirs == hd, as_tuple=False).flatten()
                if int(idx.numel()) > 0:
                    recurrent = (self._get_preplay_recurrent_weights(hd) @ states[idx].T).T
                    next_states[idx] = torch.tanh(torch.relu(recurrent - states[idx]))
            states = next_states
        return states

    def _compute_scale_mass_batched(self, activations_batch: torch.Tensor) -> torch.Tensor:
        if activations_batch.dim() == 1:
            activations_batch = activations_batch.unsqueeze(0)
        masses = [
            torch.sum(torch.abs(activations_batch[:, self.scale_boundaries[i]:self.scale_boundaries[i + 1]]), dim=1)
            for i in range(self.num_scales)
        ]
        return torch.stack(masses, dim=1) if masses else torch.zeros((activations_batch.shape[0], 0), dtype=self.dtype, device=self.device)

    def get_scale_prior_batched(self, activations_batch: torch.Tensor, preference_mix: float = 0.5) -> torch.Tensor:
        del preference_mix
        masses = self._compute_scale_mass_batched(activations_batch)
        return masses / torch.clamp(torch.sum(masses, dim=1, keepdim=True), min=1e-9)

    def get_expression_gate_per_pc(self, activations: Optional[torch.Tensor] = None, preference_mix: float = 0.5) -> Optional[torch.Tensor]:
        if activations is None:
            activations = self.place_cell_activations
        if int(activations.numel()) != self.num_pc_total:
            return None
        scale_gate = self.get_scale_prior_batched(activations.view(1, -1), preference_mix=preference_mix).squeeze(0)
        return self.expand_scale_values_to_pc(torch.clamp(scale_gate, min=0.0, max=1.0))

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
        sample_aggregation: str = "mean",
    ) -> tuple:
        n_hd = int(n_hd)
        sample_aggregation = str(sample_aggregation).strip().lower()
        if sample_aggregation not in {"mean", "max"}:
            sample_aggregation = "mean"
        turn_options = torch.tensor([-1, 0, 1], dtype=torch.long, device=self.device)
        discount = discount_factor ** torch.arange(num_steps, dtype=self.dtype, device=self.device)
        scale_gate = self.get_scale_prior_batched(self.place_cell_activations.unsqueeze(0), preference_mix=0.0).squeeze(0)
        if not torch.isfinite(scale_gate.sum()) or float(scale_gate.sum().item()) <= 1e-9:
            scale_gate = torch.full((self.num_scales,), 1.0 / max(1, self.num_scales), dtype=self.dtype, device=self.device)
        else:
            scale_gate = scale_gate / torch.clamp(scale_gate.sum(), min=1e-9)

        scale_returns = torch.zeros((self.num_scales, n_hd), dtype=self.dtype, device=self.device)
        scale_vectors = torch.zeros((self.num_scales, n_hd, 2), dtype=self.dtype, device=self.device)
        scale_variances = torch.zeros((self.num_scales, n_hd), dtype=self.dtype, device=self.device)
        for scale_idx in range(self.num_scales):
            batch_size = n_hd * int(num_samples)
            states = self.place_cell_activations.unsqueeze(0).expand(batch_size, -1).clone()
            dirs = torch.arange(n_hd, device=self.device).repeat_interleave(int(num_samples))
            returns = torch.zeros(batch_size, dtype=self.dtype, device=self.device)
            vectors = torch.zeros((batch_size, 2), dtype=self.dtype, device=self.device)
            for step_idx in range(int(num_steps)):
                if step_idx > 0:
                    if sampling_strategy == "learned":
                        logits = []
                        for turn in turn_options:
                            next_dirs = (dirs + turn) % n_hd
                            next_states = self.preplay_from_state_batched(states, next_dirs, num_steps=1)
                            logits.append(unified_rcn.compute_reward_contribution_for_scale_batched(next_states, scale_idx))
                        probs = torch.softmax(torch.stack(logits, dim=1) / max(1e-6, float(sampling_temperature)), dim=1)
                        turns = turn_options[torch.multinomial(probs, num_samples=1).squeeze(1)]
                    else:
                        turns = turn_options[torch.randint(0, 3, (batch_size,), device=self.device)]
                    dirs = (dirs + turns) % n_hd
                states = self.preplay_from_state_batched(states, dirs, num_steps=1)
                rewards = unified_rcn.compute_reward_contribution_for_scale_batched(states, scale_idx)
                returns += discount[step_idx] * torch.nan_to_num(rewards)
                angles = dirs.to(dtype=self.dtype) * (2.0 * math.pi / float(n_hd))
                vectors += torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
            returns_by_dir = returns.view(n_hd, int(num_samples))
            vectors_by_dir = vectors.view(n_hd, int(num_samples), 2)
            if sample_aggregation == "max":
                best_idx = torch.argmax(returns_by_dir, dim=1)
                scale_returns[scale_idx] = torch.max(returns_by_dir, dim=1).values
                scale_vectors[scale_idx] = vectors_by_dir[torch.arange(n_hd, device=self.device), best_idx]
            else:
                scale_returns[scale_idx] = torch.mean(returns_by_dir, dim=1)
                scale_vectors[scale_idx] = torch.mean(vectors_by_dir, dim=1)
            scale_variances[scale_idx] = torch.var(returns_by_dir, dim=1)

        macro_returns = torch.sum(scale_gate.unsqueeze(1) * scale_returns, dim=0)
        macro_vectors = torch.sum(scale_gate.view(-1, 1, 1) * scale_vectors, dim=0)
        sampling_variances = torch.sum(scale_gate.unsqueeze(1) * scale_variances, dim=0)
        centered = macro_returns - torch.max(macro_returns)
        probs = torch.softmax(within_direction_beta * centered, dim=0)
        combined_vector = torch.sum(probs.unsqueeze(1) * macro_vectors, dim=0)
        best_dir = torch.argmax(probs * macro_returns)
        if float(torch.norm(combined_vector).item()) < 1e-6:
            angle = best_dir.to(dtype=self.dtype) * (2.0 * math.pi / float(n_hd))
            combined_vector = torch.stack([torch.cos(angle), torch.sin(angle)])
        final_direction = torch.atan2(combined_vector[1], combined_vector[0]) * (180.0 / math.pi)
        final_direction = torch.where(final_direction < 0, final_direction + 360.0, final_direction)
        expected_value = torch.sum(probs * macro_returns)
        self.last_preplay_commit_dir = int(best_dir.item())
        self.last_preplay_action_deg = float(final_direction.item())
        self.last_preplay_expected_value = float(expected_value.item())
        return final_direction, expected_value, combined_vector, macro_returns, macro_vectors, sampling_variances, probs, scale_gate

    def get_activations_per_scale(self) -> List[torch.Tensor]:
        return [
            self.place_cell_activations[self.scale_boundaries[i]:self.scale_boundaries[i + 1]]
            for i in range(self.num_scales)
        ]

    def _update_grid_diagnostics(
        self,
        raw_bvc_drive: torch.Tensor,
        raw_grid_drive: torch.Tensor,
        balanced_bvc_drive: torch.Tensor,
        balanced_grid_drive: torch.Tensor,
        afferent_excitation: torch.Tensor,
        grid_mix: torch.Tensor,
    ) -> None:
        if not self.enable_live_diagnostics:
            return
        per_scale = []
        for scale_idx in range(self.num_scales):
            start, end = self.scale_boundaries[scale_idx:scale_idx + 2]
            raw_bvc = torch.mean(torch.abs(raw_bvc_drive[start:end]))
            raw_grid = torch.mean(torch.abs(raw_grid_drive[start:end]))
            bal_bvc = torch.mean(torch.abs(balanced_bvc_drive[start:end]))
            bal_grid = torch.mean(torch.abs(balanced_grid_drive[start:end]))
            denom = torch.clamp(raw_bvc + raw_grid, min=1e-12)
            bal_denom = torch.clamp(bal_bvc + bal_grid, min=1e-12)
            per_scale.append(
                {
                    "scale_idx": int(scale_idx),
                    "raw_bvc_abs_mean": float(raw_bvc.item()),
                    "raw_grid_abs_mean": float(raw_grid.item()),
                    "raw_grid_share": float((raw_grid / denom).item()),
                    "balanced_bvc_abs_mean": float(bal_bvc.item()),
                    "balanced_grid_abs_mean": float(bal_grid.item()),
                    "balanced_grid_share": float((bal_grid / bal_denom).item()),
                    "effective_grid_influence": float(torch.mean(grid_mix[start:end]).item()),
                    "afferent_mean": float(torch.mean(afferent_excitation[start:end]).item()),
                }
            )
        self.last_grid_diagnostics = {"per_scale": per_scale}
