import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch


DEFAULT_REPLAY_TIMESTEPS = 40
DEFAULT_REPLAY_TAU = 5.0
SEED_PRESENCE_THRESHOLD = 0.05
SEED_CONSISTENCY_POWER = 0.5
SEED_RETENTION_ALPHA = 0.15


class UnifiedRewardCell:
    """Single unified reward readout with additive replay propagation."""

    def __init__(
        self,
        num_place_cells: int,
        scale_configs: Optional[Sequence[Dict]] = None,
        replay_timesteps: Optional[int] = None,
        replay_tau: float = DEFAULT_REPLAY_TAU,
        device: Optional[torch.device] = None,
    ) -> None:
        self.device = device or torch.device("cpu")
        self.num_place_cells = int(num_place_cells)
        self.num_place_cells_total = self.num_place_cells
        self.scale_configs = [dict(cfg) for cfg in (scale_configs or [])]
        self.replay_timesteps = int(
            max(1, replay_timesteps if replay_timesteps is not None else DEFAULT_REPLAY_TIMESTEPS)
        )
        self.replay_tau = float(max(1e-6, replay_tau))
        self.seed_presence_threshold = float(max(0.0, SEED_PRESENCE_THRESHOLD))
        self.seed_consistency_power = float(max(0.0, SEED_CONSISTENCY_POWER))
        self.seed_retention_alpha = float(min(1.0, max(0.0, SEED_RETENTION_ALPHA)))
        self.last_seed_debug: Dict = {}
        self.last_replay_debug: Dict = {}

        self._configure_scale_boundaries()
        self.w_in = torch.zeros((1, self.num_place_cells), dtype=torch.float32, device=self.device)
        self.w_in_effective = self.w_in.clone()
        self.reward_cell_activations = torch.zeros((1, 1), dtype=torch.float32, device=self.device)

    def _ensure_runtime_defaults(self) -> None:
        if not hasattr(self, "device"):
            self.device = torch.device("cpu")
        if not hasattr(self, "num_place_cells_total"):
            self.num_place_cells_total = int(self.num_place_cells)
        if not hasattr(self, "replay_timesteps"):
            self.replay_timesteps = DEFAULT_REPLAY_TIMESTEPS
        self.replay_timesteps = int(max(1, self.replay_timesteps))
        if not hasattr(self, "replay_tau"):
            self.replay_tau = DEFAULT_REPLAY_TAU
        self.replay_tau = float(max(1e-6, self.replay_tau))
        if not hasattr(self, "seed_presence_threshold"):
            self.seed_presence_threshold = SEED_PRESENCE_THRESHOLD
        if not hasattr(self, "seed_consistency_power"):
            self.seed_consistency_power = SEED_CONSISTENCY_POWER
        if not hasattr(self, "seed_retention_alpha"):
            self.seed_retention_alpha = SEED_RETENTION_ALPHA
        if not hasattr(self, "last_seed_debug"):
            self.last_seed_debug = {}
        if not hasattr(self, "last_replay_debug"):
            self.last_replay_debug = {}
        self.w_in = torch.as_tensor(self.w_in, dtype=torch.float32, device=self.device).view(1, -1)
        self.w_in_effective = torch.as_tensor(
            getattr(self, "w_in_effective", self.w_in),
            dtype=torch.float32,
            device=self.device,
        ).view(1, -1)
        self.reward_cell_activations = torch.as_tensor(
            getattr(self, "reward_cell_activations", torch.zeros((1, 1))),
            dtype=torch.float32,
            device=self.device,
        ).view(1, 1)

    def _configure_scale_boundaries(self) -> None:
        boundaries = [0]
        for cfg in self.scale_configs:
            boundaries.append(boundaries[-1] + int(cfg.get("num_pc", 0)))
        if len(boundaries) <= 1 or boundaries[-1] != self.num_place_cells:
            boundaries = [0, self.num_place_cells]
        self.scale_boundaries = boundaries
        self.num_scales = len(self.scale_boundaries) - 1

    def reconfigure_from_scale_configs(self, scale_configs: Sequence[Dict]) -> None:
        self.scale_configs = [dict(cfg) for cfg in scale_configs]
        self._configure_scale_boundaries()
        self._ensure_runtime_defaults()

    def compute_reward_activations_batched(self, pc_batch: torch.Tensor) -> torch.Tensor:
        self._ensure_runtime_defaults()
        batch = torch.as_tensor(pc_batch, dtype=torch.float32, device=self.device)
        if batch.dim() == 1:
            batch = batch.unsqueeze(0)
        reward = (batch @ self.w_in_effective.t()).squeeze(1)
        return torch.clamp(torch.nan_to_num(reward), min=0.0, max=1e6)

    def compute_reward_contribution_for_scale_batched(
        self,
        pc_batch: torch.Tensor,
        scale_idx: int,
    ) -> torch.Tensor:
        self._ensure_runtime_defaults()
        batch = torch.as_tensor(pc_batch, dtype=torch.float32, device=self.device)
        if batch.dim() == 1:
            batch = batch.unsqueeze(0)
        scale_idx = int(scale_idx)
        if not (0 <= scale_idx < self.num_scales):
            return torch.zeros(batch.shape[0], dtype=torch.float32, device=self.device)
        start, end = self.scale_boundaries[scale_idx], self.scale_boundaries[scale_idx + 1]
        numerator = (batch[:, start:end] @ self.w_in_effective[:, start:end].t()).squeeze(1)
        return torch.clamp(torch.nan_to_num(numerator), min=0.0, max=1e6)

    def build_goal_reward_from_events(
        self,
        unified_pcn,
        goal_events: Sequence[Sequence[torch.Tensor]],
        replace: bool = True,
    ) -> bool:
        self._ensure_runtime_defaults()
        seed, debug = self._build_goal_contact_seed(goal_events)
        self.last_seed_debug = debug
        if seed is None:
            return False
        self.replay_from_seed(unified_pcn, seed, replace=replace)
        return True

    def _build_goal_contact_seed(
        self,
        goal_events: Sequence[Sequence[torch.Tensor]],
    ) -> Tuple[Optional[torch.Tensor], Dict]:
        event_seeds: List[torch.Tensor] = []
        event_sizes: List[int] = []
        for event in goal_events or []:
            states = []
            for value in event or []:
                state = torch.as_tensor(value, dtype=torch.float32, device=self.device).view(-1)
                if int(state.numel()) == self.num_place_cells:
                    states.append(torch.clamp(state, min=0.0))
            if not states:
                continue
            event_mean = torch.mean(torch.stack(states), dim=0)
            event_seed = self._peak_normalize(event_mean)
            if event_seed is None:
                continue
            event_seeds.append(event_seed)
            event_sizes.append(len(states))

        debug = {
            "goal_seed_mode": "contact_event_consensus",
            "goal_seed_event_count": int(len(event_seeds)),
            "goal_seed_event_sizes": list(event_sizes),
            "goal_seed_consensus_mode": "consistency_weighted_mean",
            "goal_seed_presence_threshold": float(self.seed_presence_threshold),
            "goal_seed_consistency_power": float(self.seed_consistency_power),
        }
        if not event_seeds:
            debug["goal_seed_reason"] = "no_goal_events"
            return None, debug

        stacked = torch.stack(event_seeds)
        event_mean = torch.mean(stacked, dim=0)
        presence = torch.mean(
            (stacked >= float(self.seed_presence_threshold)).to(torch.float32),
            dim=0,
        )
        if self.seed_consistency_power > 0.0:
            consistency_gain = torch.pow(torch.clamp(presence, min=0.0), self.seed_consistency_power)
            seed = event_mean * consistency_gain
        else:
            consistency_gain = torch.ones_like(event_mean)
            seed = event_mean
        seed = self._peak_normalize(seed)
        if seed is None:
            debug["goal_seed_reason"] = "zero_consensus"
            return None, debug

        debug["goal_seed_nonzero_count"] = int(torch.count_nonzero(seed > 1e-8).item())
        debug["goal_seed_mean_consistency"] = float(torch.mean(presence).item())
        debug["goal_seed_mean_consistency_gain"] = float(torch.mean(consistency_gain).item())
        return seed, debug

    def replay_from_seed(
        self,
        unified_pcn,
        seed: torch.Tensor,
        replace: bool = True,
    ) -> torch.Tensor:
        self._ensure_runtime_defaults()
        state = torch.as_tensor(seed, dtype=torch.float32, device=self.device).view(-1)
        if int(state.numel()) != self.num_place_cells:
            raise ValueError(
                f"seed size {int(state.numel())} does not match reward input {self.num_place_cells}"
            )
        state = self._peak_normalize(state)
        if state is None:
            return self.w_in_effective

        transition = self._build_replay_transition(unified_pcn)
        seed_anchor = state.clone()
        weight_update = torch.zeros(self.num_place_cells, dtype=torch.float32, device=self.device)
        steps_completed = 0

        for step in range(int(max(1, self.replay_timesteps))):
            decay = math.exp(-float(step) / float(self.replay_tau))
            wave = torch.clamp(torch.nan_to_num(state), min=0.0)
            peak = torch.max(wave) if int(wave.numel()) > 0 else torch.tensor(0.0, device=self.device)
            if float(peak.item()) > 1e-12:
                weight_update = weight_update + decay * (wave / peak)
            state = torch.tanh(torch.relu(transition @ state.float()))
            steps_completed = step + 1
            if float(torch.max(torch.abs(state)).item()) <= 1e-8:
                break

        if self.seed_retention_alpha > 0.0:
            weight_update = torch.maximum(weight_update, self.seed_retention_alpha * seed_anchor)
        max_val = torch.max(torch.abs(weight_update))
        if torch.isfinite(max_val) and float(max_val.item()) > 1e-12:
            weight_update = weight_update / max_val

        if replace:
            self.w_in = torch.zeros_like(self.w_in)
        self.w_in = self.w_in + torch.nan_to_num(weight_update).view(1, -1)
        self.w_in_effective = torch.clamp(self.w_in.clone(), min=0.0)
        self.reward_cell_activations = torch.zeros_like(self.reward_cell_activations)
        self.last_replay_debug = {
            "replay_mode": "additive_global",
            "replay_steps": int(steps_completed),
            "replay_tau": float(self.replay_tau),
            "transition": "hd_max_recurrent_adjacency(row_norm)",
            "masking": "disabled",
            "seed_retention_alpha": float(self.seed_retention_alpha),
        }
        return self.w_in_effective

    def _build_replay_transition(self, unified_pcn) -> torch.Tensor:
        recurrent_weights = torch.as_tensor(
            unified_pcn.w_rec_unified,
            dtype=torch.float32,
            device=self.device,
        )
        if recurrent_weights.dim() == 3:
            transition = torch.max(recurrent_weights, dim=0).values
        elif recurrent_weights.dim() == 2:
            transition = recurrent_weights
        else:
            raise ValueError("w_rec_unified must be a 2D or 3D tensor")
        transition = torch.clamp(transition, min=0.0).t()
        row_sum = torch.sum(transition, dim=1, keepdim=True)
        transition = transition / torch.clamp(row_sum, min=1e-12)
        return torch.nan_to_num(transition, nan=0.0, posinf=0.0, neginf=0.0)

    def _peak_normalize(self, values: torch.Tensor) -> Optional[torch.Tensor]:
        values = torch.clamp(torch.as_tensor(values, dtype=torch.float32, device=self.device), min=0.0).view(-1)
        peak = float(torch.max(values).item()) if int(values.numel()) > 0 else 0.0
        if peak <= 1e-8:
            return None
        return values / max(peak, 1e-8)
