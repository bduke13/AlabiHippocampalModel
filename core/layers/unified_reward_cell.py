import math
from typing import Dict, List, Optional, Sequence

import torch


DEFAULT_REPLAY_TIMESTEPS = 12
DEFAULT_REPLAY_TAU = 2.0
DEFAULT_REPLAY_LONG_TAU = 8.0
DEFAULT_REPLAY_LONG_WEIGHT = 0.10


class UnifiedRewardCell:
    """Single unified reward readout with additive neighbor-spread replay."""

    def __init__(
        self,
        num_place_cells: int,
        scale_configs: Optional[Sequence[Dict]] = None,
        replay_timesteps: Optional[int] = None,
        replay_tau: float = DEFAULT_REPLAY_TAU,
        replay_long_tau: float = DEFAULT_REPLAY_LONG_TAU,
        replay_long_weight: float = DEFAULT_REPLAY_LONG_WEIGHT,
        device: Optional[torch.device] = None,
    ) -> None:
        self.device = device or torch.device("cpu")
        self.num_place_cells = int(num_place_cells)
        self.scale_configs = [dict(cfg) for cfg in (scale_configs or [])]
        self.replay_timesteps = int(
            max(1, replay_timesteps if replay_timesteps is not None else DEFAULT_REPLAY_TIMESTEPS)
        )
        self.replay_tau = float(max(1e-6, replay_tau))
        self.replay_long_tau = float(max(1e-6, replay_long_tau))
        self.replay_long_weight = float(max(0.0, replay_long_weight))

        self._configure_scale_boundaries()
        self.w_in = torch.zeros((1, self.num_place_cells), dtype=torch.float32, device=self.device)
        self.w_in_effective = self.w_in.clone()
        self.reward_cell_activations = torch.zeros((1, 1), dtype=torch.float32, device=self.device)

    def _ensure_runtime_defaults(self) -> None:
        if not hasattr(self, "device"):
            self.device = torch.device("cpu")
        if not hasattr(self, "replay_timesteps"):
            self.replay_timesteps = DEFAULT_REPLAY_TIMESTEPS
        self.replay_timesteps = int(max(1, self.replay_timesteps))
        if not hasattr(self, "replay_tau"):
            self.replay_tau = DEFAULT_REPLAY_TAU
        self.replay_tau = float(max(1e-6, self.replay_tau))
        if not hasattr(self, "replay_long_tau"):
            self.replay_long_tau = DEFAULT_REPLAY_LONG_TAU
        self.replay_long_tau = float(max(1e-6, self.replay_long_tau))
        if not hasattr(self, "replay_long_weight"):
            self.replay_long_weight = DEFAULT_REPLAY_LONG_WEIGHT
        self.replay_long_weight = float(max(0.0, self.replay_long_weight))
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

    def compute_reward_activations_batched(
        self,
        pc_batch: torch.Tensor,
        normalize: bool = True,
    ) -> torch.Tensor:
        self._ensure_runtime_defaults()
        batch = torch.as_tensor(pc_batch, dtype=torch.float32, device=self.device)
        if batch.dim() == 1:
            batch = batch.unsqueeze(0)
        weighted = (batch @ self.w_in_effective.t()).squeeze(1)
        if not bool(normalize):
            return torch.clamp(torch.nan_to_num(weighted), min=0.0, max=1e6)
        activity_mass = torch.sum(torch.clamp(batch, min=0.0), dim=1)
        reward = weighted / torch.clamp(activity_mass, min=1e-6)
        return torch.clamp(torch.nan_to_num(reward), min=0.0, max=1e6)

    def reset_runtime_state(self) -> None:
        """Clear transient reward-cell state while preserving learned weights."""
        self._ensure_runtime_defaults()
        self.reward_cell_activations = torch.zeros((1, 1), dtype=torch.float32, device=self.device)

    def build_goal_reward_from_events(
        self,
        unified_pcn,
        goal_events: Sequence[Sequence[torch.Tensor]],
        replace: bool = True,
    ) -> bool:
        self._ensure_runtime_defaults()
        seed = self._build_goal_contact_seed(goal_events)
        if seed is None:
            return False
        self.replay_from_seed(unified_pcn, seed, replace=replace)
        return True

    def _build_goal_contact_seed(
        self,
        goal_events: Sequence[Sequence[torch.Tensor]],
    ) -> Optional[torch.Tensor]:
        event_means: List[torch.Tensor] = []
        for event in goal_events or []:
            states = []
            for value in event or []:
                state = torch.as_tensor(value, dtype=torch.float32, device=self.device).view(-1)
                if int(state.numel()) == self.num_place_cells:
                    states.append(torch.clamp(state, min=0.0))
            if not states:
                continue
            event_mean = torch.mean(torch.stack(states), dim=0)
            if self._peak_normalize(event_mean) is None:
                continue
            event_means.append(event_mean)

        if not event_means:
            return None

        consensus = torch.mean(torch.stack(event_means), dim=0)
        peak = float(torch.max(consensus).item()) if int(consensus.numel()) > 0 else 0.0
        if peak <= 1e-8:
            return None

        return consensus / max(peak, 1e-8)

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

        transitions = self._build_replay_transitions(unified_pcn)
        weight_update = self._replay_spread_profile(transitions, state, self.replay_tau)
        if self.replay_long_weight > 0.0:
            long_update = self._replay_spread_profile(transitions, state, self.replay_long_tau)
            weight_update = weight_update + self.replay_long_weight * long_update

        max_val = torch.max(torch.abs(weight_update))
        if torch.isfinite(max_val) and float(max_val.item()) > 1e-12:
            weight_update = weight_update / max_val

        if replace:
            self.w_in = torch.zeros_like(self.w_in)
        self.w_in = self.w_in + torch.nan_to_num(weight_update).view(1, -1)
        self.w_in_effective = torch.clamp(self.w_in.clone(), min=0.0)
        self.reward_cell_activations = torch.zeros_like(self.reward_cell_activations)
        return self.w_in_effective

    def _replay_spread_profile(self, transitions: torch.Tensor, seed: torch.Tensor, tau: float) -> torch.Tensor:
        state = torch.clamp(torch.nan_to_num(seed), min=0.0).view(-1)
        weight_update = torch.zeros(self.num_place_cells, dtype=torch.float32, device=self.device)
        for step in range(int(max(1, self.replay_timesteps))):
            decay = math.exp(-float(step) / float(tau))
            wave = torch.clamp(torch.nan_to_num(state), min=0.0)
            peak = torch.max(wave) if int(wave.numel()) > 0 else torch.tensor(0.0, device=self.device)
            if float(peak.item()) > 1e-12:
                weight_update = torch.maximum(weight_update, decay * wave)
            state = self._additive_replay_next_state(transitions, state)
            if float(torch.max(torch.abs(state)).item()) <= 1e-8:
                break
        max_val = torch.max(torch.abs(weight_update))
        if torch.isfinite(max_val) and float(max_val.item()) > 1e-12:
            weight_update = weight_update / max_val
        return torch.nan_to_num(weight_update)

    def _build_replay_transitions(self, unified_pcn) -> torch.Tensor:
        recurrent_weights = torch.as_tensor(
            unified_pcn.w_rec_unified,
            dtype=torch.float32,
            device=self.device,
        )
        if recurrent_weights.dim() == 3:
            transitions = recurrent_weights.transpose(1, 2)
        elif recurrent_weights.dim() == 2:
            transitions = recurrent_weights.t().unsqueeze(0)
        else:
            raise ValueError("w_rec_unified must be a 2D or 3D tensor")
        transitions = torch.clamp(transitions, min=0.0)
        row_sum = torch.sum(transitions, dim=2, keepdim=True)
        transitions = transitions / torch.clamp(row_sum, min=1e-12)
        return torch.nan_to_num(transitions, nan=0.0, posinf=0.0, neginf=0.0)

    def _additive_replay_next_state(self, transitions: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        state = torch.clamp(torch.nan_to_num(state), min=0.0).view(-1)
        collapsed = torch.max(transitions, dim=0).values
        next_state = collapsed @ state + state
        return torch.tanh(torch.relu(next_state))

    def _peak_normalize(self, values: torch.Tensor) -> Optional[torch.Tensor]:
        values = torch.clamp(torch.as_tensor(values, dtype=torch.float32, device=self.device), min=0.0).view(-1)
        peak = float(torch.max(values).item()) if int(values.numel()) > 0 else 0.0
        if peak <= 1e-8:
            return None
        return values / max(peak, 1e-8)
