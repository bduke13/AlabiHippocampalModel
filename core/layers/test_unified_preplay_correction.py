import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from core.layers.multiscale_pcn_with_gcn_v2 import UnifiedMultiScalePCN
from core.layers.unified_reward_cell import UnifiedRewardCell


def build_test_pcn() -> UnifiedMultiScalePCN:
    return UnifiedMultiScalePCN(
        scale_configs=[
            {
                "num_pc": 4,
                "num_bvc_per_dir": 1,
                "num_grid_cells": 1,
                "num_modules": 1,
                "cells_per_module": 1,
            }
        ],
        n_hd=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


class UnifiedPreplayCorrectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pcn = build_test_pcn()
        with torch.no_grad():
            self.pcn.w_rec_unified.zero_()
            self.pcn.w_rec_unified[0] = torch.tensor(
                [
                    [0.6, 0.2, 0.0, 0.0],
                    [0.4, 0.8, 0.0, 0.0],
                    [0.0, 0.0, 0.5, 0.1],
                    [0.0, 0.0, 0.5, 0.9],
                ],
                dtype=self.pcn.dtype,
            )
            self.pcn.w_rec_unified[1] = torch.tensor(
                [
                    [0.1, 0.7, 0.0, 0.0],
                    [0.9, 0.3, 0.0, 0.0],
                    [0.0, 0.0, 0.8, 0.4],
                    [0.0, 0.0, 0.2, 0.6],
                ],
                dtype=self.pcn.dtype,
            )
            self.pcn.w_rec_unified[2] = torch.full(
                (4, 4), 0.25, dtype=self.pcn.dtype
            )
            self.pcn.w_rec_unified[3] = torch.tensor(
                [
                    [0.3, 0.0, 0.2, 0.0],
                    [0.7, 0.0, 0.8, 0.0],
                    [0.0, 0.4, 0.0, 0.1],
                    [0.0, 0.6, 0.0, 0.9],
                ],
                dtype=self.pcn.dtype,
            )

    def _manual_one_step(self, states: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
        recurrent_weights = torch.clamp(torch.nan_to_num(self.pcn.w_rec_unified), min=0.0)
        sums = torch.sum(recurrent_weights, dim=1, keepdim=True)
        transitions = recurrent_weights / torch.clamp(sums, min=1e-12)
        outputs = []
        for state, hd in zip(states, dirs):
            recurrent = transitions[int(hd.item())] @ state
            outputs.append(torch.tanh(torch.relu(recurrent - state)))
        return torch.stack(outputs, dim=0)

    def test_one_step_preplay_matches_manual_expected(self) -> None:
        states = torch.tensor(
            [
                [0.4, 0.3, 0.2, 0.1],
                [0.1, 0.6, 0.2, 0.3],
            ],
            dtype=self.pcn.dtype,
        )
        dirs = torch.tensor([0, 5], dtype=torch.long)
        expected = self._manual_one_step(states, torch.tensor([0, 1], dtype=torch.long))
        actual = self.pcn._one_step_preplay_v2(states, dirs)
        torch.testing.assert_close(actual, expected)

    def test_one_step_preplay_expands_single_direction(self) -> None:
        states = torch.tensor(
            [
                [0.2, 0.4, 0.1, 0.0],
                [0.5, 0.1, 0.3, 0.2],
            ],
            dtype=self.pcn.dtype,
        )
        expanded_dirs = torch.tensor([3, 3], dtype=torch.long)
        expected = self.pcn._one_step_preplay_v2(states, expanded_dirs)
        actual = self.pcn._one_step_preplay_v2(states, 3)
        torch.testing.assert_close(actual, expected)

    def test_one_step_preplay_is_side_effect_free(self) -> None:
        with torch.no_grad():
            self.pcn.place_cell_activations.copy_(
                torch.tensor([0.9, 0.1, 0.0, 0.3], dtype=self.pcn.dtype)
            )
            self.pcn.place_cell_trace.copy_(
                torch.tensor([0.4, 0.2, 0.1, 0.0], dtype=self.pcn.dtype)
            )
        self.pcn.learning_step_count = 7
        self.pcn.last_learning_stdp_active = True

        before_place = self.pcn.place_cell_activations.clone()
        before_trace = self.pcn.place_cell_trace.clone()
        before_weights = self.pcn.w_rec_unified.clone()
        before_step_count = self.pcn.learning_step_count
        before_stdp_flag = self.pcn.last_learning_stdp_active

        _ = self.pcn._one_step_preplay_v2(
            torch.tensor([[0.2, 0.4, 0.1, 0.3]], dtype=self.pcn.dtype),
            torch.tensor([1], dtype=torch.long),
        )

        torch.testing.assert_close(self.pcn.place_cell_activations, before_place)
        torch.testing.assert_close(self.pcn.place_cell_trace, before_trace)
        torch.testing.assert_close(self.pcn.w_rec_unified, before_weights)
        self.assertEqual(self.pcn.learning_step_count, before_step_count)
        self.assertEqual(self.pcn.last_learning_stdp_active, before_stdp_flag)

    def test_preplay_from_state_batched_matches_shared_helper_for_one_step(self) -> None:
        states = torch.tensor(
            [
                [0.2, 0.4, 0.1, 0.0],
                [0.5, 0.1, 0.3, 0.2],
            ],
            dtype=self.pcn.dtype,
        )
        dirs = torch.tensor([1, 2], dtype=torch.long)
        expected = self.pcn._one_step_preplay_v2(states, dirs)
        actual = self.pcn.preplay_from_state_batched(states, dirs, num_steps=1)
        torch.testing.assert_close(actual, expected)

    def test_unified_preplay_sampling_uses_reward_cell_scoring(self) -> None:
        reward = UnifiedRewardCell(
            num_place_cells=self.pcn.num_pc_total,
            scale_configs=[{"num_pc": self.pcn.num_pc_total}],
            device=self.pcn.device,
        )
        with torch.no_grad():
            self.pcn.place_cell_activations.copy_(
                torch.tensor([0.4, 0.3, 0.2, 0.1], dtype=self.pcn.dtype)
            )
            reward.w_in_effective = torch.tensor(
                [[1.0, 0.5, 0.0, 0.0]],
                dtype=torch.float32,
                device=self.pcn.device,
            )

        dirs = torch.arange(self.pcn.n_hd, dtype=torch.long)
        start_states = self.pcn.place_cell_activations.unsqueeze(0).expand(self.pcn.n_hd, -1).clone()
        current_reward = reward.compute_reward_activations_batched(
            self.pcn.place_cell_activations.unsqueeze(0)
        ).to(dtype=self.pcn.dtype).view(1)[0]
        first_step = self.pcn._one_step_preplay_v2(start_states, dirs)
        second_step = self.pcn._one_step_preplay_v2(first_step, dirs)
        expected = (
            (reward.compute_reward_activations_batched(first_step).to(dtype=self.pcn.dtype) - current_reward)
            + 0.5 * (reward.compute_reward_activations_batched(second_step).to(dtype=self.pcn.dtype) - current_reward)
        ).to(dtype=self.pcn.dtype)
        raw_first_step = torch.sum(
            first_step * reward.w_in_effective.view(-1).to(dtype=self.pcn.dtype),
            dim=1,
        )
        self.assertFalse(torch.allclose(expected[:1], raw_first_step[:1]))

        (
            _final_direction,
            _expected_value,
            _combined_vector,
            macro_returns,
            _macro_vectors,
            _sampling_variances,
            _probs,
        ) = self.pcn.unified_preplay_sampling(
            unified_rcn=reward,
            n_hd=self.pcn.n_hd,
            num_steps=2,
            discount_factor=0.5,
            within_direction_beta=1.0,
            num_samples=1,
            sampling_temperature=1.0,
            turn_offsets=[0],
        )

        torch.testing.assert_close(macro_returns, expected)

if __name__ == "__main__":
    unittest.main()
