"""Shared unified preplay defaults.

Edit these values once to keep controller and driver preplay settings in sync.
"""

UNIFIED_PREPLAY_DEFAULTS = {
    "unified_recurrent_preplay_horizon": 4,
    "unified_preplay_discount_factor": 0.7,
    "unified_preplay_within_direction_beta": 5.0,
    "unified_preplay_num_samples": 7,
    "unified_preplay_sampling_temperature": 0.5,
    "unified_preplay_turn_offsets": [-1, 0, 1],
    "unified_preplay_normalize_transitions": True,
    "unified_preplay_global_score_normalization": False,
    "unified_preplay_executable_rollouts": True,
    "unified_preplay_boundary_mode": "hard_block",
    "unified_preplay_microtrajectory_safety_margin": 0.05,
    "unified_preplay_blocked_return_penalty": -1.0,
    "unified_preplay_no_reward_threshold": 1e-6,
    "unified_preplay_decision_diagnostics": True,
    "unified_preplay_decision_diagnostics_stride": 1,
    "exploit_loop_recovery": True,
}
