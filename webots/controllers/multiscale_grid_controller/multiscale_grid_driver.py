import csv
import json
import math
import pickle
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

try:
    from controller import Supervisor
except ImportError:
    class Supervisor:  # Allows local syntax checks outside Webots.
        pass


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.layers.head_direction_layer import HeadDirectionLayer
from core.layers.unified_multiscale_pcn import UnifiedMultiScalePCN
from core.layers.unified_reward_cell import DEFAULT_REPLAY_TAU, DEFAULT_REPLAY_TIMESTEPS, UnifiedRewardCell
from core.robot.robot_mode import RobotMode


GOALS_BY_WORLD = {
    "20x20": [-7.0, 7.0],
    "20x20_1obstacle": [-7.0, 7.0],
    "20x20_2obstacles": [-7.0, 7.0],
    "20x20_goalBehindWall": [-2.0, 2.0],
}


DEFAULT_SCALES = [
    {
        "scale_index": 0,
        "name": "small",
        "sigma_pc_s": 1.0,
        "num_pc": 800,
        "sigma_r": 0.5,
        "sigma_theta": 1,
        "num_bvc_per_dir": 100,
        "w_in_init_ratio": 0.3,
        "w_grid_init_ratio": 0.3,
        "gamma_pp": 1.0,
        "gamma_pb": 0.35,
        "d_opt": 0.7,
        "sigma_tune_k": 1.0,
        "grid_influence": 0.15,
        "gamma_pg": 0.35,
        "num_grid_cells": 800,
        "num_modules": 8,
        "cells_per_module": 100,
        "spread_range": (1.5, 1.5),
        "scale_multiplier": 4,
        "module_scale_ratio": 1.0,
        "translation_scale": 2.0,
        "mask_resolution": 256,
        "smooth_sigma": 0.5,
        "activation_cache_size": 1024,
        "activation_cache_quantization": 1e-4,
        "alpha_pb": 0.447,
        "alpha_pg": 0.447,
        "connection_decay_rate": 1e-4,
    },
    {
        "scale_index": 1,
        "name": "medium",
        "sigma_pc_s": 1.5,
        "num_pc": 450,
        "sigma_r": 1.0,
        "sigma_theta": 3,
        "num_bvc_per_dir": 80,
        "w_in_init_ratio": 0.25,
        "w_grid_init_ratio": 0.3,
        "gamma_pp": 0.9,
        "gamma_pb": 0.3,
        "d_opt": 2.5,
        "sigma_tune_k": 1.0,
        "grid_influence": 0.15,
        "gamma_pg": 0.35,
        "num_grid_cells": 400,
        "num_modules": 8,
        "cells_per_module": 50,
        "spread_range": (1.5, 1.5),
        "scale_multiplier": 5.5,
        "module_scale_ratio": 1.0,
        "translation_scale": 2.0,
        "mask_resolution": 128,
        "smooth_sigma": 0.5,
        "activation_cache_size": 1024,
        "activation_cache_quantization": 1e-4,
        "alpha_pb": 0.632,
        "alpha_pg": 0.632,
        "connection_decay_rate": 1.5e-4,
    },
    {
        "scale_index": 2,
        "name": "large",
        "sigma_pc_s": 3.0,
        "num_pc": 150,
        "sigma_r": 1.5,
        "sigma_theta": 5,
        "num_bvc_per_dir": 75,
        "w_in_init_ratio": 0.2,
        "w_grid_init_ratio": 0.2,
        "gamma_pp": 1.1,
        "gamma_pb": 0.25,
        "d_opt": 5.0,
        "sigma_tune_k": 1.0,
        "large_scale_one_sided": True,
        "large_scale_plateau": 1.0,
        "large_scale_plateau_onset_sigma": 1.0,
        "large_scale_plateau_full_sigma": 2.0,
        "grid_influence": 0.15,
        "gamma_pg": 0.25,
        "num_grid_cells": 400,
        "num_modules": 8,
        "cells_per_module": 50,
        "spread_range": (1.5, 1.5),
        "scale_multiplier": 7,
        "module_scale_ratio": 1.0,
        "translation_scale": 2.0,
        "mask_resolution": 128,
        "smooth_sigma": 0.5,
        "activation_cache_size": 1024,
        "activation_cache_quantization": 1e-4,
        "connection_decay_rate": 2e-4,
    },
]


LEARNING_MODES = {
    RobotMode.LEARN_OJAS,
    RobotMode.LEARN_OJAS_AUTO,
    RobotMode.LEARN_HEBB,
    RobotMode.DMTP,
    RobotMode.LEARNING,
    RobotMode.LEARN_LOCATIONS,
    RobotMode.LEARN_LOCATIONS_COVERAGE,
    RobotMode.LEARN_LOCATIONS_COVERAGE_AUTO,
}
COVERAGE_LEARNING_MODES = {
    RobotMode.LEARN_LOCATIONS_COVERAGE,
    RobotMode.LEARN_LOCATIONS_COVERAGE_AUTO,
}
EXPLOIT_MODES = {
    RobotMode.EXPLOIT,
    RobotMode.EXPLOIT_LOCATIONS,
    RobotMode.EXPLOIT_LOCATIONS_RANDOM,
    RobotMode.EXPLOIT_LOCATIONS_RANDOM_AUTO,
}
PLOTTING_MODES = {
    RobotMode.PLOTTING,
    RobotMode.PLOTTING_AUTO,
    RobotMode.PLOTTING_COVERAGE_AUTO,
}


class Driver(Supervisor):
    def initialization(
        self,
        mode: RobotMode = RobotMode.LEARN_LOCATIONS_COVERAGE,
        run_time_hours: float = 15.0,
        start_loc: Optional[Sequence[float]] = None,
        randomize_start_loc: bool = False,
        goal_location: Optional[Sequence[float]] = None,
        goal_radius: float = 0.7,
        scale_configs: Optional[List[Dict]] = None,
        max_dist: float = 25.0,
        reset_networks: bool = False,
        use_cuda: bool = False,
        enable_ojas: Optional[bool] = None,
        enable_stdp: Optional[bool] = None,
        pcn_learning_stdp_start_steps: int = 0,
        bvc_context_gain_floor: float = 0.10,
        bvc_context_gain_strength: float = 1.0,
        environment_size: Optional[Sequence[float]] = None,
        grid_size: float = 0.5,
        coverage_percentage: Optional[float] = 0.95,
        minimum_learning_duration_hours: float = 4.0,
        min_goal_visits: int = 5,
        goal_visit_cooldown_seconds: float = 6.0,
        goal_exit_hysteresis: float = 0.1,
        tau_w: int = 10,
        unified_recurrent_preplay_horizon: int = 2,
        unified_exploit_step_distance: float = 0.25,
        enable_live_diagnostics: bool = True,
        trials_per_goal: int = 1,
        min_spawn_distance: float = 6.0,
        wall_clearance: float = 0.5,
        path_failure_ratio: Optional[float] = None,
        auto_trial_name: Optional[str] = None,
        num_auto_trials: int = 5,
        current_auto_trial: int = 1,
    ) -> None:
        self.robot_mode = mode
        self.timestep = 96
        self.tau_w = int(max(1, tau_w))
        self.n_hd = 8
        self.max_speed = 7.0 if mode in EXPLOIT_MODES else 12.0
        self.left_speed = 0.0
        self.right_speed = 0.0
        self.wheel_radius = 0.031
        self.axle_length = 0.271756
        self.max_dist = float(max_dist)
        self.goal_radius = float(goal_radius)
        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32
        self.run_time_limit_s = max(0.0, float(run_time_hours) * 3600.0)
        self.minimum_learning_duration_s = max(0.0, float(minimum_learning_duration_hours) * 3600.0)
        self.step_count = 0
        self.done = False
        self.scale_configs = [dict(cfg) for cfg in (scale_configs or DEFAULT_SCALES)]
        self.pcn_learning_stdp_start_steps = int(max(0, pcn_learning_stdp_start_steps))
        self.bvc_context_gain_floor = float(min(0.95, max(0.0, bvc_context_gain_floor)))
        self.bvc_context_gain_strength = float(max(0.0, bvc_context_gain_strength))
        self.environment_size = list(environment_size or [20.0, 20.0])
        self.grid_size = float(max(1e-6, grid_size))
        self.coverage_percentage = coverage_percentage
        self.target_coverage_percentage = float(coverage_percentage or 0.0)
        self.min_goal_visits = int(max(0, min_goal_visits))
        self.goal_visit_cooldown_s = float(max(0.0, goal_visit_cooldown_seconds))
        self.goal_exit_hysteresis = float(max(0.0, goal_exit_hysteresis))
        self.unified_recurrent_preplay_horizon = int(max(1, unified_recurrent_preplay_horizon))
        self.unified_exploit_step_distance = float(max(0.01, unified_exploit_step_distance))
        self.enable_live_diagnostics = bool(enable_live_diagnostics)
        self.auto_trial_name = auto_trial_name
        self.num_auto_trials = int(max(1, num_auto_trials))
        self.current_auto_trial = int(max(1, current_auto_trial))
        self.trials_per_goal = int(max(1, trials_per_goal))
        self.min_spawn_distance = float(max(0.0, min_spawn_distance))
        self.wall_clearance = float(max(0.0, wall_clearance))
        self.path_failure_ratio = None if path_failure_ratio is None else float(max(0.0, path_failure_ratio))

        self.robot = self.getFromDef("agent")
        self.world_name = Path(self.getWorldPath()).stem
        self.world_walls = self._load_world_walls()
        self.goal_location = list(goal_location or GOALS_BY_WORLD.get(self.world_name, [-7.0, 7.0]))
        if mode in {
            RobotMode.LEARN_LOCATIONS_COVERAGE_AUTO,
            RobotMode.EXPLOIT_LOCATIONS_RANDOM_AUTO,
            RobotMode.PLOTTING_AUTO,
            RobotMode.PLOTTING_COVERAGE_AUTO,
        } and auto_trial_name:
            base_dir = Path(__file__).resolve().parent / f"pkl_{auto_trial_name}" / f"{self.world_name}_{self.current_auto_trial}"
        else:
            base_dir = Path(__file__).resolve().parent / "pkl" / self.world_name
        self.trial_base_dir = base_dir
        self.network_dir = base_dir / "networks"
        self.hmap_dir = base_dir / "hmaps"
        self.network_dir.mkdir(parents=True, exist_ok=True)
        self.hmap_dir.mkdir(parents=True, exist_ok=True)

        self._place_robot(start_loc, randomize_start_loc)
        self._init_devices()
        self.head_direction_layer = HeadDirectionLayer(self.n_hd, device=self.device)

        self.pcn_path = self.network_dir / "unified_pcn.pkl"
        self.rcn_path = self.network_dir / "unified_rcn_goal.pkl"
        if reset_networks:
            for path in (self.pcn_path, self.rcn_path):
                if path.exists():
                    path.unlink()
        self.pcn = self._load_or_create_pcn(self.scale_configs)
        self.rcn = self._load_or_create_rcn(self.scale_configs)
        self.pcn.enable_ojas = bool(enable_ojas) if enable_ojas is not None else mode in {
            RobotMode.LEARN_OJAS,
            RobotMode.LEARN_OJAS_AUTO,
            RobotMode.DMTP,
            RobotMode.LEARNING,
            RobotMode.LEARN_LOCATIONS,
            RobotMode.LEARN_LOCATIONS_COVERAGE,
            RobotMode.LEARN_LOCATIONS_COVERAGE_AUTO,
        }
        self.pcn.enable_stdp = bool(enable_stdp) if enable_stdp is not None else mode in {
            RobotMode.LEARN_HEBB,
            RobotMode.DMTP,
            RobotMode.LEARNING,
            RobotMode.LEARN_LOCATIONS,
            RobotMode.LEARN_LOCATIONS_COVERAGE,
            RobotMode.LEARN_LOCATIONS_COVERAGE_AUTO,
        }

        self.hmap_loc = []
        self.hmap_hdn = []
        self.hmap_prox = []
        self.hmap_pcn_scales = [[] for _ in self.scale_configs]
        self.hmap_gcn_scales = [[] for _ in self.scale_configs]
        self.hmap_reward = []
        self.goal_events: List[List[torch.Tensor]] = []
        self._current_goal_event: List[torch.Tensor] = []
        self.goal_visit_count = 0
        self.goal_last_count_time_s = -1.0e9
        self._was_at_goal = False
        self.coverage_valid_cells = self._build_coverage_valid_cells()
        self.coverage_cells = set()
        self.current_coverage_percentage = 0.0
        self.visited_cells = 0
        self.total_grid_cells = max(1, len(self.coverage_valid_cells))
        self._last_coverage_log_step = -10**9
        if self.coverage_percentage is not None:
            target_cells = int(math.ceil(self.target_coverage_percentage * self.total_grid_cells))
            print(
                f"[COVERAGE] tracking {self.total_grid_cells} valid cells; "
                f"target={target_cells} cells ({self.target_coverage_percentage * 100:.1f}%)"
            )
        self.last_unified_committed_hd_bin = None
        self.last_unified_committed_score = None
        self.last_executed_move_heading_deg = None
        self.last_executed_move_hd_bin = None
        self.last_executed_move_distance = 0.0
        self.raw_boundaries = torch.full((720,), self.max_dist, dtype=self.dtype, device=self.device)
        self.boundaries = self.raw_boundaries.clone()
        self.hd_activations = torch.zeros(self.n_hd, dtype=self.dtype, device=self.device)
        self.collided = torch.zeros(2, dtype=torch.int32, device=self.device)
        self.current_heading_deg = 0.0
        self.prox = 0.0
        self.trajectory_records: List[Dict] = []
        self.scale_diagnostics: List[Dict] = []
        self.exploit_decisions: List[Dict] = []
        self.exploit_trials: List[Dict] = []
        self.termination_reason = None
        self.trial_index = 1
        self.trial_step_count = 0
        self.trial_start_step = 0
        self.trial_start_time_s = self._sim_time()
        self.trial_start_xy = self._current_planar_xy()
        self.trial_start_location = list(self.trial_start_xy)
        self.trial_direct_distance = self._distance_to_goal(self.trial_start_xy)
        self.current_trial_path_length = 0.0
        self._last_path_xy = None
        self._finalized_current_trial = False
        self.step(self.timestep)

    def _place_robot(self, start_loc: Optional[Sequence[float]], randomize_start_loc: bool) -> None:
        if randomize_start_loc:
            for _ in range(200):
                candidate = [random.uniform(-8.0, 8.0), 0.0441865, random.uniform(-8.0, 8.0)]
                if np.linalg.norm([candidate[0] - self.goal_location[0], candidate[2] - self.goal_location[1]]) > 2.0:
                    self.robot.getField("translation").setSFVec3f(candidate)
                    self.robot.resetPhysics()
                    return
        if start_loc is not None:
            self.robot.getField("translation").setSFVec3f([float(start_loc[0]), 0.0441865, float(start_loc[1])])
            self.robot.resetPhysics()

    def _device(self, name: str, enable: bool = True):
        try:
            dev = self.getDevice(name)
            if enable and hasattr(dev, "enable"):
                dev.enable(self.timestep)
            return dev
        except Exception:
            return None

    def _init_devices(self) -> None:
        self.compass = self._device("compass")
        self.range_finder = self._device("range-finder")
        self.left_bumper = self._device("bumper_left")
        self.right_bumper = self._device("bumper_right")
        self.left_motor = self.getDevice("left wheel motor")
        self.right_motor = self.getDevice("right wheel motor")
        self.left_position_sensor = self._device("left wheel sensor")
        self.right_position_sensor = self._device("right wheel sensor")
        self.left_motor.setPosition(float("inf"))
        self.right_motor.setPosition(float("inf"))

    def _load_or_create_pcn(self, scale_configs: List[Dict]) -> UnifiedMultiScalePCN:
        if self.pcn_path.exists():
            with open(self.pcn_path, "rb") as f:
                pcn = pickle.load(f)
            expected_pc = sum(int(cfg["num_pc"]) for cfg in scale_configs)
            compatible = (
                getattr(pcn, "num_pc_total", None) == expected_pc
                and hasattr(pcn, "w_in")
                and hasattr(pcn, "w_grid")
                and hasattr(pcn, "bvc_boundaries")
                and hasattr(pcn, "grid_boundaries")
                and hasattr(pcn, "grid_layers")
            )
            if compatible:
                print(f"[MODEL] Loaded {self.pcn_path}")
                pcn.device = self.device
                pcn.dtype = self.dtype
                pcn.scale_configs = [dict(cfg) for cfg in scale_configs]
                pcn.learning_stdp_start_steps = self.pcn_learning_stdp_start_steps
                pcn.bvc_context_gain_floor = self.bvc_context_gain_floor
                pcn.bvc_context_gain_strength = self.bvc_context_gain_strength
                pcn.enable_live_diagnostics = self.enable_live_diagnostics
                pcn.configure_eta_stdp([float(cfg.get("eta_stdp", 0.3)) for cfg in scale_configs])
                pcn.configure_connection_decay_rates(
                    [float(cfg.get("connection_decay_rate", 1e-4)) for cfg in scale_configs]
                )
                return pcn
            print(f"[MODEL] Rebuilding incompatible PCN pickle: {self.pcn_path}")
        print("[MODEL] Creating unified PCN")
        connection_decay_values = [
            float(cfg.get("connection_decay_rate", 1e-4)) for cfg in scale_configs
        ]
        return UnifiedMultiScalePCN(
            scale_configs=scale_configs,
            timestep=self.timestep,
            n_hd=self.n_hd,
            n_res=720,
            max_dist=self.max_dist,
            world_name=self.world_name,
            enable_ojas=False,
            enable_stdp=False,
            grid_influence=float(np.mean([cfg.get("grid_influence", 0.3) for cfg in scale_configs])),
            gamma_pp=float(np.mean([cfg.get("gamma_pp", 0.5) for cfg in scale_configs])),
            gamma_pb=float(np.mean([cfg.get("gamma_pb", 0.3) for cfg in scale_configs])),
            gamma_pg=float(np.mean([cfg.get("gamma_pg", 0.3) for cfg in scale_configs])),
            learning_stdp_start_steps=self.pcn_learning_stdp_start_steps,
            eta_stdp=[float(cfg.get("eta_stdp", 0.3)) for cfg in scale_configs],
            connection_decay_rate=connection_decay_values,
            bvc_context_gain_floor=self.bvc_context_gain_floor,
            bvc_context_gain_strength=self.bvc_context_gain_strength,
            enable_live_diagnostics=self.enable_live_diagnostics,
            device=self.device,
            dtype=self.dtype,
        )

    def _load_or_create_rcn(self, scale_configs: List[Dict]) -> UnifiedRewardCell:
        if self.rcn_path.exists():
            with open(self.rcn_path, "rb") as f:
                rcn = pickle.load(f)
            compatible = (
                getattr(rcn, "num_place_cells", None) == self.pcn.num_place_cells_total
                and hasattr(rcn, "replay_from_seed")
                and hasattr(rcn, "build_goal_reward_from_events")
            )
            if compatible:
                print(f"[MODEL] Loaded {self.rcn_path}")
                rcn.device = self.device
                rcn.reconfigure_from_scale_configs(scale_configs)
                rcn.replay_timesteps = int(DEFAULT_REPLAY_TIMESTEPS)
                rcn.replay_tau = float(DEFAULT_REPLAY_TAU)
                return rcn
            print(f"[MODEL] Rebuilding incompatible RCN pickle: {self.rcn_path}")
        return UnifiedRewardCell(
            self.pcn.num_place_cells_total,
            scale_configs=scale_configs,
            device=self.device,
        )

    def run(self) -> None:
        print(f"[DRIVER] Starting robot in {self.robot_mode}")
        print(f"[DRIVER] Single goal at {self.goal_location}")
        if self.robot_mode in {RobotMode.EXPLOIT_LOCATIONS_RANDOM, RobotMode.EXPLOIT_LOCATIONS_RANDOM_AUTO}:
            self._run_exploit_trial_batch()
            self.stop()
            self.save()
            try:
                self.simulationSetMode(self.SIMULATION_MODE_PAUSE)
            except Exception:
                pass
            return

        while not self.done:
            if self.robot_mode in EXPLOIT_MODES:
                self.exploit()
            elif self.robot_mode in LEARNING_MODES or self.robot_mode in PLOTTING_MODES:
                self.explore()
            else:
                print(f"[DRIVER] Unknown mode {self.robot_mode}; stopping")
                self.done = True

        self.stop()
        if self.robot_mode in LEARNING_MODES:
            self._finalize_goal_event()
            if self.goal_events:
                self.build_goal_reward()
        if self.robot_mode in EXPLOIT_MODES:
            self._finalize_exploit_trial()
        self.save()
        try:
            self.simulationSetMode(self.SIMULATION_MODE_PAUSE)
        except Exception:
            pass

    def _generate_random_spawn(self) -> List[float]:
        half_w = float(self.environment_size[0]) / 2.0
        half_h = float(self.environment_size[1]) / 2.0
        margin = max(0.1, self.wall_clearance)
        for _ in range(1000):
            x = random.uniform(-half_w + margin, half_w - margin)
            y = random.uniform(-half_h + margin, half_h - margin)
            if self._distance_to_goal([x, y]) < self.min_spawn_distance:
                continue
            if self._point_clear_of_walls(x, y, margin=margin):
                return [x, y]
        return [random.uniform(-half_w + margin, half_w - margin), random.uniform(-half_h + margin, half_h - margin)]

    def _reset_exploit_trial(self, start_xy: Sequence[float], trial_index: int) -> None:
        self.stop()
        self.robot.getField("translation").setSFVec3f([float(start_xy[0]), 0.0441865, float(start_xy[1])])
        self.robot.resetPhysics()
        if hasattr(self.pcn, "reset_runtime_state"):
            self.pcn.reset_runtime_state(reset_traces=True)
        self.trial_index = int(trial_index)
        self.trial_step_count = 0
        self.trial_start_step = int(self.step_count)
        self.trial_start_time_s = self._sim_time()
        self.trial_start_xy = [float(start_xy[0]), float(start_xy[1])]
        self.trial_start_location = list(self.trial_start_xy)
        self.trial_direct_distance = self._distance_to_goal(self.trial_start_xy)
        self.current_trial_path_length = 0.0
        self.termination_reason = None
        self.done = False
        self._last_path_xy = None
        self._finalized_current_trial = False
        self._reset_unified_heading_commit_state()
        self.step(self.timestep)

    def _run_exploit_trial_batch(self) -> None:
        starts = []
        while len(starts) < self.trials_per_goal:
            starts.append(self._generate_random_spawn())
        for idx, start_xy in enumerate(starts, start=1):
            print(f"[EXPLOIT-TRIAL] {idx}/{len(starts)} start=({start_xy[0]:.2f},{start_xy[1]:.2f})")
            self._reset_exploit_trial(start_xy, idx)
            while not self.done:
                self.exploit()
            self._finalize_exploit_trial()
        self.done = True

    def _finalize_exploit_trial(self) -> None:
        if self._finalized_current_trial:
            return
        final_xy = self._current_planar_xy()
        final_distance = self._distance_to_goal(final_xy)
        reason = self.termination_reason or ("goal_reached" if final_distance <= self.goal_radius else "stopped")
        success = bool(final_distance <= self.goal_radius or reason == "goal_reached")
        self.exploit_trials.append(
            {
                "trial": int(self.trial_index),
                "start_xy": [float(self.trial_start_xy[0]), float(self.trial_start_xy[1])],
                "goal_xy": [float(self.goal_location[0]), float(self.goal_location[1])],
                "final_xy": [float(final_xy[0]), float(final_xy[1])],
                "success": success,
                "termination_reason": str(reason),
                "steps": int(self.trial_step_count),
                "time_s": float(self._sim_time() - self.trial_start_time_s),
                "path_length_m": float(self.current_trial_path_length),
                "straight_line_m": float(self.trial_direct_distance),
                "path_ratio_to_straight": float(self.current_trial_path_length / max(self.trial_direct_distance, 1e-6)),
                "final_distance_to_goal_m": float(final_distance),
                "decision_count": int(sum(1 for d in self.exploit_decisions if int(d.get("trial", -1)) == int(self.trial_index))),
            }
        )
        self._finalized_current_trial = True

    def sense(self) -> None:
        self.step(self.timestep)
        if self.range_finder is not None:
            raw = torch.as_tensor(self.range_finder.getRangeImage(), dtype=self.dtype, device=self.device).flatten()
            self.raw_boundaries = torch.nan_to_num(raw, nan=self.max_dist, posinf=self.max_dist, neginf=0.0)
            self.raw_boundaries = torch.clamp(self.raw_boundaries, min=0.0, max=self.max_dist)
        if self.compass is not None:
            self.current_heading_deg = round(self.get_bearing_in_degrees(self.compass.getValues()), 1)
        lidar_shift = (float(self.raw_boundaries.numel()) / 360.0) * float(self.current_heading_deg)
        self.boundaries = self._circular_roll_float_1d(self.raw_boundaries, shift_bins=lidar_shift)
        rad = self._clockwise_heading_deg_to_standard_rad(self.current_heading_deg)
        self.hd_activations = self.head_direction_layer.get_hd_activation(
            torch.tensor([np.cos(rad), np.sin(rad)], dtype=self.dtype, device=self.device)
        )
        if self.left_bumper is not None:
            self.collided[0] = int(self.left_bumper.getValue())
        if self.right_bumper is not None:
            self.collided[1] = int(self.right_bumper.getValue())

    def _circular_roll_float_1d(self, values: torch.Tensor, shift_bins: float) -> torch.Tensor:
        values = values.reshape(-1)
        n = int(values.shape[0])
        if n <= 1:
            return values
        shift = float(shift_bins) % float(n)
        base = int(np.floor(shift))
        frac = shift - float(base)
        if frac <= 1e-9:
            return torch.roll(values, shifts=base)
        lo = torch.roll(values, shifts=base)
        hi = torch.roll(values, shifts=base + 1)
        return ((1.0 - frac) * lo) + (frac * hi)

    def get_bearing_in_degrees(self, north) -> float:
        bearing = (np.arctan2(north[1], north[0]) - 1.5708) / np.pi * 180.0
        return float(bearing + 360.0 if bearing < 0.0 else bearing)

    def _clockwise_heading_deg_to_standard_rad(self, heading_deg: float) -> float:
        return -np.deg2rad(float(heading_deg))

    def _translation_to_planar_xy(self, translation: Sequence[float]) -> List[float]:
        return [float(translation[0]), float(translation[2])]

    def _current_planar_xy(self) -> List[float]:
        return self._translation_to_planar_xy(self.robot.getField("translation").getSFVec3f())

    def _sim_time(self) -> float:
        try:
            return float(self.getTime())
        except Exception:
            return float(self.step_count * self.timestep) / 1000.0

    def _distance_to_goal(self, xy: Optional[Sequence[float]] = None) -> float:
        if xy is None:
            xy = self._current_planar_xy()
        return float(np.linalg.norm([float(xy[0]) - self.goal_location[0], float(xy[1]) - self.goal_location[1]]))

    def _load_world_walls(self) -> List[Dict[str, float]]:
        try:
            world_path = Path(self.getWorldPath())
        except Exception:
            world_path = Path()
        if not world_path.exists():
            return []
        text = world_path.read_text(encoding="utf-8", errors="ignore")
        walls = []
        for match in re.finditer(r"Wall\s*\{(?P<body>.*?)\n\}", text, flags=re.DOTALL):
            body = match.group("body")
            trans_match = re.search(r"translation\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)", body)
            rot_match = re.search(r"rotation\s+[-+0-9.eE]+\s+[-+0-9.eE]+\s+[-+0-9.eE]+\s+([-+0-9.eE]+)", body)
            size_match = re.search(r"size\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)", body)
            if not size_match:
                continue
            tx = float(trans_match.group(1)) if trans_match else 0.0
            tz = float(trans_match.group(3)) if trans_match else 0.0
            theta = float(rot_match.group(1)) if rot_match else 0.0
            length = float(size_match.group(1))
            thickness = float(size_match.group(3))
            walls.append({"x": tx, "y": tz, "theta": theta, "length": length, "thickness": thickness})
        return walls

    def _point_clear_of_walls(self, x: float, y: float, margin: Optional[float] = None) -> bool:
        margin = self.wall_clearance if margin is None else float(margin)
        half_w = float(self.environment_size[0]) / 2.0
        half_h = float(self.environment_size[1]) / 2.0
        if not (-half_w + margin <= x <= half_w - margin and -half_h + margin <= y <= half_h - margin):
            return False
        for wall in self.world_walls:
            dx = x - wall["x"]
            dy = y - wall["y"]
            c = math.cos(-wall["theta"])
            s = math.sin(-wall["theta"])
            lx = (c * dx) - (s * dy)
            ly = (s * dx) + (c * dy)
            if abs(lx) <= 0.5 * wall["length"] + margin and abs(ly) <= 0.5 * wall["thickness"] + margin:
                return False
        return True

    def _coverage_grid_shape(self) -> Tuple[int, int]:
        nx = max(1, int(np.ceil(float(self.environment_size[0]) / self.grid_size)))
        ny = max(1, int(np.ceil(float(self.environment_size[1]) / self.grid_size)))
        return nx, ny

    def _coverage_cell_center(self, ix: int, iy: int) -> Tuple[float, float]:
        half_w = float(self.environment_size[0]) / 2.0
        half_h = float(self.environment_size[1]) / 2.0
        x = -half_w + (float(ix) + 0.5) * self.grid_size
        y = -half_h + (float(iy) + 0.5) * self.grid_size
        return x, y

    def _coverage_cell_for_xy(self, x: float, y: float) -> Optional[Tuple[int, int]]:
        half_w = float(self.environment_size[0]) / 2.0
        half_h = float(self.environment_size[1]) / 2.0
        ix = int(np.floor((float(x) + half_w) / self.grid_size))
        iy = int(np.floor((float(y) + half_h) / self.grid_size))
        nx, ny = self._coverage_grid_shape()
        if 0 <= ix < nx and 0 <= iy < ny:
            return ix, iy
        return None

    def _build_coverage_valid_cells(self) -> set:
        valid = set()
        nx, ny = self._coverage_grid_shape()
        for ix in range(nx):
            for iy in range(ny):
                x, y = self._coverage_cell_center(ix, iy)
                if self._point_clear_of_walls(x, y):
                    valid.add((ix, iy))
        if valid:
            return valid
        return {(ix, iy) for ix in range(nx) for iy in range(ny)}

    def _update_coverage(self) -> None:
        if self.coverage_percentage is None:
            return
        x, y = self._current_planar_xy()
        cell = self._coverage_cell_for_xy(x, y)
        if cell is None or cell not in self.coverage_valid_cells:
            return
        self.coverage_cells.add(cell)
        self.visited_cells = len(self.coverage_cells)
        self.current_coverage_percentage = self.visited_cells / max(1, self.total_grid_cells)

    def _learning_active(self) -> bool:
        return self.robot_mode in LEARNING_MODES

    def _compute_proximity_distance(self) -> float:
        readings = self.boundaries.detach().float().view(-1)
        readings = readings[torch.isfinite(readings)]
        if readings.numel() == 0:
            return 0.0
        return float(torch.clamp(torch.min(readings), min=0.0, max=self.max_dist).item())

    def compute_pcn_activations(self, learn: Optional[bool] = None) -> None:
        if learn is None:
            learn = self._learning_active()
        pos = self.robot.getField("translation").getSFVec3f()
        self.prox = self._compute_proximity_distance()
        self.pcn.get_place_cell_activations(
            self.boundaries,
            position=[pos[0], pos[2]],
            hd_activations=self.hd_activations,
            collided=bool(torch.any(self.collided).item()),
            proximity=self.prox,
            learn=bool(learn),
        )
        self.step(self.timestep)

    def _to_float_list(self, value) -> List[float]:
        if value is None:
            return []
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        return [float(v) for v in arr]

    def _scale_mass_values(self, scale_activations: Optional[Sequence[torch.Tensor]] = None) -> List[float]:
        masses = []
        activations = scale_activations if scale_activations is not None else self.pcn.get_activations_per_scale()
        for act in activations:
            masses.append(float(torch.sum(torch.abs(act.detach())).cpu().item()))
        return masses

    def _recurrent_inhibition_mean_values(self, scale_masses: Sequence[float]) -> List[float]:
        total = float(sum(scale_masses))
        values = []
        for idx, _mass in enumerate(scale_masses):
            start = self.pcn.scale_boundaries[idx]
            gamma = float(self.pcn.gamma_pp_per_pc[start].detach().cpu().item())
            values.append(gamma * total)
        return values

    def _record_passive_diagnostics(
        self,
        xy: Sequence[float],
        reward_value: float,
        scale_activations: Optional[Sequence[torch.Tensor]] = None,
    ) -> None:
        scale_masses = self._scale_mass_values(scale_activations)
        dominant_scale = int(np.argmax(scale_masses)) if scale_masses else -1
        scale_pref = self._to_float_list(getattr(self.pcn, "last_scale_preference", None))
        bvc_gain = self._to_float_list(getattr(self.pcn, "last_bvc_context_gain_per_scale", None))
        recurrent_inh = self._recurrent_inhibition_mean_values(scale_masses)
        grid_diag = getattr(self.pcn, "last_grid_diagnostics", None)

        if self.robot_mode in EXPLOIT_MODES and self._last_path_xy is not None:
            dx = float(xy[0]) - float(self._last_path_xy[0])
            dy = float(xy[1]) - float(self._last_path_xy[1])
            delta = float(np.sqrt(dx * dx + dy * dy))
            if np.isfinite(delta) and delta < 2.0:
                self.current_trial_path_length += delta
        self._last_path_xy = [float(xy[0]), float(xy[1])]

        if self.robot_mode in EXPLOIT_MODES:
            row = {
                "step": int(self.step_count),
                "trial": int(self.trial_index),
                "trial_step": int(self.trial_step_count),
                "time_s": float(self._sim_time()),
                "mode": str(self.robot_mode.name),
                "x": float(xy[0]),
                "y": float(xy[1]),
                "heading_deg": float(self.current_heading_deg),
                "reward": float(reward_value),
                "prox": float(self.prox),
                "dominant_scale": int(dominant_scale),
                "path_length_m": float(self.current_trial_path_length),
            }
            for idx, value in enumerate(scale_masses):
                row[f"scale_{idx}_mass"] = float(value)
            self.trajectory_records.append(row)

        self.scale_diagnostics.append(
            {
                "step": int(self.step_count),
                "trial": int(self.trial_index),
                "x": float(xy[0]),
                "y": float(xy[1]),
                "prox": float(self.prox),
                "dominant_scale": int(dominant_scale),
                "scale_masses": scale_masses,
                "scale_preference": scale_pref,
                "bvc_context_gain": bvc_gain,
                "recurrent_inhibition_mean": recurrent_inh,
                "stdp_active": bool(getattr(self.pcn, "last_learning_stdp_active", False)),
                "grid": grid_diag,
            }
        )

    def record(self) -> None:
        pos = self.robot.getField("translation").getSFVec3f()
        self.hmap_loc.append([pos[0], pos[2], pos[1]])
        self.hmap_hdn.append(self.hd_activations.detach().cpu())
        self.hmap_prox.append(float(self.prox))
        reward = self.rcn.compute_reward_activations_batched(
            self.pcn.place_cell_activations.detach().view(1, -1)
        )
        reward_value = float(reward.detach().cpu().reshape(-1)[0].item())
        self.hmap_reward.append(reward_value)
        scale_activations = self.pcn.get_activations_per_scale()
        for idx, act in enumerate(scale_activations):
            self.hmap_pcn_scales[idx].append(act.detach().cpu())
        for idx, act in enumerate(getattr(self.pcn, "last_grid_activations", [])):
            if idx < len(self.hmap_gcn_scales):
                self.hmap_gcn_scales[idx].append(act.detach().cpu())
        self._record_passive_diagnostics([pos[0], pos[2]], reward_value, scale_activations)
        self.step_count += 1
        if self.robot_mode in EXPLOIT_MODES:
            self.trial_step_count += 1

    def explore(self) -> None:
        self._reset_unified_heading_commit_state()
        for _ in range(self.tau_w):
            self.sense()
            self.compute_pcn_activations()
            if torch.any(self.collided):
                self.record()
                self.turn(float(np.random.uniform(-np.pi, np.pi)))
                break
            if (
                self.robot_mode in COVERAGE_LEARNING_MODES or self.robot_mode == RobotMode.PLOTTING_COVERAGE_AUTO
            ):
                self._update_coverage()
            self.check_goal_reached()
            self.record()
            if self.done:
                return
            self.forward()
        if not self.done:
            self.turn(float(np.random.normal(0.0, np.deg2rad(30.0))))

    def exploit(self) -> None:
        self.sense()
        self.compute_pcn_activations(learn=False)
        self.record()
        self.check_goal_reached()
        if self.done:
            return
        if self.step_count <= self.tau_w:
            return
        num_preplay_steps = int(max(1, self.unified_recurrent_preplay_horizon))
        discount_factor = 0.7
        within_direction_beta = 15.0
        num_samples_per_direction = 10
        sampling_temperature = 0.20
        weighted_direction_snap_bin_threshold = 1
        heading_commit_preserve_bin_radius = 1
        heading_commit_keep_score_ratio = 0.90

        try:
            (
                _final_direction_deg,
                expected_value,
                _combined_vector,
                macro_returns,
                macro_vectors,
                _sampling_variances,
                direction_probs,
            ) = self.pcn.unified_preplay_sampling(
                unified_rcn=self.rcn,
                n_hd=self.n_hd,
                num_steps=num_preplay_steps,
                discount_factor=discount_factor,
                within_direction_beta=within_direction_beta,
                num_samples=num_samples_per_direction,
                sampling_temperature=sampling_temperature,
            )
        except Exception as exc:
            print(f"[EXPLOIT] preplay failed, falling back to exploration: {exc}")
            self._reset_unified_heading_commit_state()
            self.explore()
            return

        macro_returns = torch.nan_to_num(macro_returns, nan=0.0, posinf=0.0, neginf=0.0)
        macro_vectors = torch.nan_to_num(macro_vectors, nan=0.0, posinf=0.0, neginf=0.0)
        direction_probs = torch.clamp(torch.nan_to_num(direction_probs, nan=0.0, posinf=0.0, neginf=0.0), min=0.0)
        prob_sum = torch.sum(direction_probs)
        if (not bool(torch.isfinite(prob_sum).item())) or float(prob_sum.item()) <= 1e-9:
            direction_probs = torch.full((self.n_hd,), 1.0 / self.n_hd, dtype=self.dtype, device=self.device)
        else:
            direction_probs = direction_probs / torch.clamp(prob_sum, min=1e-9)

        joint_scores = torch.nan_to_num(direction_probs * macro_returns, nan=0.0, posinf=0.0, neginf=0.0)
        best_idx = int(torch.argmax(joint_scores).item())
        max_score = float(torch.max(joint_scores).item())
        expected_value_scalar = float(torch.nan_to_num(expected_value).item())
        if max_score <= 1e-9 or expected_value_scalar <= 1e-9:
            print("[EXPLOIT] zero reward signal; falling back to exploration")
            self._append_exploit_decision(
                reason="fallback_zero_reward",
                direction_probs=direction_probs,
                macro_returns=macro_returns,
                joint_scores=joint_scores,
                selected_hd_bin=None,
                commanded_heading_deg=None,
                expected_value=expected_value_scalar,
                move_success=False,
            )
            self._reset_unified_heading_commit_state()
            self.explore()
            return

        action_angles = self._get_hd_bin_angles(target_device=self.device, target_dtype=self.dtype)
        combined_vector = torch.sum(direction_probs.unsqueeze(1) * macro_vectors, dim=0)
        if float(torch.norm(combined_vector).item()) < 1e-6:
            action_angle = action_angles[best_idx]
            candidate_idx = best_idx
        else:
            action_angle = torch.atan2(combined_vector[1], combined_vector[0])
            if action_angle < 0:
                action_angle += 2.0 * np.pi
            candidate_idx = int(self._angle_to_hd_bin(float(action_angle.item())))
            circular_delta = abs(candidate_idx - best_idx)
            circular_delta = min(circular_delta, self.n_hd - circular_delta)
            if circular_delta > weighted_direction_snap_bin_threshold:
                candidate_idx = best_idx
                action_angle = action_angles[best_idx]

        candidate_score = float(max(0.0, float(joint_scores[candidate_idx].item())))
        committed_prev = self.last_unified_committed_hd_bin
        final_idx = int(candidate_idx)
        final_score = candidate_score
        if committed_prev is not None:
            committed_prev = int(committed_prev) % self.n_hd
            prev_score = float(max(0.0, float(joint_scores[committed_prev].item())))
            commit_delta = abs(final_idx - committed_prev)
            commit_delta = min(commit_delta, self.n_hd - commit_delta)
            if (
                commit_delta > heading_commit_preserve_bin_radius
                and prev_score >= heading_commit_keep_score_ratio * max(final_score, 1e-9)
            ):
                final_idx = committed_prev
                final_score = prev_score
                action_angle = action_angles[final_idx]

        self.last_unified_committed_hd_bin = int(final_idx)
        self.last_unified_committed_score = float(final_score)
        commanded_heading_deg = self._planar_delta_to_clockwise_heading_deg(
            float(torch.cos(action_angle).item()),
            float(torch.sin(action_angle).item()),
        )
        if commanded_heading_deg is None:
            commanded_heading_deg = (-float(torch.rad2deg(action_angle).item())) % 360.0
        move_success = self._execute_movement(float(commanded_heading_deg))
        nominal_action_distance = self._get_nominal_forward_distance_per_step() * self._get_action_forward_steps()
        self._append_exploit_decision(
            reason="preplay_action",
            direction_probs=direction_probs,
            macro_returns=macro_returns,
            joint_scores=joint_scores,
            selected_hd_bin=final_idx,
            commanded_heading_deg=float(commanded_heading_deg),
            expected_value=expected_value_scalar,
            move_success=bool(move_success),
        )
        if (not move_success) or self.last_executed_move_distance < max(1e-3, 0.15 * nominal_action_distance):
            self._reset_unified_heading_commit_state()

    def _append_exploit_decision(
        self,
        reason: str,
        direction_probs,
        macro_returns,
        joint_scores,
        selected_hd_bin,
        commanded_heading_deg,
        expected_value: float,
        move_success: bool,
    ) -> None:
        xy = self._current_planar_xy()
        self.exploit_decisions.append(
            {
                "step": int(self.step_count),
                "trial": int(self.trial_index),
                "trial_step": int(self.trial_step_count),
                "x": float(xy[0]),
                "y": float(xy[1]),
                "reason": str(reason),
                "direction_probs": self._to_float_list(direction_probs),
                "macro_returns": self._to_float_list(macro_returns),
                "joint_scores": self._to_float_list(joint_scores),
                "selected_hd_bin": None if selected_hd_bin is None else int(selected_hd_bin),
                "commanded_heading_deg": None if commanded_heading_deg is None else float(commanded_heading_deg),
                "actual_heading_deg": self.last_executed_move_heading_deg,
                "actual_hd_bin": self.last_executed_move_hd_bin,
                "move_distance_m": float(self.last_executed_move_distance),
                "move_success": bool(move_success),
                "expected_value": float(expected_value),
                "committed_hd_bin": self.last_unified_committed_hd_bin,
                "committed_score": self.last_unified_committed_score,
            }
        )

    def _reset_unified_heading_commit_state(self) -> None:
        self.last_unified_committed_hd_bin = None
        self.last_unified_committed_score = None

    def _get_hd_bin_angles(self, target_device=None, target_dtype=None) -> torch.Tensor:
        return torch.arange(
            self.n_hd,
            dtype=target_dtype or self.dtype,
            device=target_device or self.device,
        ) * (2.0 * np.pi / float(self.n_hd))

    def _angle_to_hd_bin(self, angle_rad: float) -> int:
        angle = float(angle_rad) % (2.0 * np.pi)
        return int(round(angle / (2.0 * np.pi / float(self.n_hd)))) % self.n_hd

    def _planar_delta_to_clockwise_heading_deg(self, dx: float, dy: float) -> Optional[float]:
        if abs(dx) + abs(dy) <= 1e-12:
            return None
        standard_deg = np.rad2deg(np.arctan2(float(dy), float(dx)))
        return float((-standard_deg) % 360.0)

    def _log_coverage_progress(self) -> None:
        if self.step_count - self._last_coverage_log_step < 500:
            return
        self._last_coverage_log_step = int(self.step_count)
        coverage = self.current_coverage_percentage * 100.0
        target = self.target_coverage_percentage * 100.0
        min_time = 100.0
        if self.minimum_learning_duration_s > 0.0:
            min_time = min(100.0, 100.0 * self._sim_time() / self.minimum_learning_duration_s)
        print(
            f"[LEARN_LOCATIONS_COVERAGE] progress: "
            f"coverage={coverage:.1f}%/{target:.1f}% "
            f"cells={self.visited_cells}/{self.total_grid_cells} "
            f"goals={self.goal_visit_count}/{self.min_goal_visits} "
            f"min_time={min_time:.1f}%"
        )

    def check_goal_reached(self) -> None:
        at_goal = self._at_goal()
        if self.robot_mode in LEARNING_MODES:
            self._update_goal_contact(at_goal=at_goal)
            if self.robot_mode in COVERAGE_LEARNING_MODES:
                enough_coverage = self.current_coverage_percentage >= self.target_coverage_percentage
                enough_goals = self.goal_visit_count >= self.min_goal_visits
                enough_time = self._sim_time() >= self.minimum_learning_duration_s
                if enough_coverage and enough_goals and enough_time:
                    self.termination_reason = "learning_complete"
                    print(
                        f"[LEARN_LOCATIONS_COVERAGE] complete: "
                        f"coverage={self.current_coverage_percentage * 100:.1f}% "
                        f"goals={self.goal_visit_count}/{self.min_goal_visits}"
                    )
                    self.done = True
                else:
                    self._log_coverage_progress()
                    if self._sim_time() >= self.run_time_limit_s and self.step_count % max(1, 500) == 0:
                        print(
                            f"[LEARN_LOCATIONS_COVERAGE] time limit reached but criteria unmet; "
                            f"coverage={self.current_coverage_percentage * 100:.1f}%/"
                            f"{self.target_coverage_percentage * 100:.1f}% "
                            f"goals={self.goal_visit_count}/{self.min_goal_visits}"
                        )
                        self.termination_reason = "time_limit"
                        self.done = True
            elif self._sim_time() >= self.run_time_limit_s:
                self.termination_reason = "time_limit"
                self.done = True
            return

        if self.robot_mode in PLOTTING_MODES:
            if self.robot_mode == RobotMode.PLOTTING_COVERAGE_AUTO:
                if self.current_coverage_percentage >= self.target_coverage_percentage:
                    self.termination_reason = "coverage_complete"
                    self.done = True
            elif self._sim_time() >= self.run_time_limit_s:
                self.termination_reason = "time_limit"
                self.done = True
            return

        if self.robot_mode in EXPLOIT_MODES:
            if at_goal:
                print("[RUN] Goal reached")
                self.termination_reason = "goal_reached"
                self.done = True
                return
            if self.path_failure_ratio is not None and self.trial_direct_distance > 1e-6:
                ratio = self.current_trial_path_length / max(self.trial_direct_distance, 1e-6)
                if ratio >= self.path_failure_ratio:
                    print(f"[RUN] Exploit path limit reached: ratio={ratio:.2f}")
                    self.termination_reason = "path_limit"
                    self.done = True
                    return
            if (self._sim_time() - self.trial_start_time_s) >= self.run_time_limit_s:
                print("[RUN] Exploit timed out")
                self.termination_reason = "time_limit"
                self.done = True

    def _update_goal_contact(self, at_goal: Optional[bool] = None) -> None:
        if at_goal is None:
            at_goal = self._at_goal()
        if at_goal:
            self._current_goal_event.append(self.pcn.place_cell_activations.detach().cpu())
            now_s = self._sim_time()
            if (not self._was_at_goal) and (now_s - self.goal_last_count_time_s >= self.goal_visit_cooldown_s):
                self.goal_visit_count += 1
                self.goal_last_count_time_s = float(now_s)
                print(f"[LEARN_LOCATIONS] goal visit {self.goal_visit_count}/{self.min_goal_visits}")
            self._was_at_goal = True
            return

        if self._was_at_goal:
            x, y = self._current_planar_xy()
            dist = np.linalg.norm([x - self.goal_location[0], y - self.goal_location[1]])
            if dist > self.goal_radius + self.goal_exit_hysteresis:
                self._was_at_goal = False
                self._finalize_goal_event()

    def _finalize_goal_event(self) -> None:
        if self._current_goal_event:
            self.goal_events.append(self._current_goal_event)
            self._current_goal_event = []

    def _at_goal(self) -> bool:
        x, y = self._current_planar_xy()
        return np.linalg.norm([x - self.goal_location[0], y - self.goal_location[1]]) <= self.goal_radius

    def build_goal_reward(self) -> None:
        if self.rcn.build_goal_reward_from_events(self.pcn, self.goal_events, replace=True):
            print("[RCN] Built replay reward map from goal events")

    def rotate(self, direction: int, speed_factor: float) -> None:
        speed = self.max_speed * float(speed_factor)
        self.left_speed = speed * int(direction)
        self.right_speed = -speed * int(direction)
        self.move()

    def forward(self) -> None:
        self.left_speed = self.max_speed
        self.right_speed = self.max_speed
        self.move()
        self.sense()

    def compass_based_turn_to_heading(self, target_heading_deg: float) -> bool:
        max_single_turn = 10.0
        acceptable_error = 10.0
        max_attempts = 30

        def angle_difference(target, current):
            diff = (float(target) - float(current) + 180.0) % 360.0 - 180.0
            return diff

        if abs(angle_difference(target_heading_deg, self.current_heading_deg)) < 3.0:
            return True
        attempts = 0
        while abs(angle_difference(target_heading_deg, self.current_heading_deg)) > acceptable_error and attempts < max_attempts:
            attempts += 1
            remaining = angle_difference(target_heading_deg, self.current_heading_deg)
            turn_this_step = np.sign(remaining) * min(abs(remaining), max_single_turn)
            self.turn(float(np.deg2rad(turn_this_step)))
        return abs(angle_difference(target_heading_deg, self.current_heading_deg)) <= acceptable_error

    def _execute_movement(self, heading_deg: float) -> bool:
        self.last_executed_move_heading_deg = None
        self.last_executed_move_hd_bin = None
        self.last_executed_move_distance = 0.0

        if not self.compass_based_turn_to_heading(heading_deg):
            return False

        pre_xy = self._current_planar_xy()
        for _ in range(self._get_action_forward_steps()):
            self.sense()
            self.compute_pcn_activations(learn=False)
            self.record()
            self.forward()
            self.check_goal_reached()
            if self.done:
                return True
        post_xy = self._current_planar_xy()
        dx = post_xy[0] - pre_xy[0]
        dy = post_xy[1] - pre_xy[1]
        self.last_executed_move_distance = float(np.sqrt(dx * dx + dy * dy))
        actual_heading = self._planar_delta_to_clockwise_heading_deg(dx, dy)
        if actual_heading is not None:
            self.last_executed_move_heading_deg = float(actual_heading)
            self.last_executed_move_hd_bin = self._angle_to_hd_bin(np.deg2rad(-float(actual_heading)))
        return True

    def _get_nominal_forward_distance_per_step(self) -> float:
        dt_seconds = float(self.timestep) / 1000.0
        return float(max(1e-6, self.wheel_radius * self.max_speed * dt_seconds))

    def _get_action_forward_steps(self) -> int:
        if self.robot_mode not in EXPLOIT_MODES:
            return int(max(1, self.tau_w))
        target_distance = max(0.01, float(self.unified_exploit_step_distance))
        forward_steps = int(max(1, round(target_distance / self._get_nominal_forward_distance_per_step())))
        return int(min(forward_steps, max(1, 4 * int(self.tau_w))))

    def turn(self, angle: float) -> None:
        if self.left_position_sensor is None or self.right_position_sensor is None:
            self.rotate(1 if angle >= 0 else -1, 0.35)
            self.sense()
            self.stop()
            return
        self.stop()
        l_offset = self.left_position_sensor.getValue()
        r_offset = self.right_position_sensor.getValue()
        self.sense()
        direction = -1.0 if angle < 0.0 else 1.0
        self.left_motor.setVelocity(direction * self.max_speed / 2.0)
        self.right_motor.setVelocity(-direction * self.max_speed / 2.0)
        while True:
            l = self.left_position_sensor.getValue() - l_offset
            r = self.right_position_sensor.getValue() - r_offset
            dl = l * self.wheel_radius
            dr = r * self.wheel_radius
            orientation = direction * (dl - dr) / self.axle_length
            self.sense()
            if not orientation < direction * angle:
                break
        self.stop()
        self.sense()

    def stop(self) -> None:
        self.left_speed = 0.0
        self.right_speed = 0.0
        self.move()

    def move(self) -> None:
        self.left_motor.setPosition(float("inf"))
        self.right_motor.setPosition(float("inf"))
        self.left_motor.setVelocity(float(self.left_speed))
        self.right_motor.setVelocity(float(self.right_speed))

    def _write_records_csv(self, path: Path, records: List[Dict]) -> None:
        if not records:
            return
        keys = []
        for record in records:
            for key in record.keys():
                if key not in keys:
                    keys.append(key)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(records)

    def _json_safe(self, value):
        if isinstance(value, dict):
            return {str(k): self._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._json_safe(v) for v in value]
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, np.ndarray):
            return self._json_safe(value.tolist())
        if isinstance(value, torch.Tensor):
            return self._json_safe(value.detach().cpu().tolist())
        return value

    def _build_exploit_summary(self) -> Dict:
        if self.robot_mode in EXPLOIT_MODES and not self._finalized_current_trial:
            self._finalize_exploit_trial()
        trials = list(self.exploit_trials)
        success_count = sum(1 for t in trials if bool(t.get("success", False)))
        path_lengths = [float(t.get("path_length_m", 0.0)) for t in trials]
        ratios = [float(t.get("path_ratio_to_straight", 0.0)) for t in trials]
        return {
            "world": self.world_name,
            "mode": str(self.robot_mode.name),
            "goal_xy": [float(self.goal_location[0]), float(self.goal_location[1])],
            "trial_count": int(len(trials)),
            "success_count": int(success_count),
            "success_rate": float(success_count / max(1, len(trials))),
            "mean_path_length_m": float(np.mean(path_lengths)) if path_lengths else 0.0,
            "median_path_length_m": float(np.median(path_lengths)) if path_lengths else 0.0,
            "mean_path_ratio_to_straight": float(np.mean(ratios)) if ratios else 0.0,
            "trials": trials,
        }

    def save(self) -> None:
        with open(self.pcn_path, "wb") as f:
            pickle.dump(self.pcn, f)
        with open(self.rcn_path, "wb") as f:
            pickle.dump(self.rcn, f)
        hmap_loc = np.asarray(self.hmap_loc, dtype=np.float32)
        hmap_reward = np.asarray(self.hmap_reward, dtype=np.float32)
        hmap_prox = np.asarray(self.hmap_prox, dtype=np.float32)
        hmap_hdn = torch.stack(self.hmap_hdn) if self.hmap_hdn else torch.empty(0)
        with open(self.hmap_dir / "hmap_loc.pkl", "wb") as f:
            pickle.dump(hmap_loc, f)
        with open(self.hmap_dir / "hmap_reward.pkl", "wb") as f:
            pickle.dump(hmap_reward, f)
        with open(self.hmap_dir / "hmap_prox.pkl", "wb") as f:
            pickle.dump(hmap_prox, f)
        with open(self.hmap_dir / "hmap_hdn.pkl", "wb") as f:
            pickle.dump(hmap_hdn, f)
        hmap_pcn_scales = []
        for idx, history in enumerate(self.hmap_pcn_scales):
            if history:
                scale_history = torch.stack(history)
                hmap_pcn_scales.append(scale_history)
                with open(self.hmap_dir / f"hmap_pcn_scale_{self.scale_configs[idx]['scale_index']}.pkl", "wb") as f:
                    pickle.dump(scale_history.numpy(), f)
        hmap_pcn = torch.cat(hmap_pcn_scales, dim=1) if hmap_pcn_scales else torch.empty(0)
        for idx, history in enumerate(self.hmap_gcn_scales):
            if history:
                with open(self.hmap_dir / f"hmap_gcn_scale_{self.scale_configs[idx]['scale_index']}.pkl", "wb") as f:
                    pickle.dump(torch.stack(history).numpy(), f)
        exploit_summary = None
        if self.robot_mode in EXPLOIT_MODES:
            self._write_records_csv(self.hmap_dir / "exploit_trajectory.csv", self.trajectory_records)
            with open(self.hmap_dir / "exploit_decisions.pkl", "wb") as f:
                pickle.dump(self.exploit_decisions, f)
            exploit_summary = self._build_exploit_summary()
            with open(self.hmap_dir / "exploit_summary.json", "w", encoding="utf-8") as f:
                json.dump(self._json_safe(exploit_summary), f, indent=2)
        else:
            for name in ("trajectory.csv", "exploit_trajectory.csv", "exploit_decisions.pkl", "exploit_summary.json"):
                path = self.hmap_dir / name
                if path.exists():
                    path.unlink()
        with open(self.hmap_dir / "scale_diagnostics.pkl", "wb") as f:
            pickle.dump(self.scale_diagnostics, f)
        hmaps = {
            "hmap_loc": hmap_loc,
            "hmap_pcn": hmap_pcn,
            "hmap_hdn": hmap_hdn,
            "hmap_reward": hmap_reward,
            "hmap_prox": hmap_prox,
            "goal_location": list(self.goal_location),
            "goal_events": len(self.goal_events),
            "goal_visit_count": int(self.goal_visit_count),
            "coverage_percentage": float(self.current_coverage_percentage),
            "visited_cells": int(self.visited_cells),
            "total_grid_cells": int(self.total_grid_cells),
            "robot_mode": str(self.robot_mode.name),
        }
        if exploit_summary is not None:
            hmaps["exploit_summary"] = exploit_summary
        with open(self.hmap_dir / "unified_hmaps.pkl", "wb") as f:
            pickle.dump(hmaps, f)
        print(f"[SAVE] {self.network_dir}")
