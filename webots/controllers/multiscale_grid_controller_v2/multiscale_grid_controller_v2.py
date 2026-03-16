"""Multiscale grid controller v2 entrypoint + orchestration.

Runtime-heavy timestep logic remains in multiscale_grid_driver_v2.Driver.
This module focuses on trial scheduling, mode dispatch, and auto-trial workflow.
"""

import gc
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from analysis.stats.stats_collector import stats_collector
from core.robot.robot_mode import RobotMode as LegacyRobotMode
from model_config import (
    DEFAULT_SELECTED_MODE,
    MODES_MAP,
    SUPPORTED_V2_MODE_KEYS,
    compile_scales,
    default_mode_params,
)
from model_config_validation_v2 import (
    validate_mode_kwargs,
    validate_mode_param_map,
    validate_scale_configs,
)
from multiscale_grid_driver_v2 import Driver
from robot_modes_v2 import RobotModeV2, to_legacy_mode
from webots.controllers.multiscale_grid_controller.path_planning import (
    calculate_optimal_paths,
    generate_spawn_locations,
    save_path_visualizations,
)


@dataclass
class TrialJob:
    trial_id: str
    start_loc: Optional[List[float]]
    target_goal: Optional[str] = None
    optimal_path_distance: Optional[float] = None


@dataclass(frozen=True)
class ExecutionConfig:
    type: str = "single"
    series_name: Optional[str] = None
    num_trials: int = 1
    resume_completed: bool = True

    @property
    def is_auto(self) -> bool:
        return self.type == "auto"


@dataclass
class RunContext:
    bot: Driver
    world_name: str
    scales: List[Dict[str, Any]]
    scale_name_str: str


def _resolve_execution_config(kwargs: Dict[str, Any]) -> ExecutionConfig:
    raw = dict(kwargs.get("execution_config", {}))
    has_legacy_auto_keys = "auto_trial_name" in kwargs or "num_auto_trials" in kwargs

    # Backward compatibility for older top-level auto settings.
    if "auto_trial_name" in kwargs and "series_name" not in raw:
        raw["series_name"] = kwargs["auto_trial_name"]
    if "num_auto_trials" in kwargs and "num_trials" not in raw:
        raw["num_trials"] = kwargs["num_auto_trials"]

    default_type = "auto" if has_legacy_auto_keys else "single"
    execution_type = str(raw.get("type", default_type)).lower()
    if execution_type == "auto":
        series_name = str(raw.get("series_name") or "default_trial")
        num_trials = int(raw.get("num_trials", 5))
        resume_completed = bool(raw.get("resume_completed", True))
        return ExecutionConfig(
            type="auto",
            series_name=series_name,
            num_trials=num_trials,
            resume_completed=resume_completed,
        )

    return ExecutionConfig()


def _build_run_context(kwargs: Dict[str, Any]) -> RunContext:
    scale_names = kwargs.get("scale_names", ["small", "medium", "large"])
    scales = compile_scales(scale_names, use_grid=kwargs.get("use_grid", True))
    validate_scale_configs(scales)

    bot = Driver()
    world_name = _world_name(bot)
    scale_name_str = "_".join(scale_names)
    return RunContext(bot=bot, world_name=world_name, scales=scales, scale_name_str=scale_name_str)


def _with_execution_metadata(kwargs: Dict[str, Any], execution: ExecutionConfig, trial_num: Optional[int]) -> Dict[str, Any]:
    trial_kwargs = dict(kwargs)
    if execution.is_auto and trial_num is not None:
        trial_kwargs["auto_trial_name"] = execution.series_name
        trial_kwargs["num_auto_trials"] = execution.num_trials
        trial_kwargs["current_auto_trial"] = trial_num
    return trial_kwargs


def _is_plotting_coverage_execution(kwargs: Dict[str, Any]) -> bool:
    stop_condition = str(kwargs.get("non_exploit_stop_condition", "time_or_coverage")).lower()
    has_coverage_config = all(
        kwargs.get(key) is not None for key in ("environment_size", "grid_size", "coverage_percentage")
    )
    return has_coverage_config and stop_condition in {"coverage", "time_or_coverage"}


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _world_name(bot: Driver) -> str:
    return os.path.basename(bot.getWorldPath()).replace(".wbt", "")


def _trial_dir(auto_trial_name: str, world_name: str, trial_num: int) -> str:
    return os.path.join(
        PROJECT_ROOT,
        "webots",
        "controllers",
        f"pkl_{auto_trial_name}",
        f"{world_name}_{trial_num}",
    )


def _learning_complete(auto_trial_name: str, world_name: str, trial_num: int) -> bool:
    return os.path.exists(os.path.join(_trial_dir(auto_trial_name, world_name, trial_num), "trial_completion_time.json"))


def _plot_complete(auto_trial_name: str, world_name: str, trial_num: int, suffix: str = "") -> bool:
    return os.path.exists(os.path.join(_trial_dir(auto_trial_name, world_name, trial_num), f"plotting_complete{suffix}.txt"))


def _exploit_stats_json_dir(auto_trial_name: str, world_name: str, trial_num: int, scale_name_str: str) -> str:
    return os.path.join(
        PROJECT_ROOT,
        "analysis",
        f"stats_{auto_trial_name}",
        f"{world_name}_{trial_num}",
        scale_name_str,
        "JSON",
    )


def _exploit_completion_marker(auto_trial_name: str, world_name: str, trial_num: int, scale_name_str: str) -> str:
    return os.path.join(
        PROJECT_ROOT,
        "analysis",
        f"stats_{auto_trial_name}",
        f"{world_name}_{trial_num}",
        scale_name_str,
        "exploit_auto_complete.json",
    )


def _exploit_complete(auto_trial_name: str, world_name: str, trial_num: int, scale_name_str: str) -> bool:
    return os.path.exists(_exploit_completion_marker(auto_trial_name, world_name, trial_num, scale_name_str))


def _write_exploit_completion_marker(
    auto_trial_name: str,
    world_name: str,
    trial_num: int,
    scale_name_str: str,
    expected_trials: int,
    completed_trials: int,
) -> None:
    marker_path = _exploit_completion_marker(auto_trial_name, world_name, trial_num, scale_name_str)
    payload = {
        "world_name": world_name,
        "learning_trial_num": trial_num,
        "expected_trials": expected_trials,
        "completed_trials": completed_trials,
        "complete": completed_trials >= expected_trials,
        "timestamp": datetime.now().isoformat(),
    }
    _ensure_dir(os.path.dirname(marker_path))
    with open(marker_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _highest_trial_idx(stats_folder: Optional[str], start_loc: List[float], target_goal: Optional[str]) -> int:
    if not stats_folder or not os.path.exists(stats_folder):
        return 0

    sx, sy = start_loc[0], start_loc[1]
    base = rf"trial_(\d+)_start_{re.escape(str(sx))}_{re.escape(str(sy))}"
    tail = rf"_goal_{re.escape(target_goal)}" if target_goal else ""
    pattern = re.compile(base + tail + r"\.json$")

    values: List[int] = []
    for fname in os.listdir(stats_folder):
        match = pattern.match(fname)
        if match:
            values.append(int(match.group(1)))
    return max(values) if values else 0


def save_trial_parameters(world_name: str, trial_id: str, mode: LegacyRobotMode, scales: List[Dict[str, Any]], **kwargs) -> None:
    auto_trial_name = kwargs.get("auto_trial_name")
    current_auto_trial = kwargs.get("current_auto_trial")

    if auto_trial_name and current_auto_trial is not None:
        save_folder = _trial_dir(auto_trial_name, world_name, current_auto_trial)
    else:
        save_folder = os.path.join(PROJECT_ROOT, "webots", "controllers", "multiscale_grid_controller_v2", "pkl", world_name)

    _ensure_dir(save_folder)

    payload = {
        "trial_id": trial_id,
        "timestamp": datetime.now().isoformat(),
        "mode": str(mode),
        "world_name": world_name,
        "scale_names": kwargs.get("scale_names", []),
        "scales": scales,
        "goal_config": kwargs.get("goal_config"),
        "trial_config": kwargs.get("trial_config"),
        "environment_size": kwargs.get("environment_size"),
        "grid_size": kwargs.get("grid_size"),
        "coverage_percentage": kwargs.get("coverage_percentage"),
        "non_exploit_stop_condition": kwargs.get("non_exploit_stop_condition", "time_or_coverage"),
        "goal_constraint_mode": kwargs.get("goal_constraint_mode", "none"),
        "coverage_free_space_only": kwargs.get("coverage_free_space_only", True),
        "runtime_step_mode": kwargs.get("runtime_step_mode", "legacy"),
        "ignore_goals": kwargs.get("ignore_goals", False),
        "track_runtime_metrics": kwargs.get("track_runtime_metrics", False),
        "auto_trial_name": auto_trial_name,
        "current_auto_trial": current_auto_trial,
    }

    with open(os.path.join(save_folder, f"{trial_id}_parameters.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _run_single_trial(
    bot: Driver,
    mode: LegacyRobotMode,
    world_name: str,
    scales: List[Dict[str, Any]],
    job: TrialJob,
    collector: Optional[stats_collector],
    **kwargs,
) -> None:
    trial_kwargs = dict(kwargs)
    trial_kwargs["start_loc"] = job.start_loc

    if job.target_goal and trial_kwargs.get("goal_config"):
        goal_cfg = dict(trial_kwargs["goal_config"])
        if goal_cfg.get("type") == "multi":
            goal_cfg["target_goal"] = job.target_goal
            trial_kwargs["goal_config"] = goal_cfg

    if job.optimal_path_distance is not None:
        trial_kwargs["optimal_path_distance"] = job.optimal_path_distance

    save_trial_parameters(world_name, job.trial_id, mode, scales, **trial_kwargs)
    rcn_learning_rates = [s["rcn_learning_rate"] for s in scales]

    bot.initialization(
        mode=mode,
        run_time_hours=trial_kwargs.get("run_time_hours", 2),
        randomize_start_loc=False if job.start_loc is not None else trial_kwargs.get("randomize_start_loc", False),
        start_loc=job.start_loc,
        enable_ojas=trial_kwargs.get("enable_ojas"),
        enable_stdp=trial_kwargs.get("enable_stdp"),
        scales=scales,
        rcn_learning_rates=rcn_learning_rates,
        stats_collector=collector,
        trial_id=job.trial_id,
        world_name=world_name,
        goal_location=trial_kwargs.get("goal_location"),
        goal_config=trial_kwargs.get("goal_config"),
        trial_config=trial_kwargs.get("trial_config"),
        max_dist=trial_kwargs.get("max_dist", 25),
        plot_bvc=trial_kwargs.get("plot_bvc", False),
        td_learning=trial_kwargs.get("td_learning", False),
        use_prox_mod=trial_kwargs.get("use_prox_mod", False),
        runtime_step_mode=trial_kwargs.get("runtime_step_mode", "legacy"),
        ignore_goals=trial_kwargs.get("ignore_goals", False),
        track_runtime_metrics=trial_kwargs.get("track_runtime_metrics", False),
        environment_size=trial_kwargs.get("environment_size"),
        grid_size=trial_kwargs.get("grid_size"),
        coverage_percentage=trial_kwargs.get("coverage_percentage"),
        min_goal_visits=trial_kwargs.get("min_goal_visits", 3),
        non_exploit_stop_condition=trial_kwargs.get("non_exploit_stop_condition", "time_or_coverage"),
        goal_constraint_mode=trial_kwargs.get("goal_constraint_mode", "none"),
        coverage_free_space_only=trial_kwargs.get("coverage_free_space_only", True),
        optimal_path_distance=trial_kwargs.get("optimal_path_distance"),
        path_failure_ratio=trial_kwargs.get("path_failure_ratio"),
        paths_folder=trial_kwargs.get("paths_folder"),
        hmaps_folder=trial_kwargs.get("hmaps_folder"),
        auto_trial_name=trial_kwargs.get("auto_trial_name"),
        num_auto_trials=trial_kwargs.get("num_auto_trials", 5),
        current_auto_trial=trial_kwargs.get("current_auto_trial", 1),
    )
    bot.trial_id = job.trial_id
    bot.run()


def _build_random_jobs(
    world_name: str,
    goals: List[Dict[str, Any]],
    trials_per_goal: int,
    min_spawn_distance: float,
    wall_clearance: float,
) -> Tuple[List[TrialJob], List[Dict[str, Any]], List[Dict[str, Any]]]:
    spawn_locations = generate_spawn_locations(
        world_name=world_name,
        goals=goals,
        trials_per_goal=trials_per_goal,
        min_spawn_distance=min_spawn_distance,
        wall_clearance=wall_clearance,
    )

    combos: List[Dict[str, Any]] = []
    for goal in goals:
        for i, spawn in enumerate(spawn_locations.get(goal["name"], [])):
            combos.append({"start": spawn, "goal": goal, "trial_number": i + 1, "goal_name": goal["name"]})

    path_results = calculate_optimal_paths(world_name, combos, wall_clearance)

    jobs: List[TrialJob] = []
    for combo, path_result in zip(combos, path_results):
        if not path_result["success"]:
            continue
        jobs.append(
            TrialJob(
                trial_id=f"trial_{combo['trial_number']}_goal_{combo['goal_name']}_random",
                start_loc=combo["start"],
                target_goal=combo["goal_name"],
                optimal_path_distance=path_result["distance"],
            )
        )

    return jobs, combos, path_results


def _run_standard_trials(run_context: RunContext, legacy_mode: LegacyRobotMode, **kwargs) -> None:
    trial_config = kwargs.get("trial_config", {"type": "simple", "count": 1})
    save_data = kwargs.get("save_data", False)

    stats_folder: Optional[str] = None
    if save_data:
        base = "stats_random" if legacy_mode == LegacyRobotMode.EXPLOIT_LOCATIONS_RANDOM else "stats"
        stats_folder = _ensure_dir(
            os.path.join(PROJECT_ROOT, "analysis", base, run_context.world_name, run_context.scale_name_str, "JSON")
        )
    jobs: List[TrialJob] = []
    trial_kwargs = dict(kwargs)

    if trial_config["type"] == "simple":
        start_locations = trial_config.get("start_locations", [kwargs.get("start_loc", [0, 0])])
        count = int(trial_config.get("count", 1))
        next_idx: Dict[Tuple[float, float], int] = {}
        for start in start_locations:
            key = (start[0], start[1])
            if save_data:
                next_idx[key] = _highest_trial_idx(stats_folder, start, None) + 1
            for i in range(1, count + 1):
                if save_data:
                    idx = next_idx[key]
                    next_idx[key] += 1
                else:
                    idx = i
                jobs.append(TrialJob(trial_id=f"trial_{idx}_start_{start[0]}_{start[1]}", start_loc=start))

    elif trial_config["type"] == "combinations":
        start_locations = trial_config.get("start_locations", [kwargs.get("start_loc", [0, 0])])
        target_goals = trial_config.get("target_goals", [None])
        count = int(trial_config.get("count", 1))
        goals_by_name = {g["name"]: g for g in kwargs.get("goal_config", {}).get("goals", [])}
        next_idx: Dict[Tuple[float, float, Optional[str]], int] = {}

        for start in start_locations:
            for target_goal in target_goals:
                if target_goal in goals_by_name and goals_by_name[target_goal]["location"] == start:
                    continue
                key = (start[0], start[1], target_goal)
                if save_data:
                    next_idx[key] = _highest_trial_idx(stats_folder, start, target_goal) + 1
                for i in range(1, count + 1):
                    if save_data:
                        idx = next_idx[key]
                        next_idx[key] += 1
                    else:
                        idx = i
                    suffix = f"_goal_{target_goal}" if target_goal else ""
                    jobs.append(
                        TrialJob(
                            trial_id=f"trial_{idx}_start_{start[0]}_{start[1]}{suffix}",
                            start_loc=start,
                            target_goal=target_goal,
                        )
                    )

    elif trial_config["type"] == "random_spawns":
        goals = kwargs.get("goal_config", {}).get("goals", [])
        jobs, combos, path_results = _build_random_jobs(
            world_name=run_context.world_name,
            goals=goals,
            trials_per_goal=int(trial_config.get("trials_per_goal", kwargs.get("trials_per_goal", 20))),
            min_spawn_distance=float(kwargs.get("min_spawn_distance", 6.0)),
            wall_clearance=float(kwargs.get("wall_clearance", 0.5)),
        )

        if save_data and kwargs.get("generate_path_plots", True):
            paths_folder = _ensure_dir(
                os.path.join(PROJECT_ROOT, "analysis", "stats_random", run_context.world_name, run_context.scale_name_str, "paths")
            )
            save_path_visualizations(
                world_name=run_context.world_name,
                combinations=combos,
                path_results=path_results,
                output_dir=paths_folder,
                wall_clearance=float(kwargs.get("wall_clearance", 0.5)),
                min_spawn_distance=float(kwargs.get("min_spawn_distance", 6.0)),
            )
            trial_kwargs["paths_folder"] = paths_folder

        if save_data:
            trial_kwargs["hmaps_folder"] = _ensure_dir(
                os.path.join(PROJECT_ROOT, "analysis", "stats_random", run_context.world_name, run_context.scale_name_str, "hmaps")
            )
    else:
        raise ValueError(f"Unknown trial type: {trial_config['type']}")

    for job in jobs:
        collector = stats_collector(output_dir=stats_folder) if stats_folder else None
        _run_single_trial(run_context.bot, legacy_mode, run_context.world_name, run_context.scales, job, collector, **trial_kwargs)

    run_context.bot.simulationSetMode(run_context.bot.SIMULATION_MODE_PAUSE)


def _run_auto_learning_trials(run_context: RunContext, legacy_mode: LegacyRobotMode, execution: ExecutionConfig, **kwargs) -> None:
    for trial_num in range(1, execution.num_trials + 1):
        if execution.resume_completed and _learning_complete(execution.series_name, run_context.world_name, trial_num):
            continue

        trial_kwargs = _with_execution_metadata(kwargs, execution, trial_num)
        job = TrialJob(trial_id=f"auto_trial_{trial_num}", start_loc=trial_kwargs.get("start_loc", [0, 0]))
        _run_single_trial(run_context.bot, legacy_mode, run_context.world_name, run_context.scales, job, None, **trial_kwargs)

    run_context.bot.simulationSetMode(run_context.bot.SIMULATION_MODE_PAUSE)


def _run_auto_exploit_trials(run_context: RunContext, legacy_mode: LegacyRobotMode, execution: ExecutionConfig, **kwargs) -> None:
    available = [
        trial_num
        for trial_num in range(1, execution.num_trials + 1)
        if _learning_complete(execution.series_name, run_context.world_name, trial_num)
    ]
    if not available:
        run_context.bot.simulationSetMode(run_context.bot.SIMULATION_MODE_PAUSE)
        return

    goals = kwargs.get("goal_config", {}).get("goals", [])
    jobs, combos, path_results = _build_random_jobs(
        world_name=run_context.world_name,
        goals=goals,
        trials_per_goal=int(kwargs.get("trials_per_goal", 20)),
        min_spawn_distance=float(kwargs.get("min_spawn_distance", 6.0)),
        wall_clearance=float(kwargs.get("wall_clearance", 0.5)),
    )

    for trial_num in available:
        if execution.resume_completed and _exploit_complete(
            execution.series_name, run_context.world_name, trial_num, run_context.scale_name_str
        ):
            continue

        base = os.path.join(
            PROJECT_ROOT,
            "analysis",
            f"stats_{execution.series_name}",
            f"{run_context.world_name}_{trial_num}",
            run_context.scale_name_str,
        )
        stats_folder = _ensure_dir(
            _exploit_stats_json_dir(execution.series_name, run_context.world_name, trial_num, run_context.scale_name_str)
        )
        hmaps_folder = _ensure_dir(os.path.join(base, "hmaps"))

        if kwargs.get("generate_path_plots", True):
            paths_folder = _ensure_dir(os.path.join(base, "paths"))
            save_path_visualizations(
                world_name=run_context.world_name,
                combinations=combos,
                path_results=path_results,
                output_dir=paths_folder,
                wall_clearance=float(kwargs.get("wall_clearance", 0.5)),
                min_spawn_distance=float(kwargs.get("min_spawn_distance", 6.0)),
            )
        else:
            paths_folder = None

        completed_count = 0
        for job in jobs:
            trial_stats_path = os.path.join(stats_folder, f"{job.trial_id}.json")
            if execution.resume_completed and os.path.exists(trial_stats_path):
                completed_count += 1
                continue

            trial_kwargs = _with_execution_metadata(kwargs, execution, trial_num)
            trial_kwargs["goal_config"] = {"type": "multi", "goals": goals, "target_goal": job.target_goal}
            trial_kwargs["paths_folder"] = paths_folder
            trial_kwargs["hmaps_folder"] = hmaps_folder
            collector = stats_collector(output_dir=stats_folder)
            _run_single_trial(run_context.bot, legacy_mode, run_context.world_name, run_context.scales, job, collector, **trial_kwargs)
            completed_count += 1

        _write_exploit_completion_marker(
            auto_trial_name=execution.series_name,
            world_name=run_context.world_name,
            trial_num=trial_num,
            scale_name_str=run_context.scale_name_str,
            expected_trials=len(jobs),
            completed_trials=completed_count,
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    run_context.bot.simulationSetMode(run_context.bot.SIMULATION_MODE_PAUSE)


def _run_auto_plotting_trials(run_context: RunContext, legacy_mode: LegacyRobotMode, execution: ExecutionConfig, **kwargs) -> None:
    coverage_mode = _is_plotting_coverage_execution(kwargs)
    suffix = "_coverage" if coverage_mode else ""

    available = [
        trial_num
        for trial_num in range(1, execution.num_trials + 1)
        if _learning_complete(execution.series_name, run_context.world_name, trial_num)
    ]
    if not available:
        run_context.bot.simulationSetMode(run_context.bot.SIMULATION_MODE_PAUSE)
        return

    for trial_num in available:
        if execution.resume_completed and _plot_complete(execution.series_name, run_context.world_name, trial_num, suffix):
            continue

        trial_kwargs = _with_execution_metadata(kwargs, execution, trial_num)
        trial_kwargs["enable_ojas"] = False
        trial_kwargs["enable_stdp"] = False
        trial_kwargs["td_learning"] = False
        trial_kwargs["use_prox_mod"] = False

        job = TrialJob(trial_id=f"auto_plotting_{trial_num}{suffix}", start_loc=trial_kwargs.get("start_loc", [0, 0]))
        _run_single_trial(run_context.bot, legacy_mode, run_context.world_name, run_context.scales, job, None, **trial_kwargs)

        marker_dir = _ensure_dir(_trial_dir(execution.series_name, run_context.world_name, trial_num))
        with open(os.path.join(marker_dir, f"plotting_complete{suffix}.txt"), "w", encoding="utf-8") as f:
            f.write(f"Plotting completed for trial {trial_num}\n")
            if coverage_mode:
                f.write(f"Coverage target: {trial_kwargs.get('coverage_percentage', 0.95)}\n")
                f.write(f"Environment size: {trial_kwargs.get('environment_size', [20.0, 20.0])}\n")
                f.write(f"Grid size: {trial_kwargs.get('grid_size', 0.5)}\n")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    run_context.bot.simulationSetMode(run_context.bot.SIMULATION_MODE_PAUSE)


def _run_auto_trials(legacy_mode: LegacyRobotMode, execution: ExecutionConfig, **kwargs) -> None:
    run_context = _build_run_context(kwargs)

    if legacy_mode == LegacyRobotMode.LEARN_LOCATIONS_COVERAGE:
        _run_auto_learning_trials(run_context, legacy_mode, execution, **kwargs)
        return
    if legacy_mode == LegacyRobotMode.EXPLOIT_LOCATIONS_RANDOM:
        _run_auto_exploit_trials(run_context, legacy_mode, execution, **kwargs)
        return
    if legacy_mode == LegacyRobotMode.PLOTTING:
        _run_auto_plotting_trials(run_context, legacy_mode, execution, **kwargs)
        return

    raise ValueError(f"Auto execution is not supported for legacy mode {legacy_mode}")


def run_bot(mode: RobotModeV2, **kwargs) -> None:
    supported_modes = set(MODES_MAP.values())
    if mode not in supported_modes:
        supported_names = [name for name, value in MODES_MAP.items() if value in supported_modes]
        raise ValueError(f"Unsupported RobotModeV2 for v2 controller: {mode}. Supported keys: {supported_names}")

    validate_mode_kwargs(kwargs)
    legacy_mode = to_legacy_mode(mode)
    execution = _resolve_execution_config(kwargs)

    if execution.is_auto:
        _run_auto_trials(legacy_mode, execution, **kwargs)
        return

    run_context = _build_run_context(kwargs)
    _run_standard_trials(run_context, legacy_mode=legacy_mode, **kwargs)


if __name__ == "__main__":
    selected_mode = DEFAULT_SELECTED_MODE
    mode_params = default_mode_params()
    validate_mode_param_map(mode_params)

    if selected_mode not in SUPPORTED_V2_MODE_KEYS:
        raise ValueError(f"Unsupported v2 mode '{selected_mode}'. Available: {SUPPORTED_V2_MODE_KEYS}")
    if selected_mode not in mode_params:
        raise ValueError(f"No parameters configured for mode '{selected_mode}'")

    run_bot(MODES_MAP[selected_mode], **mode_params[selected_mode])
