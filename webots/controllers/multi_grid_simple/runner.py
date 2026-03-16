from dataclasses import dataclass
from typing import Optional

try:
    from config import (
        MODES_MAP,
        get_automation_config,
        get_execution_config,
        get_mode_params,
        get_selected_mode,
    )
    from driver import Driver
    from webots_control import (
        current_world_name,
        pause_simulation,
        reset_physics,
        reset_simulation,
        reload_world,
        resume_simulation,
        robot_position,
        robot_rotation,
        set_robot_pose,
        step_simulation,
    )
except ImportError:  # Support package import from repo root.
    from .config import (
        MODES_MAP,
        get_automation_config,
        get_execution_config,
        get_mode_params,
        get_selected_mode,
    )
    from .driver import Driver
    from .webots_control import (
        current_world_name,
        pause_simulation,
        reset_physics,
        reset_simulation,
        reload_world,
        resume_simulation,
        robot_position,
        robot_rotation,
        set_robot_pose,
        step_simulation,
    )


@dataclass(frozen=True)
class TrialResult:
    trial_index: int
    run_id: str
    mode: str
    world_name: str
    completion_reason: Optional[str]
    path_length: float
    simulation_time_seconds: float
    trial_elapsed_seconds: float


class BaseTrialRunner:
    """Minimal programmatic control layer for the isolated base model."""

    def __init__(self, driver: Optional[Driver] = None) -> None:
        self.driver = driver or Driver()
        self.default_start_rotation: Optional[list[float]] = None

    def current_world_name(self) -> str:
        return current_world_name(self.driver)

    def current_position(self) -> list[float]:
        if not hasattr(self.driver, "robot"):
            raise RuntimeError("Driver must be initialized before reading robot position.")
        return robot_position(self.driver.robot)

    def pause(self) -> None:
        pause_simulation(self.driver)

    def resume(self) -> None:
        resume_simulation(self.driver)

    def reload_world(self) -> None:
        reload_world(self.driver)

    def step(self, timestep: Optional[int] = None) -> int:
        if timestep is None:
            if not hasattr(self.driver, "timestep"):
                raise RuntimeError("Driver must be initialized before stepping with default timestep.")
            timestep = self.driver.timestep
        return step_simulation(self.driver, timestep)

    def set_start_pose(self, start_loc) -> None:
        if not hasattr(self.driver, "robot"):
            raise RuntimeError("Driver must be initialized before setting robot pose.")
        rotation = self.default_start_rotation
        set_robot_pose(self.driver.robot, start_loc, rotation=rotation)

    def _capture_default_start_rotation(self) -> list[float]:
        if self.default_start_rotation is None:
            robot = self.driver.getFromDef("agent")
            if robot is None:
                raise RuntimeError("Could not resolve agent node for start rotation capture.")
            self.default_start_rotation = robot_rotation(robot)
        return list(self.default_start_rotation)

    def initialize_session(self, mode, **kwargs) -> Driver:
        return self.initialize_trial(mode, **kwargs)

    def initialize_trial(self, mode, **kwargs) -> Driver:
        if "start_rotation" not in kwargs or kwargs["start_rotation"] is None:
            kwargs["start_rotation"] = self._capture_default_start_rotation()
        self.driver.initialization(mode=mode, **kwargs)
        return self.driver

    def run_trial(self, mode, **kwargs) -> Driver:
        self.initialize_trial(mode, **kwargs)
        self.driver.run()
        return self.driver

    def reset_for_next_trial(
        self,
        *,
        start_loc=None,
        start_rotation=None,
        randomize_start_loc: Optional[bool] = None,
        strategy: str = "pose_reset",
        settle_steps: int = 1,
    ) -> str:
        if not hasattr(self.driver, "robot"):
            raise RuntimeError("Driver must be initialized before resetting a trial.")

        if strategy == "pose_reset":
            if randomize_start_loc:
                return "pose_reset_skipped_randomized"
            target_start_loc = (
                start_loc
                if start_loc is not None
                else getattr(self.driver, "configured_start_loc", None)
            )
            if target_start_loc is None:
                raise ValueError("start_loc is required for pose_reset when randomize_start_loc is False.")
            target_rotation = (
                start_rotation
                if start_rotation is not None
                else getattr(self.driver, "configured_start_rotation", None)
                or self._capture_default_start_rotation()
            )
            self.driver.stop()
            set_robot_pose(
                self.driver.robot,
                target_start_loc,
                rotation=target_rotation,
                reset_physics=False,
            )
            reset_physics(self.driver)
            for _ in range(max(settle_steps, 0)):
                self.step()
            return "pose_reset"

        if strategy == "simulation_reset":
            reset_simulation(self.driver)
            return "simulation_reset"

        if strategy == "world_reload":
            reload_world(self.driver)
            return "world_reload"

        raise ValueError(f"Unsupported reset strategy '{strategy}'.")

    def run_trial_series(
        self,
        mode,
        *,
        num_trials: int,
        reload_world_between_trials: bool = False,
        run_id_prefix: Optional[str] = None,
        **kwargs,
    ) -> list[TrialResult]:
        if num_trials < 1:
            raise ValueError("num_trials must be at least 1.")
        if reload_world_between_trials:
            raise NotImplementedError(
                "reload_world_between_trials is not supported inside a single Webots session yet."
            )

        results: list[TrialResult] = []
        base_kwargs = dict(kwargs)

        for trial_index in range(1, num_trials + 1):
            trial_kwargs = dict(base_kwargs)
            is_final_trial = trial_index == num_trials

            if run_id_prefix and "run_id" not in trial_kwargs:
                trial_kwargs["run_id"] = f"{run_id_prefix}_trial_{trial_index:03d}"

            if not is_final_trial:
                trial_kwargs["pause_on_completion"] = False
                trial_kwargs["quit_on_completion"] = False
                trial_kwargs["export_image_on_completion"] = False

            if trial_index > 1:
                self.reset_for_next_trial(
                    start_loc=trial_kwargs.get("start_loc"),
                    start_rotation=trial_kwargs.get("start_rotation"),
                    randomize_start_loc=trial_kwargs.get("randomize_start_loc"),
                )

            driver = self.run_trial(mode, **trial_kwargs)
            results.append(
                TrialResult(
                    trial_index=trial_index,
                    run_id=driver.run_id,
                    mode=driver.robot_mode.name,
                    world_name=driver.world_name,
                    completion_reason=driver.trial_completion_reason,
                    path_length=float(driver.compute_path_length()),
                    simulation_time_seconds=float(driver.getTime()),
                    trial_elapsed_seconds=float(driver.elapsed_trial_time_seconds()),
                )
            )

        return results


def run_selected_trial() -> Driver:
    selected_mode = get_selected_mode()
    if selected_mode not in MODES_MAP:
        raise ValueError(f"Unsupported mode '{selected_mode}'. Supported modes: {list(MODES_MAP)}")

    runner = BaseTrialRunner()
    mode_params = get_mode_params(selected_mode)
    mode_params.update(get_automation_config())
    return runner.run_trial(MODES_MAP[selected_mode], **mode_params)


def run_selected_execution():
    selected_mode = get_selected_mode()
    if selected_mode not in MODES_MAP:
        raise ValueError(f"Unsupported mode '{selected_mode}'. Supported modes: {list(MODES_MAP)}")

    mode = MODES_MAP[selected_mode]
    mode_params = get_mode_params(selected_mode)
    mode_params.update(get_automation_config())
    execution_config = get_execution_config()
    runner = BaseTrialRunner()

    if execution_config.get("type", "single") == "single" or execution_config.get("num_trials", 1) == 1:
        return runner.run_trial(mode, **mode_params)

    mode_params.setdefault(
        "pause_on_completion",
        bool(execution_config.get("pause_on_completion", True)),
    )

    return runner.run_trial_series(
        mode,
        num_trials=int(execution_config.get("num_trials", 1)),
        reload_world_between_trials=bool(execution_config.get("reload_world_between_trials", False)),
        run_id_prefix=execution_config.get("run_id_prefix"),
        **mode_params,
    )
