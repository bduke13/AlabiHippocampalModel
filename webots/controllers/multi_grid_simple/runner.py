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
        reload_world,
        resume_simulation,
        robot_position,
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
        reload_world,
        resume_simulation,
        robot_position,
        set_robot_pose,
        step_simulation,
    )


@dataclass(frozen=True)
class TrialResult:
    trial_index: int
    run_id: str
    mode: str
    world_name: str


class BaseTrialRunner:
    """Minimal programmatic control layer for the isolated base model."""

    def __init__(self, driver: Optional[Driver] = None) -> None:
        self.driver = driver or Driver()

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
        set_robot_pose(self.driver.robot, start_loc)

    def initialize_trial(self, mode, **kwargs) -> Driver:
        self.driver.initialization(mode=mode, **kwargs)
        return self.driver

    def run_trial(self, mode, **kwargs) -> Driver:
        self.initialize_trial(mode, **kwargs)
        self.driver.run()
        return self.driver

    def run_trial_series(
        self,
        mode,
        *,
        num_trials: int,
        reload_world_between_trials: bool = False,
        pause_on_completion: bool = True,
        run_id_prefix: Optional[str] = None,
        **kwargs,
    ) -> list[TrialResult]:
        if num_trials < 1:
            raise ValueError("num_trials must be at least 1.")

        results: list[TrialResult] = []
        base_kwargs = dict(kwargs)

        for trial_index in range(1, num_trials + 1):
            trial_kwargs = dict(base_kwargs)
            if run_id_prefix and "run_id" not in trial_kwargs:
                trial_kwargs["run_id"] = f"{run_id_prefix}_trial_{trial_index:03d}"

            driver = self.run_trial(mode, **trial_kwargs)
            results.append(
                TrialResult(
                    trial_index=trial_index,
                    run_id=driver.run_id,
                    mode=driver.robot_mode.name,
                    world_name=driver.world_name,
                )
            )

            if trial_index < num_trials and reload_world_between_trials:
                self.reload_world()

        if pause_on_completion:
            self.pause()

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

    return runner.run_trial_series(
        mode,
        num_trials=int(execution_config.get("num_trials", 1)),
        reload_world_between_trials=bool(execution_config.get("reload_world_between_trials", False)),
        pause_on_completion=bool(execution_config.get("pause_on_completion", True)),
        run_id_prefix=execution_config.get("run_id_prefix"),
        **mode_params,
    )
