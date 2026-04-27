import os
import random
from pathlib import Path

import numpy as np


def current_world_name(supervisor) -> str:
    configured_name = os.environ.get("MULTI_GRID_SIMPLE_CANONICAL_WORLD_NAME")
    if configured_name:
        return Path(configured_name).stem
    return Path(supervisor.getWorldPath()).stem


def robot_position(robot_node) -> list[float]:
    return list(robot_node.getField("translation").getSFVec3f())


def robot_rotation(robot_node) -> list[float]:
    return list(robot_node.getField("rotation").getSFRotation())


def set_robot_pose(
    robot_node,
    start_loc,
    *,
    rotation=None,
    y: float = 0.0,
    reset_physics: bool = True,
) -> None:
    robot_node.getField("translation").setSFVec3f([start_loc[0], y, start_loc[1]])
    if rotation is not None:
        robot_node.getField("rotation").setSFRotation(list(rotation))
    if reset_physics:
        robot_node.resetPhysics()


def randomize_robot_pose(
    robot_node,
    *,
    goal_location,
    bounds: tuple[float, float] = (-2.3, 2.3),
    min_goal_distance: float = 1.0,
    rotation=None,
    y: float = 0.0,
) -> list[float]:
    while True:
        candidate = [
            random.uniform(bounds[0], bounds[1]),
            y,
            random.uniform(bounds[0], bounds[1]),
        ]
        dist_to_goal = np.sqrt(
            (candidate[0] - goal_location[0]) ** 2
            + (candidate[2] - goal_location[1]) ** 2
        )
        if dist_to_goal >= min_goal_distance:
            robot_node.getField("translation").setSFVec3f(candidate)
            if rotation is not None:
                robot_node.getField("rotation").setSFRotation(list(rotation))
            robot_node.resetPhysics()
            return candidate


def pause_simulation(supervisor) -> None:
    supervisor.simulationSetMode(supervisor.SIMULATION_MODE_PAUSE)


def resume_simulation(supervisor) -> None:
    supervisor.simulationSetMode(supervisor.SIMULATION_MODE_REAL_TIME)


def set_simulation_mode(supervisor, mode: str) -> None:
    mode_key = str(mode).lower()
    mode_map = {
        "pause": supervisor.SIMULATION_MODE_PAUSE,
        "realtime": supervisor.SIMULATION_MODE_REAL_TIME,
        "fast": supervisor.SIMULATION_MODE_FAST,
    }
    if mode_key not in mode_map:
        raise ValueError(f"Unsupported simulation mode '{mode}'.")
    supervisor.simulationSetMode(mode_map[mode_key])


def step_simulation(supervisor, timestep: int) -> int:
    return supervisor.step(timestep)


def reload_world(supervisor) -> None:
    supervisor.worldReload()


def reset_simulation(supervisor) -> None:
    supervisor.simulationReset()


def reset_physics(supervisor) -> None:
    supervisor.simulationResetPhysics()


def quit_webots(supervisor, status: int = 0) -> None:
    supervisor.simulationQuit(status)


def export_image(supervisor, output_path, quality: int = 100) -> None:
    supervisor.exportImage(str(output_path), quality)
