"""Initialization/setup helpers for Driver v2.

This module holds non-runtime setup logic for goals, trials, and coverage
tracking so the main driver can stay focused on simulation-time behavior.
"""

from typing import Any, Dict, Optional

import numpy as np

from core.robot.robot_mode import RobotMode
from core.robot.webots_worlds import WORLD_CONFIGS


def setup_goals(driver, goal_config: Optional[Dict[str, Any]], goal_location):
    """Setup unified goal system supporting single and multi-goal modes."""
    driver.goals = []
    driver.multi_goal_mode = False

    if goal_config is None:
        driver.goals.append(
            {
                "name": "default",
                "location": goal_location if goal_location else [-3, 3],
                "radius": driver.goal_r["explore"],
                "visited": False,
                "active": True,
            }
        )
        driver.goal_location = driver.goals[0]["location"]
        return

    if goal_config["type"] == "single":
        driver.goals.append(
            {
                "name": goal_config.get("name", "default"),
                "location": goal_config["location"],
                "radius": goal_config.get("radius", driver.goal_r["explore"]),
                "visited": False,
                "active": True,
            }
        )
        driver.goal_location = driver.goals[0]["location"]
        return

    if goal_config["type"] != "multi":
        raise ValueError(f"Unsupported goal config type: {goal_config['type']}")

    for goal in goal_config["goals"]:
        driver.goals.append(
            {
                "name": goal["name"],
                "location": goal["location"],
                "radius": goal["radius"],
                "visited": False,
                "active": goal_config.get("target_goal") == goal["name"] if "target_goal" in goal_config else False,
            }
        )

    if "target_goal" in goal_config:
        driver.active_goal_name = goal_config["target_goal"]

    driver.multi_goal_mode = True

    if driver.goal_constraint_mode in {"visits", "learning"}:
        driver.goal_visit_counts = {goal["name"]: 0 for goal in driver.goals}
        driver.goal_currently_in = {goal["name"]: False for goal in driver.goals}

    learning_modes = {RobotMode.LEARN_LOCATIONS_COVERAGE}
    if driver.goal_constraint_mode == "learning" and driver.robot_mode in learning_modes:
        driver.goal_place_cell_associations = {goal["name"]: [None] * len(driver.scales) for goal in driver.goals}
        driver.goal_association_step = {goal["name"]: [None] * len(driver.scales) for goal in driver.goals}
        driver.goal_place_cell_activations = {goal["name"]: [None] * len(driver.scales) for goal in driver.goals}
        driver.ACTIVATION_SIMILARITY_THRESHOLD = 0.20

    if driver.robot_mode == RobotMode.LEARN_LOCATIONS_COVERAGE:
        if driver.environment_size and driver.grid_size and driver.coverage_percentage:
            setup_coverage_tracking(driver, driver.environment_size, driver.grid_size, driver.coverage_percentage)

    print(f"[DRIVER] Multi-goal mode with {len(driver.goals)} goals:")
    for goal in driver.goals:
        status = "ACTIVE" if goal.get("active", False) else "inactive"
        print(f"  - {goal['name']}: {goal['location']} (radius: {goal['radius']}) [{status}]")


def setup_trials(driver, trial_config: Optional[Dict[str, Any]]):
    """Store trial configuration for reference."""
    if not trial_config:
        trial_config = {"type": "simple", "count": 1, "start_locations": [[0, 0]]}

    driver.trial_config = trial_config
    trial_count = trial_config.get("count") or trial_config.get("trials_per_goal", 1)
    print(f"[DRIVER] Trial config: {trial_config['type']} - {trial_count} trials")


def setup_coverage_tracking(driver, environment_size, grid_size, coverage_percentage):
    """Initialize coverage tracking with optional obstacle-aware free-space mask."""
    if environment_size is None or grid_size is None or coverage_percentage is None:
        raise ValueError("Coverage parameters must be provided for LEARN_LOCATIONS_COVERAGE mode")

    driver.environment_size = environment_size
    driver.grid_size = grid_size
    driver.target_coverage_percentage = coverage_percentage

    driver.grid_width = int(environment_size[0] / grid_size)
    driver.grid_height = int(environment_size[1] / grid_size)
    driver.total_grid_cells = driver.grid_width * driver.grid_height

    driver.coverage_grid = np.zeros((driver.grid_height, driver.grid_width), dtype=bool)
    driver.traversable_mask = np.ones((driver.grid_height, driver.grid_width), dtype=bool)

    world_cfg = WORLD_CONFIGS.get(driver.world_name)
    if driver.coverage_free_space_only and world_cfg:
        half_w = driver.environment_size[0] / 2.0
        half_h = driver.environment_size[1] / 2.0
        x_centers = (np.arange(driver.grid_width) + 0.5) * driver.grid_size - half_w
        z_centers = (np.arange(driver.grid_height) + 0.5) * driver.grid_size - half_h

        for obstacle in world_cfg.get("obstacles", []):
            if obstacle.get("type") != "rectangle":
                continue
            (min_x, min_z), (max_x, max_z) = obstacle["bounds"]
            x_hits = (x_centers >= min_x) & (x_centers <= max_x)
            z_hits = (z_centers >= min_z) & (z_centers <= max_z)
            driver.traversable_mask[np.ix_(z_hits, x_hits)] = False

    driver.total_coverable_cells = int(np.sum(driver.traversable_mask))
    if driver.total_coverable_cells <= 0:
        driver.traversable_mask[:, :] = True
        driver.total_coverable_cells = driver.total_grid_cells

    driver.visited_cells = 0
    driver.current_coverage_percentage = 0.0

    print(f"[COVERAGE] Environment: {environment_size[0]}x{environment_size[1]}m")
    print(f"[COVERAGE] Grid size: {grid_size}m")
    print(
        f"[COVERAGE] Grid dimensions: {driver.grid_width}x{driver.grid_height} "
        f"({driver.total_grid_cells} total cells)"
    )
    if driver.coverage_free_space_only:
        blocked = driver.total_grid_cells - driver.total_coverable_cells
        print(
            f"[COVERAGE] Obstacle-aware coverage enabled. "
            f"Coverable cells: {driver.total_coverable_cells}, blocked cells: {blocked}"
        )
    print(f"[COVERAGE] Target coverage: {coverage_percentage * 100:.1f}%")
