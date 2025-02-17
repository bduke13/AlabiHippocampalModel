#!/usr/bin/env python
"""
File: webots/controllers/alex_controller/alex_controller.py

Main entry point for the alex_controller.
This controller supports both single-run and grid search modes.
- In single-run mode, it runs the simulation for one trial, prompts the user to save data,
  and then stops (so that the user can update parameters and restart).
- In grid search mode, it delegates execution to the grid_search module, which runs multiple trials automatically.
"""

import sys
from pathlib import Path

# Ensure the project root is on sys.path.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from core.robot.robot_mode import RobotMode
from core.robot.movement_method import MovementMethod
from alex_driver import AlexDriver
from trial_manager import TrialManager

# Optional: import grid_search if grid search mode is enabled.
try:
    from grid_search import grid_search
except ImportError:
    grid_search = None

def main():
    # Configuration parameters for the run.
    # Add a "grid_search" flag. Set it to True to run multiple trials automatically.
    config = {
        "mode": RobotMode.LEARN_OJAS,             # Options: LEARN_ALL, LEARN_OJAS, DMTP, EXPLOIT, etc.
        "randomize_start_loc": True,
        "run_time_hours": 2,
        "start_loc": [4, -4],
        "explore_mthd": MovementMethod.RANDOM_WALK,  # Currently defaulting to random walk.
        "environment_label": "15x15",
        "scale_mode": "S",  # "S" for small-only, "L" for large-only, "MS" for multi-scale combined.
        "small_scale_params": {
             "bvc_sigma_ang": 90.0,
             "bvc_sigma_d": 0.7,
             "bvc_max_dist": 30.0,
             "num_place_cells": 1000,
             "n_hd": 8,
        },
        "large_scale_params": {
             "bvc_sigma_ang": 90.0,
             "bvc_sigma_d": 1.5,
             "bvc_max_dist": 30.0,
             "num_place_cells": 500,
             "n_hd": 8,
        },
        "save_trials": True,      # If True, trial data will be saved to a new trial folder.
        "grid_search": False      # Set True to enable grid search (multiple runs automatically).
    }

    if config.get("grid_search", False):
        # If grid search mode is enabled, delegate to the grid_search module.
        if grid_search is not None:
            print("Grid search mode enabled. Delegating execution to grid_search module.")
            grid_search()
        else:
            print("Grid search module not found. Exiting.")
        return

    # Otherwise, run a single trial.
    # Remove keys not expected by the driver.
    save_trials_flag = config.pop("save_trials", True)
    config.pop("grid_search", None)

    # Instantiate the TrialManager.
    trial_mgr = TrialManager(save_trials=save_trials_flag)
    # Optionally, allow the user to specify a trial name (None auto-assigns).
    trial_name = None  
    trial_folder = trial_mgr.start_trial(config, trial_name=trial_name)
    print(f"Trial folder created at: {trial_folder}")

    # Instantiate and initialize the driver.
    driver = AlexDriver()
    driver.initialization(**config)
    
    # Run the simulation (until the time limit is reached or run is otherwise ended).
    driver.run()

    # After the run, gather data from the driver.
    trial_data = {
        "hmap_loc": driver.hmap_loc[:driver.step_count],
        "hmap_pcn": driver.hmap_pcn[:driver.step_count].cpu().numpy(),
        "hmap_bvc": driver.hmap_bvc[:driver.step_count].cpu().numpy(),
        "hmap_hdn": driver.hmap_hdn[:driver.step_count],
        "hmap_g": driver.hmap_g[:driver.step_count],
        "visitation_map": driver.visitation_map,
        "visitation_map_metrics": driver.visitation_map_metrics,
        "weight_change_history": driver.weight_change_history,
        "metrics_over_time": driver.metrics_over_time,
    }
    
    # Save data files using TrialManager.
    for key, data in trial_data.items():
        trial_mgr.save_data(data, f"{key}.pkl")
        trial_mgr.save_data_global(data, f"{key}.pkl")
    
    # Log a trial summary.
    trial_summary = {
        "mode": str(config["mode"]),
        "explore_mthd": str(config["explore_mthd"]),
        "run_time_hours": config["run_time_hours"],
        "steps": driver.step_count,
        # Additional summary metrics can be added here.
    }
    trial_mgr.log_trial_summary(trial_summary)
    print("Trial summary logged.")

if __name__ == "__main__":
    main()
