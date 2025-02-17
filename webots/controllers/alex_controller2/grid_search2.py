#!/usr/bin/env python
"""
File: webots/controllers/alex_controller/grid_search.py

This module automates running multiple trials without restarting the simulation.
It reuses the same AlexDriver instance across trials:
  - For each configuration, it updates the driver parameters via update_config(),
    resets the internal state using reset_run(), and then runs the trial.
  - After each trial, it saves the trial data via TrialManager.
"""

import sys
from pathlib import Path
import time
sys.path.append(str(Path(__file__).resolve().parents[3]))

from alex_driver import AlexDriver
from trial_manager import TrialManager

def run_trial(driver: AlexDriver, config: dict, trial_name: str):
    """
    Runs a single trial using the given configuration and trial name.
    Updates the driver configuration, resets the run, and then runs the simulation.
    After the trial, data are saved via the TrialManager.
    """
    print(f"\n--- Starting trial: {trial_name} ---")
    # Instantiate a TrialManager for this trial.
    trial_mgr = TrialManager(save_trials=True)
    trial_folder = trial_mgr.start_trial(config, trial_name=trial_name)
    
    # Update driver configuration and reset the run.
    driver.update_config(config)
    driver.reset_run()
    
    # Run the trial.
    driver.run()
    
    # Gather data from the driver.
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
    
    # Save data files into the trial folder.
    for key, data in trial_data.items():
        trial_mgr.save_data(data, f"{key}.pkl")
    
    # Log a trial summary.
    trial_summary = {
        "trial_name": trial_name,
        "run_time_hours": config["run_time_hours"],
        "steps": driver.step_count,
    }
    trial_mgr.log_trial_summary(trial_summary)
    print(f"Trial '{trial_name}' completed.\n")
    
    # (Optionally, you may wish to wait a brief moment before the next trial.)
    time.sleep(1)

def grid_search():
    # Define a list of configurations. For example, vary run_time_hours.
    configs = []
    for run_time in [1, 2]:
        config = {
            "mode": "LEARN_ALL",  # Using string for simplicity.
            "randomize_start_loc": True,
            "run_time_hours": run_time,
            "start_loc": [4, -4],
            "explore_mthd": "RANDOM_WALK",  # Using string for now.
            "environment_label": "15x15",
            "scale_mode": "S",
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
        }
        configs.append(config)
    
    # Instantiate a single AlexDriver instance (only one is allowed).
    driver = AlexDriver()
    # For each configuration, run a trial.
    for i, config in enumerate(configs, start=1):
        trial_name = f"trial_{i}"
        run_trial(driver, config, trial_name)

if __name__ == "__main__":
    grid_search()
