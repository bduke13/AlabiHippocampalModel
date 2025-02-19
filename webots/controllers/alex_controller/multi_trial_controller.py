""""multi_trial_controller for running multiple trials in succession."""
import sys
from pathlib import Path

# Remove popups.
# (No tkinter imports are needed.)

# Set project root and add it to sys.path so that modules can be imported.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from core.robot.robot_mode import RobotMode
from alex_driver import AlexDriver

# Define a list of trial parameter dictionaries.
# Note: We add "disable_save_popup": True so that the driver's save() method won’t show any popup.
# Generate a list of trials.
# For sigma_ang from 0.01 to 0.1 (in steps of 0.01) and sigma_d from 0.1 to 0.9 (in steps of 0.1)
trial_list = []
for i in range(1, 11):
    sigma_ang = round(i * 0.01, 2)  # 0.01, 0.02, ..., 0.10
    for j in range(1, 10):
        sigma_d = round(j * 0.1, 2)   # 0.1, 0.2, ..., 0.9
        trial_name = f"sigma_ang_{sigma_ang}_sigma_d_{sigma_d}"
        trial = {
            "environment_label": "TestEnvironment",
            "max_runtime_hours": 10,           # 10 hours per trial
            "randomize_start_loc": False,     # random start is off
            "start_loc": [4, -4],
            "goal_location": [-3, 3],         # default goal location
            "mode": RobotMode.LEARN_OJAS,      # using LEARN_OJAS mode (adjust as needed)
            "movement_method": "default",
            "sigma_ang": sigma_ang,
            "sigma_d": sigma_d,
            "max_dist": 20,
            "num_bvc_per_dir": 100,
            "num_place_cells": 1000,
            "n_hd": 8,
            "save_trial_data": True,
            "trial_name": trial_name,
            "disable_save_popup": True      # disable popup in multi-trial mode
        }
        trial_list.append(trial)

def run_trials(trial_list):
    # default_params provides a baseline configuration to create the initial AlexDriver instance.
    # Since only one Robot instance is allowed per controller process, we create a single driver.
    # Then, before each trial, we call bot.trial_setup(trial_params) to update its parameters and reset its state.
    default_params = {
        "environment_label": "default",
        "max_runtime_hours": 0.5,
        "randomize_start_loc": True,
        "start_loc": [4, -4],
        "goal_location": [-3, 3],
        "mode": trial_list[0].get("mode"),
        "movement_method": "default",
        "sigma_ang": 0.01,
        "sigma_d": 0.5,
        "max_dist": 15,
        "num_bvc_per_dir": 50,
        "num_place_cells": 500,
        "n_hd": 8,
        "save_trial_data": True,
        "trial_name": "default_trial",
        "run_multiple_trials": False,  # Always false internally.
        "disable_save_popup": True
    }
    bot = AlexDriver(**default_params)
    
    for trial_params in trial_list:
        print(f"Starting trial with parameters: {trial_params}")
        bot.trial_setup(trial_params)
        # Record the trial's start time so that runtime is measured relative to it.
        bot.trial_start_time = bot.getTime()
        bot.run_trial()
        # Resume simulation mode between trials.
        bot.simulationSetMode(bot.SIMULATION_MODE_FAST)
        print("Trial complete. Proceeding to next trial...\n")

    # After all trials are complete, pause the simulation.
    print("All trials have been completed. Pausing simulation.")
    bot.simulationSetMode(bot.SIMULATION_MODE_PAUSE)

if __name__ == "__main__":
    run_trials(trial_list)
    print("All trials have been completed.")

