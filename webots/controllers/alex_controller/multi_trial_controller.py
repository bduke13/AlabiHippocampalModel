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
trial_list = [
    {
        "environment_label": "Env_A",
        "max_runtime_hours": 0.5,  # Trial duration: 0.5 hours (relative)
        "randomize_start_loc": True,
        "start_loc": [4, -4],
        "goal_location": [-3, 3],
        "mode": RobotMode.LEARN_OJAS,
        "movement_method": "default",
        "sigma_ang": 0.01,
        "sigma_d": 0.5,
        "max_dist": 15,
        "num_bvc_per_dir": 50,
        "num_place_cells": 500,
        "n_hd": 8,
        "save_trial_data": True,
        "trial_name": "Env_A_trial",
        "disable_save_popup": True
    },
    {
        "environment_label": "Env_B",
        "max_runtime_hours": 1.0,  # Trial duration: 1 hour (relative)
        "randomize_start_loc": True,
        "start_loc": [4, -4],
        "goal_location": [-3, 3],
        "mode": RobotMode.LEARN_HEBB,
        "movement_method": "default",
        "sigma_ang": 0.02,
        "sigma_d": 0.6,
        "max_dist": 15,
        "num_bvc_per_dir": 60,
        "num_place_cells": 600,
        "n_hd": 8,
        "save_trial_data": True,
        "trial_name": "Env_B_trial",
        "disable_save_popup": True
    }
    # Add additional trial configurations as needed.
]

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

