import os
import numpy as np

# Add root directory to python to be able to import
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # Moves two levels up
sys.path.append(str(PROJECT_ROOT))  # Add project root to sys.path

from core.robot.robot_mode import RobotMode
from webots.controllers.create3_3D_bvc.driver_3D_bvc import Driver

# Constant across trials
run_time_hours = 4
start_location = [1, 1]
goal_location = [-1, 1]
n_hd = 8
num_place_cells = 500
show_bvc_activation_plot = False
max_dist = 15

# Changes across trials
model_names = ["_model1", "_model2", "_model3", "_model4"]
experimental_num_bvc_per_dir = [120, 60, 40, 39]
experimental_phi_vert_preferred = [
    [np.radians(x) for x in [0]],
    [np.radians(x) for x in [0, 30]],
    [np.radians(x) for x in [0, 30, 60]],
    [np.radians(x) for x in [0, 30, 60, 90]],
]


bot = Driver()
# Create walls with angles for this trial
# for index, model_type in enumerate(models):

world_path = bot.getWorldPath()  # Get the full path to the world file


for index, model in enumerate(model_names):

    world_name = os.path.splitext(os.path.basename(world_path))[
        0
    ]  # Extract just the world name
    world_name += model

    phi_vert_preferred = experimental_phi_vert_preferred[index]
    num_bvc_per_dir = experimental_num_bvc_per_dir[index]
    sigma_rs = [0.5] * len(phi_vert_preferred)
    sigma_thetas = [np.radians(1)] * len(phi_vert_preferred)
    sigma_phis = [np.radians(1)] * len(phi_vert_preferred)
    scaling_factors = [1.0, 0.8, 0.6, 0.4][0 : len(phi_vert_preferred)]

    print(f"Model: {model}")
    print(f"vertical layers {phi_vert_preferred}")
    print(f"sigma_rs {sigma_rs}")
    print(f"sigma_thetas {sigma_thetas}")
    print(f"sigma_phis {sigma_phis}")
    print(f"scaling factors {scaling_factors}")

    bot.initialization(
        mode=RobotMode.LEARN_OJAS,
        randomize_start_location=True,
        run_time_hours=run_time_hours,
        phi_vert_preferred=phi_vert_preferred,
        sigma_rs=sigma_rs,
        sigma_thetas=sigma_thetas,
        sigma_phis=sigma_phis,
        scaling_factors=scaling_factors,
        n_hd=n_hd,
        num_bvc_per_dir=num_bvc_per_dir,
        max_dist=max_dist,
        show_bvc_activation_plot=show_bvc_activation_plot,
        show_save_dialogue_and_pause=False,
        world_name=world_name,
    )

    bot.run()
    bot.simulationReset()

bot.simulationSetMode(bot.SIMULATION_MODE_PAUSE)
