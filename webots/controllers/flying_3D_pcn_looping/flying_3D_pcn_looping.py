import os
import numpy as np

# Add root directory to python to be able to import
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # Moves two levels up
sys.path.append(str(PROJECT_ROOT))  # Add project root to sys.path

from core.robot.robot_mode import RobotMode
from webots.controllers.flying_3D_pcn.driver_3D_pcn import DriverFlying


run_time_hours = 4

phi_vert_preferred = [np.radians(x) for x in [-90, -60, -30, 0, 30, 60, 90]]
sigma_rs = [0.4] * len(phi_vert_preferred)
sigma_thetas = [np.radians(2)] * len(phi_vert_preferred)
sigma_phis = [np.radians(2)] * len(phi_vert_preferred)
scaling_factors = [1] * len(phi_vert_preferred)
visual_bvc = False
visual_pcn = False
n_hd_bvc = 8
n_hd_hdn = 20

num_place_cells = 1000
num_bvc_per_dir = 50
max_dist = 8

start_location = [1, 1]
goal_location = [-1, 1]

bot = DriverFlying()
# Create walls with angles for this trial
# for index, model_type in enumerate(models):

world_path = bot.getWorldPath()  # Get the full path to the world file


tau_denoms = [1000, 500, 250]
gamma_pps = [0.3, 0.4, 0.5]  # np.arange(0.1, 1.0, 0.1)
gamma_pbs = [0.1, 0.2, 0.3, 0.4]  # np.arange(0.1, 1.0, 0.1)

for tau_denom in tau_denoms:
    for gamma_pp in gamma_pps:
        for gamma_pb in gamma_pbs:

            world_name = os.path.splitext(os.path.basename(world_path))[
                0
            ]  # Extract just the world name

            world_name += f"_pp{gamma_pp}_pb{gamma_pb}_denom{tau_denom}"

            bot.initialization(
                mode=RobotMode.LEARN_OJAS,
                randomize_start_location=True,
                run_time_hours=run_time_hours,
                phi_vert_preferred=phi_vert_preferred,
                sigma_rs=sigma_rs,
                sigma_thetas=sigma_thetas,
                sigma_phis=sigma_phis,
                scaling_factors=scaling_factors,
                visual_bvc=visual_bvc,
                visual_pcn=visual_pcn,
                n_hd_bvc=n_hd_bvc,
                n_hd_hdn=n_hd_hdn,
                max_dist=max_dist,
                num_place_cells=num_place_cells,
                num_bvc_per_dir=num_bvc_per_dir,
                world_name=world_name,
                gamma_pp=gamma_pp,
                gamma_pb=gamma_pb,
                tau_denom=tau_denom,
                show_save_dialogue_and_pause=False,
            )

            bot.run()
            bot.simulationReset()

bot.simulationSetMode(bot.SIMULATION_MODE_PAUSE)
