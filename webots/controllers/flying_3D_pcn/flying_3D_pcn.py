"""my_controller_iCreate controller."""

from driver_3D_pcn import DriverFlying

# Add root directory to python to be able to import
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # Moves two levels up
sys.path.append(str(PROJECT_ROOT))  # Add project root to sys.path

from core.robot.robot_mode import RobotMode

run_time_hours = 2

preferred_va = [-1.2, -0.6, 0.0, 0.6, 1.2]
sigma_rs = [0.5] * len(phi_vert_preferred)
sigma_thetas = [0.01] * len(phi_vert_preferred)
sigma_phis = [0.01] * len(phi_vert_preferred)
scaling_factors = [1] * len(phi_vert_preferred)
visual_bvc = False
n_hd_bvc = 8
n_hd_hdn = 20

start_location = [1, 1]
goal_location = [-1, 1]

bot = DriverFlying()
# Create walls with angles for this trial
# for index, model_type in enumerate(models):

bot.initialization(
    mode=RobotMode.LEARN_OJAS,
    randomize_start_location=True,
    run_time_hours=run_time_hours,
    phi_vert_preferred=phi_vert_preferred,
    sigma_rs=sigma_rs,
    sigma_thetas=sigma_thetas,
    sigma_phis=sigma_phis,
    scaling_factors=scaling_factors,
    num_bvc_per_dir=num_bvc_per_dir,
    visual_bvc=visual_bvc,
    n_hd_bvc=n_hd_bvc,
    n_hd_hdn=n_hd_hdn,
)

bot.run()

bot.simulationSetMode(bot.SIMULATION_MODE_PAUSE)
