"""my_controller_iCreate controller."""

from driver_3D_bvc import Driver

# Add root directory to python to be able to import
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # Moves two levels up
sys.path.append(str(PROJECT_ROOT))  # Add project root to sys.path

from core.robot.robot_mode import RobotMode

run_time_hours = 2

num_bvc_per_dir = 50
num_place_cells = 100

phi_vert_preferred = [0.0, 0.3, 0.6]
sigma_rs = [1] * len(phi_vert_preferred)
sigma_thetas = [0.02] * len(phi_vert_preferred)
sigma_phis = [0.02] * len(phi_vert_preferred)
scaling_factors = [1] * len(phi_vert_preferred)
visual_bvc = False
max_dist = 10

start_location = [1, 1]
goal_location = [-1, 1]


bot = Driver()
# 1. LEARN_OJAS
# 2. LEARN_HEBB
# 3. DMTP
# 4. EXPLOIT
# (optional) PLOTTING

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
    max_dist=max_dist,
)
bot.run()
