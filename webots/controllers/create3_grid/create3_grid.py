"""my_controller_iCreate controller with grid cells."""

#from driver_with_grid import DriverGrid
from driver_with_grid import DriverGrid

# Add root directory to python to be able to import
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # Moves two levels up
sys.path.append(str(PROJECT_ROOT))  # Add project root to sys.path

from core.robot.robot_mode import RobotMode

bot = DriverGrid()
# Robot Modes
# 1. LEARN_OJAS
# 2. LEARN_HEBB
# 3. DMTP
# 4. EXPLOIT
# (optional) PLOTTING

bot.initialization(
    mode=RobotMode.LEARN_OJAS,
    run_time_hours=1,
    randomize_start_loc=False,
    start_loc=[0, 0],
    goal_location=[-3, 3],
    max_dist=25,
    show_bvc_activation=False,
    num_place_cells=1000,
    num_grid_cells=400,
    grid_influence=0.5,
)

bot.run()