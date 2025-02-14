"""my_controller_iCreate controller."""

from driver import Driver

# Add root directory to python to be able to import
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # Moves two levels up
sys.path.append(str(PROJECT_ROOT))  # Add project root to sys.path

from core.robot.robot_mode import RobotMode


bot = Driver()
# 1. LEARN_OJAS
# 2. LEARN_HEBB
# 3. DMTP
# 4. EXPLOIT
# (optional) PLOTTING

bot.initialization(
    mode=RobotMode.EXPLOIT,
    run_time_hours=4,
    randomize_start_loc=False,
    start_loc=[8, -8],
    goal_location=[-7, 7],
    max_dist=30,
    num_pc=2000
)

bot.run()
