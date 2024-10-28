"""my_controller_iCreate controller."""

from driver import Driver, RobotStage
import numpy as np
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

np.random.seed(0)
bot = Driver()
# 1. LEARN_OJAS
# 2. LEARN_HEBB
# 3. EXPLOIT
# 4. (optional) PLOTTING

bot.initialization(
    stage=RobotStage.EXPLOIT, randomize_start_loc=True, run_time_hours=1
)
bot.run()
