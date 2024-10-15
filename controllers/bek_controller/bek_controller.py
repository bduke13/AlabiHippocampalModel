"""my_controller_iCreate controller."""

from driver import Driver, RobotStage
import numpy as np
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

np.random.seed(0)
print("starting bot")
bot = Driver()
bot.initialization(
    new_stage=RobotStage.EXPLOIT, randomize_start_loc=True, run_time_hours=1
)
bot.run()
