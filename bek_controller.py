"""my_controller_iCreate controller."""
from driver import Driver
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

np.random.seed(0)
bot = Driver()

def run_bot(run_step, randomize_start_loc):
    if run_step not in [1, 2, 3, 4]:
        raise ValueError("Invalid stage. Choose either '1', '2', '3', or '4'")
    if run_step == 1:
        bot.clear()
        bot.initialization(stage="learning", run_mode="explore", randomize_start_loc=randomize_start_loc)
        bot.run()
    elif run_step == 2:
        bot.initialization(stage="dmtp", run_mode="explore", randomize_start_loc=randomize_start_loc)
        bot.run()
    elif run_step == 3:
        bot.initialization(stage="dmtp", run_mode="exploit", randomize_start_loc=randomize_start_loc)
        bot.run()
    elif run_step == 4: # plotting stage
        bot.initialization(stage="explore", run_mode="explore", randomize_start_loc=randomize_start_loc)
        bot.run()

run_bot(run_step=3, randomize_start_loc=True)

# 1 learning, explore (RUN FOR ABOUT 2 HRS SIMULATION TIME, OR OBSERVE BVC-PCV WEIGHTS)
# 2 dmtp, explore (RUN ONCE)
# 3 dmtp, exploit