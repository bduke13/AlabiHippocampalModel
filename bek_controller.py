"""my_controller_iCreate controller."""
from driver import Driver
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

np.random.seed(0)
bot = Driver()

def run_bot(stage, randomize_start_loc):
    if stage not in [1, 2, 3]:
        raise ValueError("Invalid mode. Choose either '1', '2', or '3'")
    if stage == 1:
        bot.clear()
        bot.initialization(mode="learning", randomize_start_loc=randomize_start_loc)
        bot.run("explore")
    elif stage == 2:
        bot.initialization(mode="dmtp", randomize_start_loc=randomize_start_loc)
        bot.run("explore")
    elif stage == 3:
        bot.initialization(mode="dmtp", randomize_start_loc=randomize_start_loc)
        bot.run("exploit")
    else: # plotting mode
        bot.initialization(mode="explore", randomize_start_loc=randomize_start_loc)
        bot.run("explore")

run_bot(1, randomize_start_loc=True)

# 1 learning, explore (RUN FOR ABOUT 2 HRS SIMULATION TIME, OR OBSERVE BVC-PCV WEIGHTS)
# 2 dmtp, explore (RUN ONCE)
# 3 dmtp, exploit