"""my_controller_iCreate controller."""
from driver import Driver
import numpy as np
import matplotlib.pyplot as plot
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

np.random.seed(0)
bot = Driver()
bot.initialization(mode="learning", randomize_start_loc=True)
# bot.clear()
# bot.visualize_replay(4, 'all')
bot.run("explore")


# 1 learning, explore (RUN FOR ABOUT 2 HRS SIMULATION TIME, OR OBSERVE BVC-PCV WEIGHTS)
# 2 dmtp, explore (RUN ONCE)
# 3 dmtp, exploit