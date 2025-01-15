"""my_controller_iCreate controller."""

from driver import Driver, RobotMode, ExploreMethod

bot = Driver()
# Robot Modes
# 1. LEARN_OJAS
# 2. DMTP
# 3. EXPLOIT
# 4. (optional) PLOTTING
#
# Explore Modes
# 1. RANDOM_WALK
# 2. CURIOSITY
# 3. HYBRID

bot.initialization(
    mode=RobotMode.LEARN_HEBB,
    randomize_start_loc=True,
    run_time_hours=3,
    start_loc=[2, -2],
    explore_mthd=ExploreMethod.RANDOM_WALK,
    use_existing_visitation_map=False,
) 
bot.run()
