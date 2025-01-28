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
# 4. INTELLIGENT_CURIOSITY

bot.initialization(
    mode=RobotMode.LEARN_OJAS,
    randomize_start_loc=True,
    run_time_hours=1,
    start_loc=[2, -2],
    explore_mthd=ExploreMethod.INTELLIGENT_CURIOSITY,
    use_existing_visitation_map=False,
    environment_label="15x15",
    bvc_max_dist=30.0,
    bvc_sigma_ang=90.0,
    bvc_sigma_d=0.7,
    num_place_cells=1000,
    n_hd=8,
) 
bot.run()
