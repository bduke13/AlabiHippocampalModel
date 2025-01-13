"""my_controller_iCreate controller."""

if __name__ == "__main__":
    import numpy as np
    import tensorflow as tf
    from robot_mode import RobotMode
    from driver import Driver
    import random

    run_time_hours = 4
    start_locations = [[4, 4], [-4, 4], [4, -4], [-4, -4]]
    model_type = "2D_250"
    world_name = "upright"

    print(f"Running model {model_type} in world {world_name}")

    bot = Driver()
    # Create walls with angles for this trial
    start_location = random.choice(start_locations)

    bot.initialization(
        mode=RobotMode.LEARN_OJAS,
        randomize_start_loc=False,
        run_time_hours=run_time_hours,
        start_loc=start_location,
        file_prefix=f"IJCNN/{model_type}/{world_name}/",
    )

    bot.run()

    bot.simulationSetMode(bot.SIMULATION_MODE_PAUSE)
