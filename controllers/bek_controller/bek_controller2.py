"""my_controller_iCreate controller."""

if __name__ == "__main__":
    import numpy as np
    import tensorflow as tf
    from robot_mode import RobotMode
    from driver import Driver
    import random

    run_time_hours = 4
    number_tests = 1
    start_locations = [[4, 4], [-4, 4], [4, -4], [-4, -4]]
    models = ["2D"]
    world_index = "outside"

    bot = Driver()
    # Create walls with angles for this trial
    for index, model_type in enumerate(models):
        for test_index in range(number_tests):
            start_location = random.choice(start_locations)

            bot.initialization(
                mode=RobotMode.LEARN_OJAS,
                randomize_start_loc=False,
                run_time_hours=run_time_hours,
                start_loc=start_location,
                file_prefix=f"IJCNN/{model_type}/world_{world_index}/trial_{test_index}/",
            )

            bot.run()
            bot.simulationReset()

    bot.simulationSetMode(bot.SIMULATION_MODE_PAUSE)
