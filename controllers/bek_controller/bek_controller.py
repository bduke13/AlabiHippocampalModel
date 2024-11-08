# my_controller.py

from driver import Driver, RobotMode
import sys

def main():
    # Check command-line arguments to determine the phase
    if len(sys.argv) < 2:
        print("Please specify the phase: LEARN_OJAS, DMTP, or EXPLOIT")
        return

    phase = sys.argv[1].upper()

    # Initialize the robot
    bot = Driver()

    if phase == "LEARN_OJAS":
        print("Starting LEARN_OJAS Phase...")
        bot.initialization(
            mode=RobotMode.LEARN_OJAS,
            randomize_start_loc=True,
            run_time_hours=2,
            start_loc=[0, 0],
        )
        bot.run()
        # At the end of LEARN_OJAS, save the PCN weights
        bot.on_save()

    elif phase == "DMTP":
        print("Starting DMTP Phase...")
        # Load the PCN from the previous phase
        bot.initialization(
            mode=RobotMode.DMTP,
            randomize_start_loc=True,
            run_time_hours=2,
            start_loc=[0, 0],
        )
        bot.run()
        # At the end of DMTP, save the updated PCN and the tripartite connections
        bot.on_save()

    elif phase == "EXPLOIT":
        print("Starting EXPLOIT Phase...")
        # Load the PCN and RCN from previous phases
        bot.initialization(
            mode=RobotMode.EXPLOIT,
            randomize_start_loc=False,
            start_loc=[0, 0],  # Starting position
            run_time_hours=2,
        )

        # Extract adjacency matrix and create graph
        bot.extract_adjacency_matrix()
        bot.compute_place_cell_centers()
        bot.create_graph()

        # Define start and goal positions
        start_position = [0, 0]     # Starting at (X=0, Z=0)
        goal_position = [-1, 1]     # Goal at the red cylinder's position

        # Find the shortest path using the graph
        path = bot.find_shortest_path(start_position, goal_position)
        if path:
            print("Path found:", path)
            # Navigate along the path
            bot.navigate_along_path(path)
        else:
            print("No path found.")
    else:
        print("Invalid phase specified. Choose LEARN_OJAS, DMTP, or EXPLOIT.")

if __name__ == "__main__":
    main()
