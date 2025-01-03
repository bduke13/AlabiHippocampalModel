"""my_controller_iCreate controller."""


def create_cross_walls(bot, center1_angle, center2_angle):
    """Create walls in a cross pattern programmatically."""
    # Delete existing walls first
    delete_walls(bot)

    # Get the root node of the scene tree
    root_node = bot.getRoot()
    children_field = root_node.getField("children")

    # Create horizontal wall (Center1)
    children_field.importMFNodeFromString(
        -1,
        """
        Wall {
          rotation 0 -1 0 %f
          name "wall(Center1)"
          size 5 2.4 0.3
          appearance Roughcast {
            colorOverride 0.8 0.45 0.25
            textureTransform TextureTransform {
              scale 1 2.4
            }
          }
        }
        """
        % (center1_angle),
    )

    # Create vertical wall (Center2)
    children_field.importMFNodeFromString(
        -1,
        """
        Wall {
          rotation 0 1 0 %f
          name "wall(Center2)"
          size 5 2.4 0.3
          appearance Roughcast {
            colorOverride 0.8 0.45 0.25
            textureTransform TextureTransform {
              scale 1 2.4
            }
          }
        }
        """
        % (center2_angle),
    )


if __name__ == "__main__":
    import tensorflow as tf
    from robot_mode import RobotMode
    from driver_vertical import DriverVertical
    from driver import Driver

    # Define the angles for each trial (in radians)
    center1_angles = [-1.57081, 1.58784, 1.63784, 1.71778]
    center2_angles = [0, -0.261805, -0.523605, -0.785405]
    bot = DriverVertical()
    for test_index in range(4):

        # Create walls with angles for this trial
        create_cross_walls(bot, center1_angles[test_index], center2_angles[test_index])

        bot.initialization(
            mode=RobotMode.LEARN_OJAS,
            randomize_start_loc=True,
            run_time_hours=0.005,
            start_loc=[4, -4],
        )
        bot.run()
        bot.simulationReset()

    bot.simulationSetMode(bot.SIMULATION_MODE_PAUSE)
