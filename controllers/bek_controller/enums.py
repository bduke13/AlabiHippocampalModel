from enum import Enum, auto


# Enum for defining robot stages
class RobotMode(Enum):
    # Random exploration only enabling competition (ojas) between place cells until time limit
    LEARN_OJAS = auto()

    # Random exploration with both ojas and tripartite learning until time limit. Only run after learn_ojas (place cells havent stabilized before then)
    LEARN_HEBB = auto()

    # Random exploration with both ojas and tripartite learning until goal reached then updates reward map in rcn
    DMTP = auto()

    # Drives towards goal using learned reward map. Both ojas and tripartite are enabled
    EXPLOIT = auto()

    # Random exploration with no learning enabled in pcn or rcn
    PLOTTING = auto()

    # Manual control from user to robot in Webots
    MANUAL_CONTROL = auto()

    # Random exploration with no learning. Saves out sensor data for offline driver
    RECORDING = auto()

