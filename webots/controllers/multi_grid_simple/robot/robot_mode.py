from enum import Enum, auto


class RobotMode(Enum):
    """Simple controller mode set used by the local multi_grid_simple fork."""

    LEARN_OJAS = auto()
    LEARN_HEBB = auto()
    DMTP = auto()
    EXPLOIT = auto()
    PLOTTING = auto()

