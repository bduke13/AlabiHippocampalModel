from enum import Enum, auto


class RobotMode(Enum):
    """Controller modes supported by the cleaned multiscale Webots driver."""

    LEARN_OJAS = auto()
    LEARN_HEBB = auto()
    DMTP = auto()
    LEARNING = auto()
    LEARN_LOCATIONS = auto()
    LEARN_LOCATIONS_COVERAGE = auto()
    LEARN_LOCATIONS_COVERAGE_AUTO = auto()
    EXPLOIT = auto()
    EXPLOIT_LOCATIONS = auto()
    EXPLOIT_LOCATIONS_RANDOM = auto()
    EXPLOIT_LOCATIONS_RANDOM_AUTO = auto()
    PLOTTING = auto()
    PLOTTING_AUTO = auto()
    PLOTTING_COVERAGE_AUTO = auto()
    LEARN_OJAS_AUTO = auto()
