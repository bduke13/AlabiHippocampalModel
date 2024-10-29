from enum import Enum, auto

# Enum for defining robot stages
class RobotStage(Enum):
    LEARN_OJAS = auto()
    LEARN_HEBB = auto()
    EXPLOIT = auto()
    PLOTTING = auto()
    MANUAL_CONTROL = auto()
    RECORDING = auto()

# Enum for defining robot operational modes
class RobotMode(Enum):
    EXPLORE = auto()
    LEARNING = auto()
    DMTP = auto()
    EXPLOIT = auto()