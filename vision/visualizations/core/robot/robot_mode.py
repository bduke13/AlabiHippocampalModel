from enum import Enum, auto


class RobotMode(Enum):
    """Defines the different operating modes for the robot's behavior and learning.

    The robot can operate in several modes that control its behavior, learning mechanisms,
    and data collection. These modes determine how the robot explores its environment,
    learns from experiences, and utilizes learned information.

    Modes:
        LEARN_OJAS: Initial learning phase where the robot randomly explores while only
            enabling competition (Oja's rule) between place cells. Runs until time limit.

        LEARN_HEBB: Secondary learning phase with both Oja's rule and tripartite (Hebbian)
            learning enabled during random exploration. Must run after LEARN_OJAS since
            place cells need to stabilize first.

        DMTP: Delayed Matching to Place task. Random exploration with both learning rules
            enabled until goal is reached, then updates reward map in reward cell network.

        EXPLOIT: Goal-directed navigation using the learned reward map. Both learning
            rules remain enabled while the robot navigates to known goals.

        PLOTTING: Random exploration mode with all learning disabled in both place cell
            and reward cell networks. Used for visualization and analysis.

        MANUAL_CONTROL: Enables direct user control of the robot in the Webots simulator
            through keyboard inputs.

        RECORDING: Random exploration with learning disabled, focused on collecting and
            saving sensor data for offline analysis or training.
    """

    LEARN_OJAS = auto()
    LEARN_HEBB = auto()
    DMTP = auto()
    EXPLOIT = auto()
    PLOTTING = auto()
    MANUAL_CONTROL = auto()
    RECORDING = auto()
