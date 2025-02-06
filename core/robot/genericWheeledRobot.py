from abc import ABC, abstractmethod
from controller import Robot
from typing import Optional


class GenericWheeledRobot(ABC):
    """Abstract base class for wheeled robots defining common movement interface."""

    def __init__(self):
        self.max_speed: float = 0.0
        self.left_speed: float = 0.0
        self.right_speed: float = 0.0
        self.wheel_radius: float = 0.0
        self.axle_length: float = 0.0
        self.timestep: int = 32

        # These will be initialized by concrete classes
        self.left_motor = None
        self.right_motor = None
        self.left_position_sensor = None
        self.right_position_sensor = None

    @abstractmethod
    def move(self) -> None:
        """Updates motor positions and velocities."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stops all wheel movement."""
        pass

    @abstractmethod
    def forward(self) -> None:
        """Moves robot forward at maximum speed."""
        pass

    @abstractmethod
    def rotate(self, direction: int, speed_factor: float = 0.3) -> None:
        """Rotates robot in specified direction.

        Args:
            direction (int): 1 for clockwise, -1 for counterclockwise
            speed_factor (float): Multiplier for rotation speed (0.0 to 1.0)
        """
        pass

    @abstractmethod
    def turn(self, angle: float, circle: bool = False) -> None:
        """Turns robot by specified angle.

        Args:
            angle (float): Angle to turn in radians
            circle (bool): If True, pivot around one wheel
        """
        pass

    # @abstractmethod
    # def sense(self) -> None:
    #    """Updates all sensor readings."""
    #    pass
