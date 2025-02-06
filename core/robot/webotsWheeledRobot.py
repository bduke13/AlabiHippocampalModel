from controller import Supervisor
import numpy as np
from typing import Optional, List

from core.layers.boundary_vector_cell_layer import BoundaryVectorCellLayer
from core.robot.genericWheeledRobot import GenericWheeledRobot


class Create3Supervisor(Supervisor, GenericWheeledRobot):
    """Create3 robot controller with supervisor capabilities."""

    def __init__(self):
        GenericWheeledRobot.__init__(self)
        Supervisor.__init__(self)

        # Create3 specific parameters
        self.max_speed = 16
        self.left_speed = self.max_speed
        self.right_speed = self.max_speed

        self.wheel_radius = 0.031
        self.axle_length = 0.271756
        self.timestep = 32 * 3  # time step for webots in ms between simulation frames
        self.tau_w = 5  # time constant for the window function

        # Initialize hardware
        self.robot = self.getFromDef("agent")
        # self.keyboard = self.getKeyboard()
        self.compass = self.getDevice("compass")
        self.range_finder = self.getDevice("range-finder")
        self.left_bumper = self.getDevice("bumper_left")
        self.right_bumper = self.getDevice("bumper_right")
        self.rotation_field = self.robot.getField("rotation")

        # Initialize motors and sensors
        self.left_motor = self.getDevice("left wheel motor")
        self.right_motor = self.getDevice("right wheel motor")
        self.left_position_sensor = self.getDevice("left wheel sensor")
        self.right_position_sensor = self.getDevice("right wheel sensor")

        # Enable devices
        self.keyboard.enable(self.timestep)
        self.compass.enable(self.timestep)
        self.range_finder.enable(self.timestep)
        self.left_bumper.enable(self.timestep)
        self.right_bumper.enable(self.timestep)
        self.left_position_sensor.enable(self.timestep)
        self.right_position_sensor.enable(self.timestep)

    def move(self) -> None:
        """Updates motor positions and velocities."""
        self.left_motor.setPosition(float("inf"))
        self.right_motor.setPosition(float("inf"))
        self.left_motor.setVelocity(self.left_speed)
        self.right_motor.setVelocity(self.right_speed)

    def stop(self) -> None:
        """Stops the robot."""
        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)

    def forward(self) -> None:
        """Moves forward at max speed."""
        self.left_speed = self.max_speed
        self.right_speed = self.max_speed
        self.move()

    def rotate(self, direction: int, speed_factor: float = 0.3) -> None:
        """Rotates the robot."""
        speed = self.max_speed * speed_factor
        self.left_speed = speed * direction
        self.right_speed = -speed * direction
        self.move()

    def turn(self, angle: float, circle: bool = False) -> None:
        """Turns by specified angle."""
        self.stop()
        self.move()
        l_offset = self.left_position_sensor.getValue()
        r_offset = self.right_position_sensor.getValue()
        self.sense()
        neg = -1.0 if (angle < 0.0) else 1.0
        if circle:
            self.left_motor.setVelocity(0)
        else:
            self.left_motor.setVelocity(neg * self.max_speed / 2)
        self.right_motor.setVelocity(-neg * self.max_speed / 2)
        while True:
            l = self.left_position_sensor.getValue() - l_offset
            r = self.right_position_sensor.getValue() - r_offset
            dl = l * self.wheel_radius
            dr = r * self.wheel_radius
            orientation = neg * (dl - dr) / self.axle_length
            self.sense()
            if not orientation < neg * angle:
                break
        self.stop()

    def manual_control(self):
        """Enables manual control of the robot using keyboard inputs.

        Controls:
            w or UP_ARROW: Move forward
            a or LEFT_ARROW: Rotate counterclockwise
            s or DOWN_ARROW: Stop movement
            d or RIGHT_ARROW: Rotate clockwise

        Note:
        If control is not working try to click into the sim environment again.
        Sometimes resetting the sim makes the keyboard disconnect.
        """
        k = self.keyboard.getKey()
        if k == ord("W") or k == self.keyboard.UP:
            self.forward()
        elif k == ord("A") or k == self.keyboard.LEFT:
            self.rotate(direction=1, speed_factor=0.3)
        elif k == ord("D") or k == self.keyboard.RIGHT:
            self.rotate(direction=-1, speed_factor=0.3)
        elif k == ord("S") or k == self.keyboard.DOWN:
            self.stop()

        # Always step simulation forward and update sensors
        self.sense()
        self.step(self.timestep)

    def get_bearing_in_degrees(self, north: List[float]) -> float:
        """Converts compass readings to bearing in degrees.

        Args:
            north (List[float]): List containing the compass sensor values [x, y, z].

        Returns:
            float: Bearing angle in degrees (0-360), where 0 is North.
        """
        rad = np.arctan2(north[1], north[0])
        bearing = (rad - 1.5708) / np.pi * 180.0
        if bearing < 0:
            bearing = bearing + 360.0

        return bearing
