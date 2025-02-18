from driver import Driver
from core.robot.robot_mode import RobotMode

driver = Driver()
driver.initialization(mode=RobotMode.MANUAL_CONTROL)
driver.run()
