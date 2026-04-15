from .robot_model import RobotModel, create_robot  # noqa: I001
from .manipulators import *  # noqa: I001
from .compositional import *  # noqa: I001


def is_robosuite_robot(robot: str) -> bool:
    """Robot is robosuite repo robot if can import robot class from robosuite.models.robots."""
    try:
        module = __import__("robosuite.models.robots", fromlist=[robot])
        getattr(module, robot)
        return True
    except (ImportError, AttributeError):
        return False
