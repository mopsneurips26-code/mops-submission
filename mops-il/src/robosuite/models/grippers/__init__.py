from .bd_gripper import BDGripper
from .fourier_hands import FourierLeftHand, FourierRightHand
from .gripper_factory import gripper_factory
from .gripper_model import GripperModel
from .gripper_tester import GripperTester
from .inspire_hands import InspireLeftHand, InspireRightHand
from .jaco_three_finger_gripper import (
    JacoThreeFingerDexterousGripper,
    JacoThreeFingerGripper,
)
from .null_gripper import NullGripper
from .panda_gripper import PandaGripper
from .rethink_gripper import RethinkGripper
from .robotiq_85_gripper import Robotiq85Gripper
from .robotiq_140_gripper import Robotiq140Gripper
from .robotiq_three_finger_gripper import (
    RobotiqThreeFingerDexterousGripper,
    RobotiqThreeFingerGripper,
)
from .wiping_gripper import WipingGripper
from .xarm7_gripper import XArm7Gripper

GRIPPER_MAPPING = {
    "RethinkGripper": RethinkGripper,
    "PandaGripper": PandaGripper,
    "JacoThreeFingerGripper": JacoThreeFingerGripper,
    "JacoThreeFingerDexterousGripper": JacoThreeFingerDexterousGripper,
    "WipingGripper": WipingGripper,
    "Robotiq85Gripper": Robotiq85Gripper,
    "Robotiq140Gripper": Robotiq140Gripper,
    "RobotiqThreeFingerGripper": RobotiqThreeFingerGripper,
    "RobotiqThreeFingerDexterousGripper": RobotiqThreeFingerDexterousGripper,
    "BDGripper": BDGripper,
    "InspireLeftHand": InspireLeftHand,
    "InspireRightHand": InspireRightHand,
    "FourierLeftHand": FourierLeftHand,
    "FourierRightHand": FourierRightHand,
    "XArm7Gripper": XArm7Gripper,
    None: NullGripper,
}

ALL_GRIPPERS = GRIPPER_MAPPING.keys()


def register_gripper(target_class):
    GRIPPER_MAPPING[target_class.__name__] = target_class
    return target_class
