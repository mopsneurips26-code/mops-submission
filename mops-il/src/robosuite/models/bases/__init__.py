from .floating_legged_base import FloatingLeggedBase
from .leg_base_model import LegBaseModel
from .mobile_base_model import MobileBaseModel
from .mount_model import MountModel
from .no_actuation_base import NoActuationBase
from .null_base import NullBase
from .null_base_model import NullBaseModel
from .null_mobile_base import NullMobileBase
from .null_mount import NullMount
from .omron_mobile_base import OmronMobileBase
from .rethink_minimal_mount import RethinkMinimalMount
from .rethink_mount import RethinkMount
from .robot_base_factory import robot_base_factory
from .robot_base_model import RobotBaseModel
from .spot_base import Spot, SpotFloating

BASE_MAPPING = {
    "RethinkMount": RethinkMount,
    "RethinkMinimalMount": RethinkMinimalMount,
    "NullMount": NullMount,
    "OmronMobileBase": OmronMobileBase,
    "NullMobileBase": NullMobileBase,
    "NoActuationBase": NoActuationBase,
    "FloatingLeggedBase": FloatingLeggedBase,
    "Spot": Spot,
    "SpotFloating": SpotFloating,
    "NullBase": NullBase,
}

ALL_BASES = BASE_MAPPING.keys()


def register_base(target_class):
    BASE_MAPPING[target_class.__name__] = target_class
    return target_class
