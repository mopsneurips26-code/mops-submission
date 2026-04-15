"""Defines the MountModel class (Fixed Base that is mounted to the robot)."""

from robosuite.models.bases.robot_base_model import RobotBaseModel


class MountModel(RobotBaseModel):
    @property
    def naming_prefix(self) -> str:
        return f"fixed_mount{self.idn}_"
