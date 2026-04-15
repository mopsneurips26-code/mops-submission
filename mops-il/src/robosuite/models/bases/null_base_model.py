"""Defines the null base model."""

from robosuite.models.bases.robot_base_model import RobotBaseModel


class NullBaseModel(RobotBaseModel):
    @property
    def naming_prefix(self) -> str:
        return f"nullbase{self.idn}_"
