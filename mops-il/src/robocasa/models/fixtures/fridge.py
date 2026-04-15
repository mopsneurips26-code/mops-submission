from robocasa.models.fixtures import Fixture


class Fridge(Fixture):
    """Fridge fixture class."""

    def __init__(
        self,
        xml="fixtures/appliances/fridges/pack_1/model.xml",
        name="fridge",
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            xml=xml, name=name, duplicate_collision_geoms=False, *args, **kwargs
        )

    @property
    def nat_lang(self) -> str:
        return "fridge"
