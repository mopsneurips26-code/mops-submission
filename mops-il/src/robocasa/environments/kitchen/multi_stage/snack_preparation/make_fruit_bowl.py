from robocasa.environments.kitchen.kitchen import *


class MakeFruitBowl(Kitchen):
    """Make Fruit Bowl: composite task for Snack Preparation activity.

    Simulates the preparation of a fruit bowl snack.

    Steps:
        Pick the fruit from the cabinet and place them in the bowl.

    Args:
        cab_id (int): Enum which serves as a unique identifier for different
            cabinet types. Used to choose the cabinet from which the fruit are
            picked.

    """

    def __init__(
        self, cab_id=FixtureType.DOOR_TOP_HINGE_DOUBLE, *args, **kwargs
    ) -> None:
        self.cab_id = cab_id
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self) -> None:
        super()._setup_kitchen_references()

        self.cab = self.register_fixture_ref("cab", {"id": self.cab_id})
        self.counter = self.register_fixture_ref(
            "counter", {"id": FixtureType.COUNTER, "ref": self.cab, "size": (0.6, 0.4)}
        )
        self.init_robot_base_pos = self.cab

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        fruit1_name = self.get_obj_lang("fruit1")
        fruit2_name = self.get_obj_lang("fruit2")
        ep_meta["lang"] = (
            "Open the cabinet. "
            f"Pick the {fruit1_name} and {fruit2_name} from the cabinet and place them into the bowl. "
            "Then close the cabinet."
        )

        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            {
                "name": "bowl",
                "obj_groups": "bowl",
                "graspable": True,
                "placement": {
                    "fixture": self.counter,
                    "sample_region_kwargs": {"ref": self.cab, "top_size": (0.6, 0.4)},
                    "size": (1, 0.40),
                    "pos": ("ref", -1.0),
                },
            }
        )

        cfgs.append(
            {
                "name": "fruit1",
                "obj_groups": "fruit",
                "graspable": True,
                "placement": {
                    "fixture": self.cab,
                    "size": (0.50, 0.20),
                    "pos": (-0.5, -1.0),
                },
            }
        )

        cfgs.append(
            {
                "name": "fruit2",
                "obj_groups": "fruit",
                "graspable": True,
                "placement": {
                    "fixture": self.cab,
                    "size": (0.50, 0.20),
                    "pos": (0.5, -1.0),
                },
            }
        )

        cfgs.append(
            {
                "name": "distr_cab",
                "obj_groups": "all",
                "placement": {
                    "fixture": self.cab,
                    "size": (1.0, 0.20),
                    "pos": (0.0, 1.0),
                },
            }
        )

        return cfgs

    def _check_success(self):
        fruit1_in_bowl = OU.check_obj_in_receptacle(self, "fruit1", "bowl")
        fruit2_in_bowl = OU.check_obj_in_receptacle(self, "fruit2", "bowl")

        door_state = self.cab.get_door_state(env=self)

        door_closed = True
        for joint_p in door_state.values():
            if joint_p > 0.05:
                door_closed = False
                break

        return fruit1_in_bowl and fruit2_in_bowl and door_closed
