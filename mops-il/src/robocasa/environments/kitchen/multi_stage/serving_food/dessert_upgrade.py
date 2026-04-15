from robocasa.environments.kitchen.kitchen import *


class DessertUpgrade(Kitchen):
    """Dessert Upgrade: composite task for Serving Food activity.

    Simulates the task of serving dessert.

    Steps:
        Move the dessert items from the plate to the tray.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self) -> None:
        super()._setup_kitchen_references()
        self.counter = self.register_fixture_ref(
            "counter", {"id": FixtureType.COUNTER_NON_CORNER, "size": (1.0, 0.4)}
        )
        self.init_robot_base_pos = self.counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()

        ep_meta["lang"] = "Move the dessert items from the plate to the tray."

        return ep_meta

    def _reset_internal(self) -> None:
        """Resets simulation internal configurations."""
        super()._reset_internal()

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            {
                "name": "receptacle",
                "obj_groups": "tray",
                "graspable": False,
                "placement": {
                    "fixture": self.counter,
                    "sample_region_kwargs": {"top_size": (1.0, 0.4)},
                    "size": (1, 0.4),
                    "pos": (0, -1),
                },
            }
        )

        cfgs.append(
            {
                "name": "dessert1",
                "obj_groups": "sweets",
                "graspable": True,
                "placement": {
                    "fixture": self.counter,
                    "size": (1, 0.4),
                    "pos": (0, -1),
                    "try_to_place_in": "plate",
                },
            }
        )

        cfgs.append(
            {
                "name": "dessert2",
                "obj_groups": "sweets",
                "graspable": True,
                "placement": {
                    "fixture": self.counter,
                    "size": (1, 0.4),
                    "pos": (0, -1),
                    "try_to_place_in": "plate",
                },
            }
        )

        return cfgs

    def _check_success(self):
        sweets_on_tray = OU.check_obj_in_receptacle(
            self, "dessert1", "receptacle"
        ) and OU.check_obj_in_receptacle(self, "dessert2", "receptacle")

        return sweets_on_tray and OU.gripper_obj_far(self, "receptacle")
