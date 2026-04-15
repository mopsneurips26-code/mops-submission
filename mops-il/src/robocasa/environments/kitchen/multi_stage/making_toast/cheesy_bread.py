from robocasa.environments.kitchen.kitchen import *


class CheesyBread(Kitchen):
    """Cheesy Bread: composite task for Making Toast activity.

    Simulates the task of making cheesy bread.

    Steps:
        Start with a slice of bread already on a plate and a wedge of cheese on the
        counter. Pick up the wedge of cheese and place it on the slice of bread to
        prepare a simple cheese on bread dish.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self) -> None:
        super()._setup_kitchen_references()

        self.counter = self.register_fixture_ref(
            "counter", {"id": FixtureType.COUNTER_NON_CORNER, "size": (0.6, 0.6)}
        )
        self.init_robot_base_pos = self.counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            "Pick up the wedge of cheese and place it on the slice of bread to prepare a simple cheese on bread dish."
        )

        return ep_meta

    def _reset_internal(self) -> None:
        """Resets simulation internal configurations."""
        super()._reset_internal()

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            {
                "name": "bread",
                "obj_groups": "bread",
                "placement": {
                    "fixture": self.counter,
                    "size": (0.5, 0.7),
                    "pos": (0, -1.0),
                    "try_to_place_in": "cutting_board",
                },
            }
        )
        cfgs.append(
            {
                "name": "cheese",
                "obj_groups": "cheese",
                "placement": {
                    "fixture": self.counter,
                    "size": (1.0, 0.3),
                    "pos": (0, -1.0),
                },
            }
        )

        # Distractor on the counter
        cfgs.append(
            {
                "name": "distr_counter",
                "obj_groups": "all",
                "placement": {
                    "fixture": self.counter,
                    "size": (1.0, 0.20),
                    "pos": (0, 1.0),
                },
            }
        )
        return cfgs

    def _check_success(self):
        # Bread is still on the cutting board, and cheese is on top
        return (
            OU.check_obj_in_receptacle(self, "bread", "bread_container")
            and OU.gripper_obj_far(self, obj_name="cheese")
            and self.check_contact(self.objects["cheese"], self.objects["bread"])
        )
