from robocasa.environments.kitchen.kitchen import *


class ColorfulSalsa(Kitchen):
    """Colorful Salsa: composite task for Mixing And Blending activity.

    Simulates the task of preparing a colorful salsa.

    Steps:
        Place the avocado, onion, tomato and bell pepper on the cutting board.
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
        ep_meta["lang"] = (
            "Place the avocado, onion, tomato and bell pepper on the cutting board."
        )
        return ep_meta

    def _reset_internal(self) -> None:
        """Resets simulation internal configurations."""
        super()._reset_internal()

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            {
                "name": "receptacle",
                "obj_groups": "cutting_board",
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
                "name": "bell_pepper",
                "obj_groups": "bell_pepper",
                "placement": {
                    "fixture": self.counter,
                    # sample_region_kwargs=dict(
                    #     top_size=(1.0, 0.4)
                    # ),
                    "size": (1, 0.4),
                    "pos": (0, -1),
                },
            }
        )

        cfgs.append(
            {
                "name": "tomato",
                "obj_groups": "tomato",
                "placement": {
                    "fixture": self.counter,
                    # sample_region_kwargs=dict(
                    #     top_size=(1.0, 0.4)
                    # ),
                    "size": (1, 0.4),
                    "pos": (0, -1),
                },
            }
        )

        cfgs.append(
            {
                "name": "avocado",
                "obj_groups": "avocado",
                "placement": {
                    "fixture": self.counter,
                    # sample_region_kwargs=dict(
                    #     top_size=(1.0, 0.4)
                    # ),
                    "size": (1, 0.4),
                    "pos": (0, -1),
                },
            }
        )

        cfgs.append(
            {
                "name": "onion",
                "obj_groups": "onion",
                "placement": {
                    "fixture": self.counter,
                    # sample_region_kwargs=dict(
                    #     top_size=(1.0, 0.4)
                    # ),
                    "size": (1, 0.4),
                    "pos": (0, -1),
                },
            }
        )

        return cfgs

    def _check_success(self):
        vegetables_on_board = (
            OU.check_obj_in_receptacle(self, "onion", "receptacle")
            and OU.check_obj_in_receptacle(self, "avocado", "receptacle")
            and OU.check_obj_in_receptacle(self, "tomato", "receptacle")
            and OU.check_obj_in_receptacle(self, "bell_pepper", "receptacle")
        )

        return vegetables_on_board and OU.gripper_obj_far(self, "receptacle")
