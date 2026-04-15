from robocasa.environments.kitchen.kitchen import *


class PastryDisplay(Kitchen):
    """Pastry Display: composite task for Baking activity.

    Simulates the task of displaying pastries.

    Steps:
        Place the pastries on the plates.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self) -> None:
        super()._setup_kitchen_references()
        self.sink = self.register_fixture_ref("sink", {"id": FixtureType.SINK})
        self.counter = self.register_fixture_ref(
            "counter", {"id": FixtureType.COUNTER, "ref": self.sink}
        )
        self.init_robot_base_pos = self.counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Place the pastries on the plates."
        return ep_meta

    def _reset_internal(self) -> None:
        """Resets simulation internal configurations."""
        super()._reset_internal()

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            {
                "name": "receptacle1",
                "obj_groups": "plate",
                "graspable": False,
                "washable": True,
                "placement": {
                    "fixture": self.counter,
                    "sample_region_kwargs": {
                        "ref": self.sink,
                        "loc": "left_right",
                    },
                    "size": (0.40, 0.40),
                    "pos": ("ref", -1.0),
                },
            }
        )

        cfgs.append(
            {
                "name": "receptacle2",
                "obj_groups": "plate",
                "graspable": False,
                "washable": True,
                "placement": {
                    "fixture": self.counter,
                    "sample_region_kwargs": {
                        "ref": self.sink,
                        "loc": "left_right",
                    },
                    "size": (0.40, 0.40),
                    "pos": ("ref", -1.0),
                },
            }
        )

        # use offserts and to make it easier to initialuze pastry1 and pastry2
        cfgs.append(
            {
                "name": "pastry1",
                "obj_groups": "pastry",
                "graspable": True,
                "placement": {
                    "fixture": self.counter,
                    "sample_region_kwargs": {
                        "ref": self.sink,
                        "loc": "left_right",
                    },
                    "size": (0.30, 0.30),
                    "pos": ("ref", -0.2),
                    "offset": (0.1, 0.0),
                },
            }
        )

        cfgs.append(
            {
                "name": "pastry2",
                "obj_groups": "pastry",
                "graspable": True,
                "placement": {
                    "fixture": self.counter,
                    "sample_region_kwargs": {
                        "ref": self.sink,
                        "loc": "left_right",
                    },
                    "size": (0.30, 0.30),
                    "pos": ("ref", -0.2),
                    "offset": (-0.1, 0.0),
                },
            }
        )

        return cfgs

    def _check_success(self):
        gripper_pastry1_far = OU.gripper_obj_far(self, obj_name="pastry1")
        gripper_pastry2_far = OU.gripper_obj_far(self, obj_name="pastry2")
        pastry1_in_receptacle1 = OU.check_obj_in_receptacle(
            self, "pastry1", "receptacle1"
        )
        pastry1_in_receptacle2 = OU.check_obj_in_receptacle(
            self, "pastry1", "receptacle2"
        )
        pastry2_in_receptacle1 = OU.check_obj_in_receptacle(
            self, "pastry2", "receptacle1"
        )
        pastry2_in_receptacle2 = OU.check_obj_in_receptacle(
            self, "pastry2", "receptacle2"
        )

        pastrys_placed = (pastry1_in_receptacle1 and pastry2_in_receptacle2) or (
            pastry1_in_receptacle2 and pastry2_in_receptacle1
        )

        return gripper_pastry1_far and gripper_pastry2_far and pastrys_placed
