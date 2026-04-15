from robocasa.environments.kitchen.kitchen import *


class SweetSavoryToastSetup(Kitchen):
    """Sweet Savory Toast Setup: composite task for Making Toast activity.

    Simulates the task of setting up the ingredients for making sweet and savory
    toast.

    Steps:
        Pick the avocado and bread from the counter and place it on the plate.
        Then pick the jam from the cabinet and place it next to the plate.
        Lastly, close the cabinet door.

    Args:
        cab_id (str): Enum which serves as a unique identifier for different
            cabinet types. Used to specify the cabinet where the jam is placed.

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
            "counter", {"id": FixtureType.COUNTER, "ref": self.cab, "size": (0.6, 0.6)}
        )

        self.init_robot_base_pos = self.cab

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            "Pick the avocado and bread from the counter and place them on the plate. "
            "Then pick the jam from the cabinet and place it next to the plate. "
            "Lastly, close the cabinet door."
        )
        return ep_meta

    def _reset_internal(self) -> None:
        """Resets simulation internal configurations."""
        super()._reset_internal()
        self.cab.set_door_state(min=0.90, max=1.0, env=self, rng=self.rng)

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            {
                "name": "plate",
                "obj_groups": "plate",
                "placement": {
                    "fixture": self.counter,
                    "sample_region_kwargs": {
                        "ref": self.cab,
                    },
                    "size": (1.0, 0.5),
                    "pos": ("ref", -1.0),
                },
            }
        )

        cfgs.append(
            {
                "name": "avocado",
                "obj_groups": "avocado",
                "placement": {
                    "fixture": self.counter,
                    "sample_region_kwargs": {
                        "ref": self.cab,
                    },
                    "size": (0.6, 0.6),
                    "pos": ("ref", -1.0),
                },
            }
        )

        cfgs.append(
            {
                "name": "bread",
                "obj_groups": "bread",
                "placement": {
                    "fixture": self.counter,
                    "sample_region_kwargs": {
                        "ref": self.cab,
                    },
                    "size": (0.6, 0.6),
                    "pos": (0, -1.0),
                },
            }
        )

        cfgs.append(
            {
                "name": "jam",
                "obj_groups": "jam",
                "graspable": True,
                "placement": {
                    "fixture": self.cab,
                    "size": (0.4, 0.4),
                    "pos": (0, -1.0),
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
                    "offset": (0.0, 0.0),
                },
            }
        )

        return cfgs

    def _check_success(self):
        gripper_obj_far = OU.gripper_obj_far(self, "plate")
        jam_on_counter = self.check_contact(self.objects["jam"], self.counter)
        food_on_plate = OU.check_obj_in_receptacle(
            self, "bread", "plate"
        ) and OU.check_obj_in_receptacle(self, "avocado", "plate")
        door_state = self.cab.get_door_state(env=self)

        closed = True
        for joint_p in door_state.values():
            if joint_p > 0.05:
                closed = False
                break

        return gripper_obj_far and food_on_plate and jam_on_counter and closed
