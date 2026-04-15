from robocasa.environments.kitchen.kitchen import *


class MeatTransfer(Kitchen):
    """Meat Transfer: composite task for Chopping Food activity.

    Simulates the task of transferring meat to a container.

    Steps:
        Retrieve a container (either a pan or a bowl) from the cabinet, then place
        the raw meat into the container to avoid contamination.

    Args:
        cab_id: Enum which serves as a unique identifier for different cabinets.
            Default to FixtureType.DOOR_TOP_HINGE_DOUBLE to have space for
            initializing bowl/pan

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
            "counter", {"id": FixtureType.COUNTER, "ref": self.cab, "size": (0.5, 0.5)}
        )

        self.init_robot_base_pos = self.cab

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        cont_name = self.get_obj_lang("container")
        ep_meta["lang"] = (
            f"Retrieve the {cont_name} from the cabinet, "
            f"then place the raw meat into the {cont_name} to avoid contamination."
        )
        return ep_meta

    def _reset_internal(self) -> None:
        """Resets simulation internal configurations."""
        super()._reset_internal()

    def _get_obj_cfgs(self):
        cfgs = []

        if self.rng.random() < 0.5:
            cfgs.append(
                {
                    "name": "container",
                    "obj_groups": "pan",
                    "graspable": True,
                    "placement": {
                        "fixture": self.cab,
                        # ensure_object_boundary_in_range=False because the pans handle is a part of the
                        # bounding box making it hard to place it if set to True
                        "ensure_object_boundary_in_range": False,
                        "size": (0.05, 0.02),
                        "pos": (0, 0),
                        # apply a custom rotation for the pan so that it fits better in the cabinet
                        # (if the handle sticks out it may not fit)
                        "rotation": (2 * np.pi / 8, 3 * np.pi / 8),
                    },
                }
            )
        else:
            cfgs.append(
                {
                    "name": "container",
                    "obj_groups": "bowl",
                    "graspable": True,
                    "placement": {
                        "fixture": self.cab,
                        "ensure_object_boundary_in_range": False,
                        "size": (0.02, 0.02),
                        "pos": (0, 0),
                    },
                }
            )

        cfgs.append(
            {
                "name": "meat",
                "obj_groups": "meat",
                "placement": {
                    "fixture": self.counter,
                    "sample_region_kwargs": {"ref": self.cab},
                    "size": (0.5, 0.4),
                    "pos": (0.0, -1.0),
                },
            }
        )
        return cfgs

    def _check_success(self):
        return (
            OU.check_obj_fixture_contact(self, "container", self.counter)
            and OU.gripper_obj_far(self, obj_name="meat")
            and OU.check_obj_in_receptacle(self, "meat", "container")
        )
