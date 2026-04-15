import numpy as np
from loguru import logger

from robocasa.utils.robomimic.robomimic_env_wrapper import EnvRobocasa

from .annotation_handler import AnnotationHandler


class ObservationExtender:
    def __init__(self, env: EnvRobocasa) -> None:
        self.env = env
        self.mj_model = env.sim.model
        self.annotation_handler = AnnotationHandler()
        self._precompute_mappings()

    def _precompute_mappings(self) -> None:
        """Pre-compute class and affordance mappings for all geoms."""
        ngeom = self.mj_model.ngeom
        # Use uint8 for class IDs and uint32 for bitmasks
        self.uid_to_class = np.zeros(ngeom, dtype=np.uint8)
        self.uid_to_aff = np.zeros(ngeom, dtype=np.uint32)

        self.fixture_mapping = {
            k: v.__class__.__name__ for (k, v) in self.env.fixtures.items()
        }

        for uid in range(ngeom):
            try:
                geom_name = self.mj_model.geom_id2name(uid)
                if not geom_name:
                    continue

                class_name = _parse_geom_name(geom_name, self.fixture_mapping)
                self.uid_to_class[uid] = self.annotation_handler.get_class_id(
                    class_name
                )
                self.uid_to_aff[uid] = self.annotation_handler.get_affordance_bitmask(
                    class_name
                )
            except (ValueError, Exception):
                continue

    def extend_segmentation(
        self,
        segmentation_map: np.ndarray,
        cam_name: str,
        include_class: bool = False,
        flip_vert: bool = True,
    ) -> dict:
        """Extend the segmentation map with new class annotations."""
        # Handle (H, W, 1) or (H, W)
        segm_flat = segmentation_map.squeeze()
        # vertical flip to match original image orientation
        if flip_vert:
            segm_flat = np.flipud(segm_flat)

        # Create lookup indices, replacing -1 with 0 (safe index)
        # We assume segmentation_map contains geom IDs or -1
        valid_mask = segm_flat >= 0

        # Use int32 for indices to avoid overflow if original was smaller/larger
        lookup_indices = np.where(valid_mask, segm_flat, 0).astype(np.int32)

        # Clip to ensure we don't go out of bounds
        np.clip(lookup_indices, 0, len(self.uid_to_class) - 1, out=lookup_indices)

        # Vectorized lookup
        aff_segm = self.uid_to_aff[lookup_indices]

        # Zero out invalid pixels
        if not valid_mask.all():
            aff_segm[~valid_mask] = 0

        # Restore dimensions to match input
        if segmentation_map.ndim == 3:
            aff_segm = aff_segm[..., None]

        ext_segm = {
            f"{cam_name}_segmentation_affordance": aff_segm,
        }

        if include_class:
            class_segm = self.uid_to_class[lookup_indices]
            if not valid_mask.all():
                class_segm[~valid_mask] = 0
                class_segm = class_segm[..., None]
            ext_segm[f"{cam_name}_segmentation_class"] = class_segm

        return ext_segm

    def augment_observation(self, observation: dict, flip_vert: bool = True) -> dict:
        new_obs = {}
        for sensor in observation:
            if "_segmentation_" in sensor:
                extended_segm = self.extend_segmentation(
                    observation[sensor],
                    sensor.split("_segmentation")[0],
                    flip_vert=flip_vert,
                )
                new_obs.update(extended_segm)

        if len(new_obs) == 0:
            logger.warning(
                "No segmentation sensors found in the observation to extend."
            )

        new_obs = observation | new_obs
        return new_obs


def _parse_geom_name(geom_name: str, fixture_mapping: dict) -> str:
    """Parse the geometry name to extract the class name."""
    for fixture_name, class_name in fixture_mapping.items():
        if geom_name.startswith(fixture_name):
            return class_name.lower()
    return _simplify_geom_name(geom_name).lower()


def _simplify_geom_name(geom_name: str) -> str:
    """Parse the geometry name to extract the class name."""
    class_name_segments = []
    geom_name_split = geom_name.split("_")
    for geom_name_segment in geom_name_split:
        if _check_generic_word(geom_name_segment):
            break
        class_name_segments.append(geom_name_segment)
    return "".join(class_name_segments).lower()


def _check_generic_word(word: str) -> bool:
    """Check if a word is considered generic."""
    generic_words = {
        "main",
        "room",
        "right",
        "left",
        "fixed",
        "robot0",
        "mobilebase0",
        "gripper0",
        "top",
        "bottom",
        "housing",
    }  # Use set, not list
    return len(word) <= 2 or word.lower() in generic_words or word[1].isdigit()
