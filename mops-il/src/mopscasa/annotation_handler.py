import json
from importlib import resources

import polars as pl

SINGLETON_ANNOTATION_HANDLER = None


def load_annotations():
    global SINGLETON_ANNOTATION_HANDLER
    if SINGLETON_ANNOTATION_HANDLER is None:
        SINGLETON_ANNOTATION_HANDLER = AnnotationHandler()
    return SINGLETON_ANNOTATION_HANDLER


class AnnotationHandler:
    def __init__(self) -> None:
        with (
            resources.files("mopscasa.resources")
            .joinpath("red_class_aff.parquet")
            .open("rb") as f
        ):
            class_affordance_df = pl.read_parquet(f)

        all_classes = ["background"] + class_affordance_df[
            "model_cat"
        ].unique().to_list()

        self.class_name_to_id = {name: idx for idx, name in enumerate(all_classes)}
        self.all_classes = all_classes

        self.affordance_id_map = {}
        self._fill_affordance_map()

        # NEW: Cache for affordance lists
        self._affordance_list_cache = {}
        self._build_affordance_cache(class_affordance_df)

        self.ignored_classes = []

    def _fill_affordance_map(self) -> None:
        with (
            resources.files("mopscasa.resources")
            .joinpath("critical_affordances.json")
            .open("r") as f
        ):
            self.affordance_id_map = json.load(f)

        # NEW: Reverse map for fast ID -> name lookup
        self.affordance_id_to_name = {
            idx: aff for aff, idx in self.affordance_id_map.items()
        }

    def _build_affordance_cache(self, df: pl.DataFrame) -> None:
        """Pre-compute affordance lists for all classes."""
        for row in df.select(["model_cat", "affordance_bitmask"]).iter_rows(named=True):
            self._affordance_list_cache[row["model_cat"]] = row["affordance_bitmask"]

    def get_num_affords(self) -> int:
        """Get the number of unique affordances."""
        return len(self.affordance_id_map)

    def _add_class(self, class_name: str) -> int:
        """Add a new class to the handler if it does not already exist.

        Args:
            class_name (str): The name of the class to add.
        """
        if class_name not in self.class_name_to_id:
            new_id = len(self.class_name_to_id)
            self.class_name_to_id[class_name] = new_id
            self.all_classes.append(class_name)
        return self.class_name_to_id[class_name]

    def get_class_id(self, class_name: str) -> int:
        """Get the ID of a class by its name.

        Args:
            class_name (str): The name of the class.

        Returns:
            int: The ID of the class
        """
        if class_name in self.class_name_to_id:
            return self.class_name_to_id[class_name]
        return self._add_class(class_name)

    def get_affordance_bitmask(self, class_name: str) -> int:
        """Get the bitmask of affordances for a given class name.

        Args:
            class_name (str): The name of the class.

        Returns:
            int: Bitmask of affordances associated with the class.
        """
        if class_name in self._affordance_list_cache:
            return self._affordance_list_cache[class_name]

        # Handle unknown classes
        if class_name not in self.ignored_classes:
            # logger.warning(f"Class name {class_name} not found in affordance cache.")
            self.ignored_classes.append(class_name)
        return 0

    def get_affordance_name(self, affordance_id: int) -> str:
        """Get the name of an affordance by its ID.

        Args:
            affordance_id (int): The ID of the affordance.

        Returns:
            str: The name of the affordance, or an empty string if the ID does not exist.
        """
        return self.affordance_id_to_name.get(affordance_id, "")

    def get_affordance_id(self, affordance_name: str) -> int:
        """Get the ID of an affordance by its name.

        Args:
            affordance_name (str): The name of the affordance.

        Returns:
            int: The ID of the affordance, or -1 if the name does not exist.
        """
        return self.affordance_id_map.get(affordance_name, -1)

    def get_known_name(self, parsed_name: str, sim_name: str) -> str:
        """Get the known class name from either parsed name or sim name.

        Args:
            parsed_name (str): The parsed class name.
            sim_name (str): The simulated class name.

        Returns:
            str: The known class name.
        """
        if sim_name in self.all_classes:
            return sim_name
        return parsed_name


if __name__ == "__main__":
    obj = AnnotationHandler()
    for cat in obj.all_classes:
        print(cat)
    # print(obj.get_affordance_bitmask("alcohol"))
