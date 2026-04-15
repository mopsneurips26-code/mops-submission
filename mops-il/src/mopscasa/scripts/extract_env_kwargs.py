import argparse
import glob
import json
import pathlib
from importlib import resources

import h5py
from loguru import logger


def glob_h5py_files(directory: pathlib.Path) -> list[str]:
    """Get a list of all .h5py files in a directory."""
    h5py_files = glob.glob(str(directory / "**/*.hdf5"), recursive=True)
    return [f for f in h5py_files if "kitchen_navigate" not in f]


def extract_env_meta(h5py_file: str) -> dict:
    """Extract environment kwargs from a dataset h5py file.

    Args:
        h5py_file (str): Path to the .h5py dataset file.

    Returns:
        dict: Extracted environment kwargs.
    """
    with h5py.File(h5py_file, "r") as f:
        env_meta = json.loads(f["data"].attrs["env_args"])

    return env_meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract environment kwargs from dataset h5py files."
    )
    parser.add_argument(
        "--dataset_dir",
        type=pathlib.Path,
        required=True,
        help="Directory containing dataset .h5py files.",
    )
    args = parser.parse_args()

    h5py_files = glob_h5py_files(args.dataset_dir)
    logger.info(f"Found {len(h5py_files)} .h5py files in {args.dataset_dir}")

    all_kwargs = {}
    for h5py_file in h5py_files:
        env_meta = extract_env_meta(h5py_file)
        all_kwargs[env_meta["env_name"]] = env_meta["env_kwargs"]

    # Save to JSON file in resources
    output_path = resources.files("mopscasa.resources").joinpath("env_kwargs.json")
    with open(output_path, "w") as f:
        json.dump(all_kwargs, f, indent=4)
    logger.info(f"Extracted environment kwargs saved to {output_path}")
