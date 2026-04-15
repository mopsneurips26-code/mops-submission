"""Utilities for loading RoboCasa environment configurations."""

import json
from importlib import resources


def load_env_info() -> dict:
    """Load environment keyword arguments from the bundled resource file.

    Returns:
        Dict mapping environment names to their creation kwargs.
    """
    with resources.open_text("mopscasa.resources", "env_kwargs.json") as f:
        env_info = json.load(f)
    return env_info
