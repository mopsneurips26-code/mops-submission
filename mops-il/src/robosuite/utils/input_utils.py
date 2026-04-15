"""Utility functions for grabbing user inputs."""

import robosuite as suite

# from robosuite.devices import *
from robosuite.models.robots import *
from robosuite.robots import *


def choose_environment():
    """Prints out environment options, and returns the selected env_name choice.

    Returns:
        str: Chosen environment name

    """
    # get the list of all environments
    envs = sorted(suite.ALL_ENVIRONMENTS)

    # Select environment to run
    print("Here is a list of environments in the suite:\n")

    for k, env in enumerate(envs):
        print(f"[{k}] {env}")
    print()
    try:
        s = input(
            "Choose an environment to run "
            + f"(enter a number from 0 to {len(envs) - 1}): "
        )
        # parse input into a number within range
        k = min(max(int(s), 0), len(envs))
    except:
        k = 0
        print(f"Input is not valid. Use {envs[k]} by default.\n")

    # Return the chosen environment name
    return envs[k]


def choose_controller(part_controllers=False):
    """Prints out controller options, and returns the requested controller name.

    Returns:
        str: Chosen controller name

    """
    # get the list of all controllers
    controllers = (
        list(suite.ALL_PART_CONTROLLERS)
        if part_controllers
        else list(suite.ALL_COMPOSITE_CONTROLLERS)
    )

    # Select controller to use
    print("Here is a list of controllers in the suite:\n")

    for k, controller in enumerate(controllers):
        print(f"[{k}] {controller}")
    print()
    try:
        s = input(
            "Choose a controller for the robot "
            + f"(enter a number from 0 to {len(controllers) - 1}): "
        )
        # parse input into a number within range
        k = min(max(int(s), 0), len(controllers) - 1)
    except:
        k = 0
        print(f"Input is not valid. Use {controllers} by default."[k])

    # Return chosen controller
    return controllers[k]


def choose_multi_arm_config():
    """Prints out multi-arm environment configuration options, and returns the requested config name.

    Returns:
        str: Requested multi-arm configuration name

    """
    # Get the list of all multi arm configs
    env_configs = {
        "Opposed": "opposed",
        "Parallel": "parallel",
        "Single-Robot": "single-robot",
    }

    # Select environment configuration
    print(
        "A multi-arm environment was chosen. Here is a list of multi-arm environment configurations:\n"
    )

    for k, env_config in enumerate(list(env_configs)):
        print(f"[{k}] {env_config}")
    print()
    try:
        s = input(
            "Choose a configuration for this environment "
            + f"(enter a number from 0 to {len(env_configs) - 1}): "
        )
        # parse input into a number within range
        k = min(max(int(s), 0), len(env_configs))
    except:
        k = 0
        print(f"Input is not valid. Use {list(env_configs)[k]} by default.")

    # Return requested configuration
    return list(env_configs.values())[k]


def choose_robots(
    exclude_bimanual=False, use_humanoids=False, exclude_single_arm=False
):
    """Prints out robot options, and returns the requested robot. Restricts options to single-armed robots if
    @exclude_bimanual is set to True (False by default). Restrict options to humanoids if @use_humanoids is set to True (Flase by default).

    Args:
        exclude_bimanual (bool): If set, excludes bimanual robots from the robot options
        use_humanoids (bool): If set, use humanoid robots

    Returns:
        str: Requested robot name

    """
    # Get the list of robots
    if exclude_single_arm:
        robots = set()
    else:
        robots = {
            "Sawyer",
            "Panda",
            "Jaco",
            "Kinova3",
            "IIWA",
            "UR5e",
            "SpotWithArmFloating",
            "XArm7",
        }

    # Add Baxter if bimanual robots are not excluded
    if not exclude_bimanual:
        robots.add("Baxter")
        robots.add("GR1ArmsOnly")
        robots.add("Tiago")
    if use_humanoids:
        robots.add("GR1ArmsOnly")

    # Make sure set is deterministically sorted
    robots = sorted(robots)

    # Select robot
    print("Here is a list of available robots:\n")

    for k, robot in enumerate(robots):
        print(f"[{k}] {robot}")
    print()
    try:
        s = input("Choose a robot " + f"(enter a number from 0 to {len(robots) - 1}): ")
        # parse input into a number within range
        k = min(max(int(s), 0), len(robots))
    except:
        k = 0
        print(f"Input is not valid. Use {list(robots)[k]} by default.")

    # Return requested robot
    return list(robots)[k]
