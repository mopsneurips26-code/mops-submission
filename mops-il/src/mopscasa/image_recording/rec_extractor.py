import json
from copy import deepcopy

import numpy as np

import robocasa.utils.robomimic.robomimic_tensor_utils as TensorUtils
from mopscasa.observation_extender import ObservationExtender


def extract_trajectory(
    env,
    initial_state,
    states,
    actions,
    done_mode,
    add_datagen_info=False,
):
    """Extract trajectory data.

    Helper function to extract observations, rewards,
    and dones along a trajectory using the simulator environment.

    Args:
        env (instance of EnvBase): environment
        initial_state (dict): initial simulation state to load
        states (np.array): array of simulation states to load to extract information
        actions (np.array): array of actions
        done_mode (int): how to write done signal. If 0, done is 1 whenever s' is a
            success state. If 1, done is 1 at the end of each trajectory.
            If 2, do both.
    """
    assert states.shape[0] == actions.shape[0]

    # load the initial state
    env.reset()
    observer = ObservationExtender(env.env)
    obs = env.reset_to(initial_state)

    # get updated ep meta in case it's been modified
    ep_meta = env.env.get_ep_meta()
    initial_state["ep_meta"] = json.dumps(ep_meta, indent=4)

    traj = {
        "obs": [],
        "next_obs": [],
        "rewards": [],
        "dones": [],
        "actions": np.array(actions),
        # actions_abs=[],
        "states": np.array(states),
        "initial_state_dict": initial_state,
        "datagen_info": [],
    }
    traj_len = states.shape[0]
    # iteration variable @t is over "next obs" indices
    for t in range(traj_len):
        obs = deepcopy(env.reset_to({"states": states[t]}))
        obs = observer.augment_observation(obs)

        # extract datagen info
        if add_datagen_info:
            datagen_info = env.base_env.get_datagen_info(action=actions[t])
        else:
            datagen_info = {}

        # infer reward signal
        # note: our tasks use reward r(s'), reward AFTER transition, so this is
        #       the reward for the current timestep
        r = env.get_reward()

        # infer done signal
        done = False
        if (done_mode == 1) or (done_mode == 2):
            # done = 1 at end of trajectory
            done = done or (t == traj_len)
        if (done_mode == 0) or (done_mode == 2):
            # done = 1 when s' is task success state
            done = done or env.is_success()["task"]
        done = int(done)

        # get the absolute action
        # action_abs = env.base_env.convert_rel_to_abs_action(actions[t])

        # collect transition
        traj["obs"].append(obs)
        traj["rewards"].append(r)
        traj["dones"].append(done)
        traj["datagen_info"].append(datagen_info)
        # traj["actions_abs"].append(action_abs)

    # convert list of dict to dict of list for obs dictionaries (for convenient writes to hdf5 dataset)
    traj["obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["obs"])
    traj["datagen_info"] = TensorUtils.list_of_flat_dict_to_dict_of_list(
        traj["datagen_info"]
    )

    # list to numpy array
    for k in traj:
        if k == "initial_state_dict":
            continue
        if isinstance(traj[k], dict):
            for kp in traj[k]:
                traj[k][kp] = np.array(traj[k][kp])
        else:
            traj[k] = np.array(traj[k])

    return traj
