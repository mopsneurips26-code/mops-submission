import json
import queue
import time
import traceback

import h5py
import numpy as np
from loguru import logger

import robocasa.utils.robomimic.robomimic_dataset_utils as DatasetUtils
import robocasa.utils.robomimic.robomimic_env_utils as EnvUtils

from .rec_extractor import extract_trajectory


# runs multiple trajectory. If there has been an unrecoverable error, the system puts the current work back into the queue and exits
def extract_multiple_trajectories(
    process_num, current_work_array, work_queue, lock, args2, num_finished, mul_queue
) -> None:
    try:
        extract_multiple_trajectories_with_error(
            process_num, current_work_array, work_queue, lock, args2, mul_queue
        )
    except Exception as e:
        # If not stopping on error, put the current index back for another worker
        if not getattr(args2, "stop_on_error", False):
            work_queue.put(current_work_array[process_num])
        print("*>*" * 50)
        logger.error(f"Error process num {process_num}:")
        logger.error(e)
        logger.error(traceback.format_exc())
        logger.error("*>*" * 50)
        logger.error("")

    num_finished.value = num_finished.value + 1


def retrieve_new_index(process_num, current_work_array, work_queue, lock):
    with lock:
        if work_queue.empty():
            return -1
        try:
            tmp = work_queue.get(False)
            current_work_array[process_num] = tmp
            return tmp
        except queue.Empty:
            return -1


def extract_multiple_trajectories_with_error(
    process_num, current_work_array, work_queue, lock, args, mul_queue
) -> None:
    # create environment to use for data processing

    if args.add_datagen_info:
        import mimicgen.utils.file_utils as MG_FileUtils

        env_meta = MG_FileUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
    else:
        env_meta = DatasetUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
    if args.generative_textures:
        env_meta["env_kwargs"]["generative_textures"] = "100p"
    if args.randomize_cameras:
        env_meta["env_kwargs"]["randomize_cameras"] = True

    env_meta["env_kwargs"]["camera_segmentations"] = "element"
    env_meta["env_kwargs"]["camera_depths"] = True

    env = EnvUtils.create_env_for_data_processing(
        env_meta=env_meta,
        camera_names=args.camera_names,
        camera_height=args.camera_height,
        camera_width=args.camera_width,
        reward_shaping=args.shaped,
    )

    time.time()

    logger.info("==== Using environment with the following metadata ====")
    logger.info(json.dumps(env.serialize(), indent=4))
    logger.info("")

    # list of all demonstration episodes (sorted in increasing number order)
    f = h5py.File(args.dataset, "r")
    if args.filter_key is not None:
        logger.info(f"using filter key: {args.filter_key}")
        demos = [
            elem.decode("utf-8") for elem in np.array(f[f"mask/{args.filter_key}"])
        ]
        logger.info(f"Number of demos for filter key {args.filter_key}: {len(demos)}")
    else:
        demos = list(f["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    # maybe reduce the number of demonstrations to playback
    if args.n is not None:
        demos = demos[: args.n]

    # Track retry attempts per demo index to optionally cap retries
    retry_counts: dict[int, int] = {}

    ind = retrieve_new_index(process_num, current_work_array, work_queue, lock)
    while ind != -1:
        try:
            # print("Running {} index".format(ind))
            ep = demos[ind]

            # prepare initial state to reload from
            states = f[f"data/{ep}/states"][()]
            initial_state = {"states": states[0]}
            initial_state["model"] = f[f"data/{ep}"].attrs["model_file"]
            initial_state["ep_meta"] = f[f"data/{ep}"].attrs.get("ep_meta", None)

            # extract obs, rewards, dones
            actions = f[f"data/{ep}/actions"][()]

            traj = extract_trajectory(
                env=env,
                initial_state=initial_state,
                states=states,
                actions=actions,
                done_mode=args.done_mode,
                add_datagen_info=args.add_datagen_info,
            )

            # maybe copy reward or done signal from source file
            if args.copy_rewards:
                traj["rewards"] = f[f"data/{ep}/rewards"][()]
            if args.copy_dones:
                traj["dones"] = f[f"data/{ep}/dones"][()]

            ep_grp = f[f"data/{ep}"]

            states = ep_grp["states"][()]
            initial_state = {"states": states[0]}
            initial_state["model"] = ep_grp.attrs["model_file"]
            initial_state["ep_meta"] = ep_grp.attrs.get("ep_meta", None)

            # store transitions

            # IMPORTANT: keep name of group the same as source file, to make sure that filter keys are
            #            consistent as well
            # print("(process {}): ADD TO QUEUE index {}".format(process_num, ind))
            mul_queue.put([ep, traj, process_num])

            # reset retry count on success
            if ind in retry_counts:
                del retry_counts[ind]

            ind = retrieve_new_index(process_num, current_work_array, work_queue, lock)
        except Exception as e:
            logger.error("_" * 50)
            logger.error(f"Process {process_num}:")
            logger.error(f"Error processing demo index {ind}: {e}")
            logger.error(traceback.format_exc())
            logger.error("_" * 50)
            # If configured, stop immediately on error and propagate to wrapper
            if getattr(args, "stop_on_error", False):
                raise

            # Increment retry count and decide whether to continue retrying
            retry_counts[ind] = retry_counts.get(ind, 0) + 1
            max_retries = getattr(args, "max_retries", None)
            if (max_retries is not None) and (retry_counts[ind] > max_retries):
                # Exceeded retry budget: propagate error to wrapper
                raise

            # Otherwise, recreate the env and retry the same index
            del env
            env = EnvUtils.create_env_for_data_processing(  # when it errors, it like blows up the environment for some reason
                env_meta=env_meta,
                camera_names=args.camera_names,
                camera_height=args.camera_height,
                camera_width=args.camera_width,
                reward_shaping=args.shaped,
            )

    f.close()
    logger.info(f"Process {process_num} finished")
