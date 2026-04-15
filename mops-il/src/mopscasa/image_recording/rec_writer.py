import json
import time

import h5py
import numpy as np
from loguru import logger

import robocasa.utils.robomimic.robomimic_dataset_utils as DatasetUtils
import robocasa.utils.robomimic.robomimic_env_utils as EnvUtils


def write_traj_to_file(
    args, output_path, total_samples, total_run, processes, mul_queue, stop_event
) -> None:
    f = h5py.File(args.dataset, "r")
    f_out = h5py.File(output_path, "w")
    data_grp = f_out.create_group("data")
    start_time = time.time()
    num_processed = 0

    try:
        while not stop_event.is_set() or not mul_queue.empty():
            if not mul_queue.empty():
                num_processed = num_processed + 1
                item = mul_queue.get()
                ep = item[0]
                traj = item[1]
                process_num = item[2]
                try:
                    ep_data_grp = data_grp.create_group(ep)
                    ep_data_grp.create_dataset(
                        "actions", data=np.array(traj["actions"])
                    )
                    ep_data_grp.create_dataset("states", data=np.array(traj["states"]))
                    ep_data_grp.create_dataset(
                        "rewards", data=np.array(traj["rewards"])
                    )
                    ep_data_grp.create_dataset("dones", data=np.array(traj["dones"]))
                    # ep_data_grp.create_dataset(
                    #     "actions_abs", data=np.array(traj["actions_abs"])
                    # )
                    for k in traj["obs"]:
                        if args.no_compress:
                            ep_data_grp.create_dataset(
                                f"obs/{k}", data=np.array(traj["obs"][k])
                            )
                        else:
                            ep_data_grp.create_dataset(
                                f"obs/{k}",
                                data=np.array(traj["obs"][k]),
                                compression="gzip",
                            )
                        if args.include_next_obs:
                            if args.no_compress:
                                ep_data_grp.create_dataset(
                                    f"next_obs/{k}",
                                    data=np.array(traj["next_obs"][k]),
                                )
                            else:
                                ep_data_grp.create_dataset(
                                    f"next_obs/{k}",
                                    data=np.array(traj["next_obs"][k]),
                                    compression="gzip",
                                )

                    if "datagen_info" in traj:
                        for k in traj["datagen_info"]:
                            ep_data_grp.create_dataset(
                                f"datagen_info/{k}",
                                data=np.array(traj["datagen_info"][k]),
                            )

                    # copy action dict (if applicable)
                    if f"data/{ep}/action_dict" in f:
                        action_dict = f[f"data/{ep}/action_dict"]
                        for k in action_dict:
                            ep_data_grp.create_dataset(
                                f"action_dict/{k}",
                                data=np.array(action_dict[k][()]),
                            )

                    # episode metadata
                    ep_data_grp.attrs["model_file"] = traj["initial_state_dict"][
                        "model"
                    ]  # model xml for this episode
                    ep_data_grp.attrs["ep_meta"] = traj["initial_state_dict"][
                        "ep_meta"
                    ]  # ep meta data for this episode
                    # if "ep_meta" in f["data/{}".format(ep)].attrs:
                    #     ep_data_grp.attrs["ep_meta"] = f["data/{}".format(ep)].attrs["ep_meta"]
                    ep_data_grp.attrs["num_samples"] = traj["actions"].shape[
                        0
                    ]  # number of transitions in this episode

                    total_samples.value += traj["actions"].shape[0]
                except Exception as e:
                    print("++" * 50)
                    print(
                        f"Error at Process {process_num} on episode {ep} with \n\n {e}"
                    )
                    print("++" * 50)
                    raise Exception("Write out to file has failed")
                print(
                    "ep {}: wrote {} transitions to group {} at process {} with {} finished. Datagen rate: {:.2f} sec/demo".format(
                        num_processed,
                        ep_data_grp.attrs["num_samples"],
                        ep,
                        process_num,
                        total_run.value,
                        (time.time() - start_time) / num_processed,
                    )
                )
            else:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("Control C pressed. Closing File and ending \n\n\n\n\n\n\n")

    if "mask" in f:
        f.copy("mask", f_out)

    # global metadata
    data_grp.attrs["total"] = total_samples.value
    env_meta = DatasetUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
    if args.generative_textures:
        env_meta["env_kwargs"]["generative_textures"] = "100p"
    if args.randomize_cameras:
        env_meta["env_kwargs"]["randomize_cameras"] = True
    env = EnvUtils.create_env_for_data_processing(
        env_meta=env_meta,
        camera_names=args.camera_names,
        camera_height=args.camera_height,
        camera_width=args.camera_width,
        reward_shaping=args.shaped,
    )
    logger.info(f"total processes end {total_run.value}")
    data_grp.attrs["env_args"] = json.dumps(
        env.serialize(), indent=4
    )  # environment info
    logger.info(f"Wrote {total_samples.value} total samples to {output_path}")
    f_out.close()
    f.close()

    DatasetUtils.extract_action_dict(dataset=output_path)
    DatasetUtils.make_demo_ids_contiguous(dataset=output_path)
    for num_demos in [
        10,
        20,
        30,
        40,
        50,
        60,
        70,
        75,
        80,
        90,
        100,
        125,
        150,
        200,
        250,
        300,
        400,
        500,
        600,
        700,
        800,
        900,
        1000,
        1500,
        2000,
        2500,
        3000,
        4000,
        5000,
        10000,
    ]:
        DatasetUtils.filter_dataset_size(
            output_path,
            num_demos=num_demos,
        )

    logger.info("Writing has finished")

    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    logger.info(f"Time elapsed: {elapsed_time:.2f} seconds")
    return
