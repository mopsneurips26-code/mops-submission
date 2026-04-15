import multiprocessing

import h5py
import numpy as np
from loguru import logger

import robocasa.utils.robomimic.robomimic_dataset_utils as DatasetUtils

from .rec_config import RecorderConfig
from .rec_worker import extract_multiple_trajectories
from .rec_writer import write_traj_to_file


def dataset_states_to_obs_multiprocessing(cfg: RecorderConfig) -> None:
    # create environment to use for data processing

    # output file in same directory as input file
    output_path = cfg.output_name

    logger.info(f"input file: {cfg.dataset}")
    logger.info(f"output file: {output_path}")

    f = h5py.File(cfg.dataset, "r")
    if cfg.filter_key is not None:
        logger.info(f"using filter key: {cfg.filter_key}")
        demos = [elem.decode("utf-8") for elem in np.array(f[f"mask/{cfg.filter_key}"])]
    else:
        demos = list(f["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    if cfg.n is not None:
        demos = demos[: cfg.n]

    num_demos = len(demos)
    f.close()

    DatasetUtils.get_env_metadata_from_dataset(dataset_path=cfg.dataset)
    num_processes = cfg.num_procs

    index = multiprocessing.Value("i", 0)
    lock = multiprocessing.Lock()
    total_samples_shared = multiprocessing.Value("i", 0)
    num_finished = multiprocessing.Value("i", 0)
    mul_queue = multiprocessing.Queue()
    stop_event = multiprocessing.Event()
    work_queue = multiprocessing.Queue()
    for index in range(num_demos):
        work_queue.put(index)
    current_work_array = multiprocessing.Array("i", num_processes)
    processes = []
    for i in range(num_processes):
        process = multiprocessing.Process(
            target=extract_multiple_trajectories,
            args=(
                i,
                current_work_array,
                work_queue,
                lock,
                cfg,
                num_finished,
                mul_queue,
            ),
        )
        processes.append(process)

    process1 = multiprocessing.Process(
        target=write_traj_to_file,
        args=(
            cfg,
            output_path,
            total_samples_shared,
            num_finished,
            num_processes,
            mul_queue,
            stop_event,
        ),
    )
    processes.append(process1)

    for process in processes:
        process.start()

    # Join workers first
    for process in processes[:-1]:
        process.join()

    # Signal writer to stop
    stop_event.set()

    # Join writer
    processes[-1].join()

    logger.info("Finished Multiprocessing")
    logger.info(f"Output file saved to: {output_path}")

    return
