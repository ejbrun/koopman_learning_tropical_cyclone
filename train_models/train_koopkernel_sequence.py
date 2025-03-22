"""Comparing KNF and koopkernel sequence models."""

import logging
import os
import random
import time
from datetime import datetime
from itertools import product

import numpy as np

from klearn_tcyclone.climada.utils import get_TCTrack_dict
from klearn_tcyclone.koopkernel_sequencer_utils import (
    train_koopkernel_seq2seq_model,
)
from klearn_tcyclone.training_utils.training_utils import (
    extend_by_default_flag_values,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


time_lag = 1
feature_list = [
    "lon",
    "lat",
    "max_sustained_wind",
    # "radius_max_wind",
    # "radius_oci",
    "central_pressure",
    "environmental_pressure",
]


# Set training settings
training_settings = {
    # "koopman_kernel_length_scale": [0.06, 0.08, 0.1, 0.12, 0.14],
    "koopman_kernel_length_scale": [0.16, 0.18, 0.20, 0.22, 0.24],
    "koopman_kernel_num_centers": [1000],
    "context_mode": ["full_context", "last_context"],
    # "context_mode": ["no_context", "full_context", "last_context"],
    "mask_koopman_operator": [True, False],
    "mask_version": [1],
    # "mask_version": [0, 1],
    "use_nystroem_context_window": [False, True],
    "output_length": [1],
}

# koopman_kernel_length_scale_arr = [0.16, 0.18, 0.2, 0.22, 0.24]
# koopman_kernel_length_scale_arr = [0.06, 0.08, 0.1, 0.12, 0.14]
# koopman_kernel_length_scale_arr = [1e-2, 1e-1, 1e0, 1e1]
# koopman_kernel_num_centers_arr = [100, 200]
# koopman_kernel_num_centers_arr = [1000]
# koopman_kernel_num_centers_arr = [1000, 2000]
tc_tracks_time_step = 3.0


flag_params = {
    # "year_range": [1980, 2021],
    # "year_range": [2018, 2021],
    "year_range": [2000, 2021],
}
flag_params = extend_by_default_flag_values(flag_params)
# FIXME add context_mode, mask_koopman_operator, mask_version,
#   use_nystroem_context_window to default flat_params parameters

flag_params["batch_size"] = 32
# flag_params["num_epochs"] = 10
flag_params["num_epochs"] = 100

flag_params["num_steps"] = 1
# FIXME Remove num_steps, not accessed for the KooplearnSequencer.
flag_params["time_step_h"] = tc_tracks_time_step
flag_params["basin"] = "NA"


random.seed(flag_params["seed"])  # python random generator
np.random.seed(flag_params["seed"])  # numpy random generator


# Datasets
tc_tracks_dict = get_TCTrack_dict(
    basins=[flag_params["basin"]],
    time_step_h=flag_params["time_step_h"],
    year_range=flag_params["year_range"],
)
tc_tracks = tc_tracks_dict[flag_params["basin"]]


# Logging and define save paths
current_file_dir_path = os.path.dirname(os.path.abspath(__file__))
# current_file_dir_path = os.getcwd()

model_strings = ["koopkernelseq"]
for model_str in model_strings:
    print()
    print("===================================")
    print(f"Train {model_str}.")
    print("===================================")
    print()

    date_time = datetime.fromtimestamp(time.time())
    str_date_time = date_time.strftime("%Y-%m-%d-%H-%M-%S")
    flag_params["model"] = model_str

    results_dir = os.path.join(
        current_file_dir_path,
        "training_results",
        "comparison_knf_koopkernel_seq",
        "{}_yrange{}_basin{}_tstep{}".format(
            flag_params["dataset"],
            "".join(map(str, flag_params["year_range"])),
            flag_params["basin"],
            flag_params["time_step_h"],
        ),
        flag_params["model"],
        str_date_time,
    )
    logs_dir = os.path.join(
        current_file_dir_path,
        "logs",
        "comparison_knf_koopkernel_seq",
        "{}_yrange{}_basin{}_tstep{}".format(
            flag_params["dataset"],
            "".join(map(str, flag_params["year_range"])),
            flag_params["basin"],
            flag_params["time_step_h"],
        ),
        flag_params["model"],
    )
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    log_file_name = os.path.join(logs_dir, flag_params["model"] + ".log")

    logFormatter = logging.Formatter(
        "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
        # style="{",
        # datefmt="%Y-%m-%d %H:%M:%S",
    )
    fileHandler = logging.FileHandler(log_file_name)
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)
    # consoleHandler = logging.StreamHandler()
    # consoleHandler.setFormatter(logFormatter)
    # logger.addHandler(consoleHandler)

    # logging.basicConfig(
    #     filename=os.path.join(logs_dir, flag_params["model"] + ".log"),
    #     encoding="utf-8",
    #     filemode="a",
    #     format="{asctime} - {name} - {filename}:{lineno} - {levelname} - {message}",
    #     style="{",
    #     datefmt="%Y-%m-%d %H:%M:%S",
    #     force=True,
    # )
    # logger = logging.getLogger(flag_params["model"] + "_logger")

    for (
        koopman_kernel_length_scale,
        koopman_kernel_num_centers,
        context_mode,
        mask_koopman_operator,
        mask_version,
        use_nystroem_context_window,
        output_length,
    ) in product(*training_settings.values()):
        print()
        print()
        print("=============================================================")
        print("Iteration:")
        print(
            koopman_kernel_length_scale,
            koopman_kernel_num_centers,
            context_mode,
            mask_koopman_operator,
            mask_version,
            use_nystroem_context_window,
            output_length,
        )
        # if not mask_koopman_operator:
        #     if mask_version == 0:
        #         print("Skip iteration.")
        #         continue
        if context_mode == "last_context":
            if mask_koopman_operator:
                print("Skip iteration.")
                continue
            # if mask_version == 0:
            #     print("Skip iteration.")
            #     continue

        flag_params["train_output_length"] = output_length
        flag_params["test_output_length"] = flag_params["train_output_length"]

        flag_params["koopman_kernel_length_scale"] = koopman_kernel_length_scale
        flag_params["koopman_kernel_num_centers"] = koopman_kernel_num_centers
        flag_params["context_mode"] = context_mode
        flag_params["mask_koopman_operator"] = mask_koopman_operator
        flag_params["mask_version"] = mask_version
        flag_params["use_nystroem_context_window"] = use_nystroem_context_window
        if flag_params["context_mode"] == "no_context":
            flag_params["input_length"] = (
                4  # small input_length for context_mode = no_context
            )
            flag_params["input_dim"] = 1
        else:
            flag_params["input_length"] = 12
            flag_params["input_dim"] = 4
        flag_params["context_length"] = (
            flag_params["input_length"] + flag_params["train_output_length"]
        )
        assert (
            flag_params["context_length"]
            == flag_params["input_length"] + flag_params["train_output_length"]
        )
        if flag_params["input_length"] % flag_params["input_dim"] != 0:
            raise Exception("input_length must be divisible by input_dim")

        logger.info(flag_params)

        _, _ = train_koopkernel_seq2seq_model(
            tc_tracks,
            feature_list,
            flag_params,
            basin=flag_params["basin"],
            log_file_handler=fileHandler,
            results_dir=results_dir,
        )
