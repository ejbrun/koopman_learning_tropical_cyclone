"""Comparing KNF and koopkernel sequence models."""

import logging
import os
import random

import numpy as np
from sklearn.model_selection import train_test_split

from klearn_tcyclone.climada.tc_tracks_tools import BASINS_SELECTION
from klearn_tcyclone.climada.utils import get_TCTrack_dict
from klearn_tcyclone.knf_model_utils import train_KNF_model
from klearn_tcyclone.koopkernel_sequencer_utils import (
    train_koopkernel_seq2seq_model,
)
from klearn_tcyclone.training_utils.training_utils import (
    extend_by_default_flag_values,
)

time_lag = 1
basins = BASINS_SELECTION
feature_list = [
    "lon",
    "lat",
    "max_sustained_wind",
    # "radius_max_wind",
    # "radius_oci",
    "central_pressure",
    "environmental_pressure",
]


# Set flag_params
flag_params = {
    # "year_range": [1980, 2021],
    "year_range": [2000, 2021],
    "koopman_kernel_num_centers": 1500,
    "time_step_h": 1.0,
}
flag_params = extend_by_default_flag_values(flag_params)
flag_params["batch_size"] = 32
flag_params["num_epochs"] = 30
flag_params["train_output_length"] = 1
flag_params["test_output_length"] = flag_params["train_output_length"]
flag_params["input_length"] = 24
flag_params["input_dim"] = 6
flag_params["num_steps"] = 3
flag_params["context_length"] = (
    flag_params["input_length"] + flag_params["train_output_length"]
)
flag_params["koopman_kernel_length_scale"] = 1e-1

assert (
    flag_params["context_length"]
    == flag_params["input_length"] + flag_params["train_output_length"]
)
if flag_params["input_length"] % flag_params["input_dim"] != 0:
    raise Exception("input_length must be divisible by input_dim")


random.seed(flag_params["seed"])  # python random generator
np.random.seed(flag_params["seed"])  # numpy random generator


# FIXME fix this
basins = ["NA"]
# Datasets
tc_tracks_dict = get_TCTrack_dict(
    basins=basins,
    time_step_h=flag_params["time_step_h"],
    year_range=flag_params["year_range"],
)
tc_tracks = tc_tracks_dict[basins[0]]


# Logging and define save paths
current_file_dir_path = os.path.dirname(os.path.abspath(__file__))
# current_file_dir_path = os.getcwd()


for model_str in ["KNF", "koopkernelseq"]:
    print()
    print("===================================")
    print(f"Train {model_str}.")
    print("===================================")
    print()

    flag_params["model"] = model_str

    results_dir = os.path.join(
        current_file_dir_path,
        "training_results",
        "comparison_knf_koopkernel_seq",
        "{}_yrange{}_basin{}".format(
            flag_params["dataset"],
            "".join(map(str, flag_params["year_range"])),
            flag_params["basin"],
        ),
        flag_params["model"],
    )
    logs_dir = os.path.join(
        current_file_dir_path,
        "logs",
        "comparison_knf_koopkernel_seq",
        "{}_yrange{}_basin{}".format(
            flag_params["dataset"],
            "".join(map(str, flag_params["year_range"])),
            flag_params["basin"],
        ),
        flag_params["model"],
    )
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(logs_dir, flag_params["model"] + ".log"),
        encoding="utf-8",
        filemode="a",
        format="{asctime} - {name} - {filename}:{lineno} - {levelname} - {message}",
        style="{",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    logger = logging.getLogger(flag_params["model"] + "_logger")
    logger.info(flag_params)

    if flag_params["model"] == "KNF":
        tc_tracks_train, tc_tracks_test = train_test_split(
            tc_tracks.data, test_size=0.1, random_state=flag_params["seed"]
        )
        model_knf = train_KNF_model(
            tc_tracks_train,
            tc_tracks_test,
            feature_list,
            flag_params=flag_params,
            logger=logger,
            results_dir=results_dir,
        )

    elif flag_params["model"] == "koopkernelseq":
        context_mode = "last_context"
        # FIXME other context modes

        model_koopkernelseq = train_koopkernel_seq2seq_model(
            tc_tracks,
            feature_list,
            flag_params,
            basin=flag_params["basin"],
            logger=logger,
            results_dir=results_dir,
            context_mode=context_mode,
        )

    else:
        raise Exception("Wrong model_str.")
