"""Comparing KNF and koopkernel sequence models."""

import os
import random

import numpy as np
import torch
from matplotlib import pyplot as plt

from klearn_tcyclone.models_utils import get_model_name, get_model_name_old
from klearn_tcyclone.climada.tc_tracks_tools import BASINS_SELECTION
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
    # "koopman_kernel_length_scale": 10.0,
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

assert (
    flag_params["context_length"]
    == flag_params["input_length"] + flag_params["train_output_length"]
)
if flag_params["input_length"] % flag_params["input_dim"] != 0:
    raise Exception("input_length must be divisible by input_dim")


random.seed(flag_params["seed"])  # python random generator
np.random.seed(flag_params["seed"])  # numpy random generator


# Logging and define save paths
current_file_dir_path = os.path.dirname(os.path.abspath(__file__))
# current_file_dir_path = os.getcwd()


basins = ["NA"]
model_strings = ["KNF", "koopkernelseq"]
flag_params["basin"] = "EP"


end_result = {}
epoch_results = {}
for model_str in model_strings:
    flag_params["model"] = model_str
    results_dir = os.path.join(
        current_file_dir_path,
        "../../train_models/training_results",
        "comparison_knf_koopkernel_seq",
        "{}_yrange{}_basin{}".format(
            flag_params["dataset"],
            "".join(map(str, flag_params["year_range"])),
            flag_params["basin"],
        ),
        flag_params["model"],
    )
    model_name = get_model_name_old(flag_params)
    results_file_name = os.path.join(results_dir, model_name)

    # keys:
    # "test_preds"
    # "test_tgts"
    # "eval_score"
    # "train_rmses"
    # "eval_rmses"
    end_result[model_str] = torch.load(
        os.path.join(results_dir, "test_" + model_name + ".pt"),
    )

    # epoch_results[model_str] = {}
    # for epoch_index, epoch in enumerate(range(flag_params["num_epochs"])):

    #     # keys:
    #     # "test_preds"
    #     # "test_tgts"
    #     # "eval_score"
    #     epoch_results[model_str][epoch_index] = torch.load(os.path.join(results_dir, f"ep{epoch}_test" + model_name + ".pt"))


fig, ax = plt.subplots()

ax.plot(range(flag_params["num_epochs"]), end_result["KNF"]["train_rmses"], label="KNF")
ax.plot(
    range(flag_params["num_epochs"]),
    end_result["koopkernelseq"]["train_rmses"],
    label="koopkernelseq",
)
ax.legend()
ax.set_yscale("log")

fig.savefig("comparison_knf_koopkernel_seq.pdf")
