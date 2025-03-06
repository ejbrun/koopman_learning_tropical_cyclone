"""Comparing KNF and koopkernel sequence models."""

import os
import random
import torch

import numpy as np
from matplotlib import pyplot as plt
from klearn_tcyclone.models_utils import get_model_name
from klearn_tcyclone.training_utils.training_utils import (
    extend_by_default_flag_values,
)


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


# Set flag_params
# koopman_kernel_length_scale_arr = [1e-2, 1e-1, 1e0, 1e1]
koopman_kernel_length_scale_arr = [0.16, 0.18, 0.2, 0.22, 0.24]
# koopman_kernel_length_scale_arr = [0.06, 0.08, 0.1, 0.12, 0.14]
koopman_kernel_num_centers_arr = [1000, 2000]
tc_tracks_time_step = 3.0

flag_params = {
    # "year_range": [1980, 2021],
    "year_range": [2000, 2021],
}
flag_params = extend_by_default_flag_values(flag_params)
flag_params["batch_size"] = 32
# flag_params["num_epochs"] = 10
flag_params["num_epochs"] = 100
flag_params["train_output_length"] = 1
flag_params["test_output_length"] = flag_params["train_output_length"]
flag_params["input_length"] = 12
flag_params["input_dim"] = 6
flag_params["num_steps"] = 3
flag_params["context_length"] = (
    flag_params["input_length"] + flag_params["train_output_length"]
)
flag_params["time_step_h"] = tc_tracks_time_step
flag_params["basin"] = "NA"


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


str_date_time = "2025-03-01-18-35-05"
# str_date_time = "2025-03-01-17-43-51"
# str_date_time = "2025-03-01-17-27-17"
# str_date_time = "2025-03-01-15-35-04"
# str_date_time = "2025-03-01-13-16-15"
# str_date_time = "2025-03-01-11-23-05"
end_result = {}
epoch_results = {}
model_strings = ["koopkernelseq"]
for model_str in model_strings:
    print()
    print("===================================")
    print(f"Train {model_str}.")
    print("===================================")
    print()

    flag_params["model"] = model_str

    results_dir = os.path.join(
        current_file_dir_path,
        "../../train_models/training_results",
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

    for koopman_kernel_length_scale in koopman_kernel_length_scale_arr:
        end_result[koopman_kernel_length_scale] = {}
        for koopman_kernel_num_centers in koopman_kernel_num_centers_arr:
            flag_params["koopman_kernel_length_scale"] = koopman_kernel_length_scale
            flag_params["koopman_kernel_num_centers"] = koopman_kernel_num_centers
            model_name = get_model_name(flag_params)

            context_mode = "last_context"

            load_path = os.path.join(results_dir, "test_" + model_name + ".pt")
            # keys:
            # "test_preds"
            # "test_tgts"
            # "eval_score"
            # "train_rmses"
            # "eval_rmses"
            try:
                end_result[koopman_kernel_length_scale][koopman_kernel_num_centers] = (
                    torch.load(
                        load_path,
                    )
                )
                print(
                    f"{koopman_kernel_length_scale, koopman_kernel_num_centers} available."
                )
            except:
                pass

            # epoch_results[model_str] = {}
            # for epoch_index, epoch in enumerate(range(flag_params["num_epochs"])):

            #     # keys:
            #     # "test_preds"
            #     # "test_tgts"
            #     # "eval_score"
            #     epoch_results[model_str][epoch_index] = torch.load(os.path.join(results_dir, f"ep{epoch}_test" + model_name + ".pt"))


print()

fig, ax = plt.subplots()

for koopman_kernel_length_scale in koopman_kernel_length_scale_arr:
    for koopman_kernel_num_centers in koopman_kernel_num_centers_arr[:]:
        print(koopman_kernel_length_scale, koopman_kernel_num_centers)
        # koopman_kernel_length_scale = koopman_kernel_length_scale_arr[0]
        # koopman_kernel_num_centers = koopman_kernel_num_centers_arr[0]

        try:
            res = end_result[koopman_kernel_length_scale][koopman_kernel_num_centers]

            y = end_result[koopman_kernel_length_scale][koopman_kernel_num_centers][
                "train_rmses"
            ]
            num_epochs = len(y)
            # num_epochs = flag_params["num_epochs"]
            x = range(num_epochs)
            label = f"ls={koopman_kernel_length_scale}, nc={koopman_kernel_num_centers}"

            ax.plot(
                x,
                y,
                label=label,
            )
        except:
            print(
                f"{koopman_kernel_length_scale, koopman_kernel_num_centers} not available."
            )

ax.set_yscale("log")
ax.legend()
ax.set_yscale("log")

figure_path = f"koopkernel_sequence_{str_date_time}.pdf"
fig.savefig(figure_path)
