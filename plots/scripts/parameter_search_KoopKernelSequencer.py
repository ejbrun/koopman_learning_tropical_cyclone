"""Comparing KNF and koopkernel sequence models."""

import os
import random
import time
from matplotlib import pyplot as plt
from datetime import datetime
from itertools import product

import numpy as np
import torch

from klearn_tcyclone.climada.utils import get_TCTrack_dict
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


# Set training settings
training_settings = {
    "koopman_kernel_length_scale": [0.06, 0.08, 0.1, 0.12, 0.14],
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


# # Datasets
# tc_tracks_dict = get_TCTrack_dict(
#     basins=[flag_params["basin"]],
#     time_step_h=flag_params["time_step_h"],
#     year_range=flag_params["year_range"],
# )
# tc_tracks = tc_tracks_dict[flag_params["basin"]]


# Logging and define save paths
current_file_dir_path = os.path.dirname(os.path.abspath(__file__))
# current_file_dir_path = os.getcwd()

res_dict = {}
model_strings = ["koopkernelseq"]
for model_str in model_strings:
    print()
    print("===================================")
    print(f"Load data for {model_str}.")
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
        "parameter_search",
    )

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
        if context_mode == "last_context":
            if mask_koopman_operator:
                print("Skip iteration.")
                continue

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

        model_name = get_model_name(flag_params)

        res_dict[
            (
                koopman_kernel_length_scale,
                koopman_kernel_num_centers,
                context_mode,
                mask_koopman_operator,
                mask_version,
                use_nystroem_context_window,
                output_length,
            )
        ] = torch.load(
            # {
            #     "test_preds": test_preds,
            #     "test_tgts": test_tgts,
            #     "eval_score": eval_metric(
            #         test_preds, test_tgts
            #     ),  # FIXME eval_metric() here is a tuple of four elements, why? Should be a single number.
            #     "train_rmses": all_train_rmses,
            #     "eval_rmses": all_eval_rmses,
            #     "training_runtime": training_runtime,
            # },
            os.path.join(results_dir, "test_" + model_name + ".pt"),
            weights_only=False,
        )


plot_settings = {
    "koopman_kernel_length_scale": [0.06, 0.08, 0.1, 0.12, 0.14],
    "koopman_kernel_num_centers": [1000],
    "context_mode": ["full_context", "last_context"],
    # "context_mode": ["no_context", "full_context", "last_context"],
    "mask_koopman_operator": [True, False],
    "mask_version": [1],
    # "mask_version": [0, 1],
    "use_nystroem_context_window": [False, True],
    "output_length": [1],
}

nrows = len(
    list(
        product(
            plot_settings["mask_koopman_operator"],
            plot_settings["use_nystroem_context_window"],
        )
    )
)

fig, ax = plt.subplots(nrows=nrows, ncols=2, layout="constrained")
fig.set_size_inches(10, 15)

koopman_kernel_num_centers = plot_settings["koopman_kernel_num_centers"][0]
mask_version = plot_settings["mask_version"][0]
output_length = plot_settings["output_length"][0]

# for (
#     koopman_kernel_length_scale,
#     koopman_kernel_num_centers,
#     context_mode,
#     mask_koopman_operator,
#     mask_version,
#     use_nystroem_context_window,
#     output_length,
# ) in product(*plot_settings.values()):
for idx_row, (mask_koopman_operator, use_nystroem_context_window) in enumerate(
    product(
        plot_settings["mask_koopman_operator"],
        plot_settings["use_nystroem_context_window"],
    )
):
    for idx_kkls, koopman_kernel_length_scale in enumerate(
        plot_settings["koopman_kernel_length_scale"]
    ):
        context_mode = plot_settings["context_mode"][0]
        conf = (
            koopman_kernel_length_scale,
            koopman_kernel_num_centers,
            context_mode,
            mask_koopman_operator,
            mask_version,
            use_nystroem_context_window,
            output_length,
        )
        y = res_dict[conf]["eval_rmses"]
        ymin = np.min(y)
        x = range(len(y))
        ax[idx_row, 0].plot(
            x, y, color=f"C{idx_kkls}", label=str(conf) + f", {ymin:.3f}"
        )
        y = res_dict[conf]["train_rmses"]
        x = range(len(y))
        ax[idx_row, 0].plot(x, y, color=f"C{idx_kkls}", linestyle="dashed")

        if not mask_koopman_operator:
            context_mode = plot_settings["context_mode"][1]
            conf = (
                koopman_kernel_length_scale,
                koopman_kernel_num_centers,
                context_mode,
                mask_koopman_operator,
                mask_version,
                use_nystroem_context_window,
                output_length,
            )
            y = res_dict[conf]["eval_rmses"]
            ymin = np.min(y)
            x = range(len(y))
            ax[idx_row, 1].plot(
                x, y, color=f"C{idx_kkls}", label=str(conf) + f", {ymin:.3f}"
            )
            y = res_dict[conf]["train_rmses"]
            x = range(len(y))
            ax[idx_row, 1].plot(x, y, color=f"C{idx_kkls}", linestyle="dashed")

    ax[idx_row, 0].set_ylim(9e-3, 1e0)
    ax[idx_row, 0].set_yscale("log")
    ax[idx_row, 0].legend()

    if not mask_koopman_operator:
        ax[idx_row, 1].set_ylim(9e-3, 2e0)
        ax[idx_row, 1].set_yscale("log")
        ax[idx_row, 1].legend()


# plot_settings = {
#     "koopman_kernel_length_scale": [0.06, 0.08, 0.1, 0.12, 0.14],
#     "koopman_kernel_num_centers": [1000],
#     "context_mode": ["last_context"],
#     # "context_mode": ["no_context", "full_context", "last_context"],
#     "mask_koopman_operator": [False],
#     "mask_version": [1],
#     # "mask_version": [0, 1],
#     "use_nystroem_context_window": [False, True],
#     "output_length": [1],
# }

# for (
#     koopman_kernel_length_scale,
#     koopman_kernel_num_centers,
#     context_mode,
#     mask_koopman_operator,
#     mask_version,
#     use_nystroem_context_window,
#     output_length,
# ) in product(*plot_settings.values()):

#     conf = (
#         koopman_kernel_length_scale,
#         koopman_kernel_num_centers,
#         context_mode,
#         mask_koopman_operator,
#         mask_version,
#         use_nystroem_context_window,
#         output_length,
#     )
#     y = res_dict[conf]["eval_rmses"]
#     # y = res_dict[conf]["train_rmses"]
#     x = range(len(y))

#     ax[1].plot(x, y, label=str(conf))

# ax[1].set_yscale("log")
# ax[1].legend()

fig.savefig("parameter_search_KoopKernelSequencer.pdf")
