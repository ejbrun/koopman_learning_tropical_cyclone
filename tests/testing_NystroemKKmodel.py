"""Comparing KNF and koopkernel sequence models."""

import json

from klearn_tcyclone.koopkernel_seq2seq_utils import train_one_epoch, eval_one_epoch
import logging
import os
import pickle
import random
import time
from datetime import datetime

import numpy as np
import torch
from kooplearn.data import TensorContextDataset
from matplotlib.axes._axes import Axes
from sklearn.model_selection import train_test_split

# from torch.utils.tensorboard import SummaryWriter
# from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from xarray import Dataset

from klearn_tcyclone.climada.tc_tracks import TCTracks
from klearn_tcyclone.climada.utils import get_TCTrack_dict
from klearn_tcyclone.data_utils import (
    LinearScaler,
    standardized_context_dataset_from_TCTracks,
)
from klearn_tcyclone.KNF.modules.eval_metrics import RMSE_TCTracks
from klearn_tcyclone.knf_data_utils import TCTrackDataset
from klearn_tcyclone.koopkernel_seq2seq import (
    KoopKernelLoss,
    NystroemKoopKernelSequencer,
    RBFKernel,
)
from klearn_tcyclone.koopkernel_seq2seq_utils import (
    standardized_batched_context_from_TCTracks,
    train_koopkernel_seq2seq_model,
)
from klearn_tcyclone.models_utils import get_model_name
from klearn_tcyclone.plot_utils import plot_TCTrackDataset_item_2D
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


# Set flag_params
tc_tracks_time_step = 3.0

flag_params = {
    # "year_range": [1980, 2021],
    # "year_range": [2018, 2021],
    "year_range": [2018, 2021],
}
flag_params = extend_by_default_flag_values(flag_params)

# flag_params["context_mode"] = "no_context"
flag_params["context_mode"] = "full_context"
# flag_params["context_mode"] = "last_context"
# FIXME add context_mode to default flat_params parameters

flag_params["batch_size"] = 32
# flag_params["num_epochs"] = 10
flag_params["num_epochs"] = 100
flag_params["train_output_length"] = 1
flag_params["test_output_length"] = flag_params["train_output_length"]
if flag_params["context_mode"] == "no_context":
    flag_params["input_length"] = 4  # small input_length for context_mode = no_context
    flag_params["input_dim"] = 1
else:
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


# # Datasets
# tc_tracks_dict = get_TCTrack_dict(
#     basins=[flag_params["basin"]],
#     time_step_h=flag_params["time_step_h"],
#     year_range=flag_params["year_range"],
# )
# tc_tracks = tc_tracks_dict[flag_params["basin"]]

current_file_dir_path = os.path.dirname(os.path.abspath(__file__))
tc_tracks_test_set_path = "../data/tc_tracks_test_set/"
tc_tracks_test_set_name = f"tc_tracks_basin{flag_params['basin']}_yrange{flag_params['year_range']}_tstep{flag_params['time_step_h']}"

## Save data
# os.makedirs(tc_tracks_test_set_path, exist_ok=True)
# with open(
#     os.path.join(tc_tracks_test_set_path, tc_tracks_test_set_name + ".pickle"), "wb"
# ) as f:
#     pickle.dump(tc_tracks, f)

## Load data
with open(
    os.path.join(current_file_dir_path, tc_tracks_test_set_path, tc_tracks_test_set_name + ".pickle"), "rb"
) as f:
    tc_tracks = pickle.load(f)





flag_params["model"] = "koopkernelseq"


flag_params["koopman_kernel_length_scale"] = 0.1
flag_params["koopman_kernel_num_centers"] = 111
mask_koopman_operator = True

scaler = LinearScaler()
num_feats = len(feature_list)

rbf = RBFKernel(length_scale=flag_params["koopman_kernel_length_scale"])
koopkernelmodel = NystroemKoopKernelSequencer(
    kernel=rbf,
    input_dim=num_feats,
    input_length=flag_params["input_length"],
    output_length=flag_params["train_output_length"],
    output_dim=1,
    num_steps=1,
    num_nys_centers=flag_params["koopman_kernel_num_centers"],
    rng_seed=42,
    context_mode=flag_params["context_mode"],
    mask_koopman_operator=mask_koopman_operator,
)


tc_tracks_train, tc_tracks_test = train_test_split(
    tc_tracks.data, test_size=0.1, random_state=flag_params["seed"]
)

tensor_context_train_standardized = standardized_context_dataset_from_TCTracks(
    tc_tracks_train,
    feature_list=feature_list,
    scaler=scaler,
    context_length=flag_params["context_length"],
    time_lag=1,
    fit=True,
    periodic_shift=True,
    basin=flag_params["basin"],
    input_length=flag_params["input_length"],
    output_length=flag_params["train_output_length"],
)

koopkernelmodel._initialize_nystrom_data(tensor_context_train_standardized)
del tensor_context_train_standardized


# parameter_list = list(koopkernelmodel.parameters())

# # print(koopkernelmodel.koopman_blocks)

# for parameter in koopkernelmodel.parameters():
#     print(parameter)
#     print(parameter.shape)

total_params = sum(p.numel() for p in koopkernelmodel.parameters())
print(f"Number of parameters: {total_params}")

print("done")

split_valid_set = True
if split_valid_set:
    tc_tracks_train, tc_tracks_valid = train_test_split(
        tc_tracks_train, test_size=0.1, random_state=flag_params["seed"] + 1
    )
else:
    tc_tracks_valid = tc_tracks_test

tensor_context_inps_train, tensor_context_tgts_train = (
    standardized_batched_context_from_TCTracks(
        tc_tracks_train,
        flag_params["batch_size"],
        feature_list,
        scaler,
        context_length=flag_params["context_length"],
        time_lag=1,
        fit=True,
        periodic_shift=True,
        basin=flag_params["basin"],
        input_length=flag_params["input_length"],
        output_length=flag_params["train_output_length"],
    )
)
tensor_context_inps_valid, tensor_context_tgts_valid = (
    standardized_batched_context_from_TCTracks(
        tc_tracks_valid,
        flag_params["batch_size"],
        feature_list,
        scaler,
        context_length=flag_params["context_length"],
        time_lag=1,
        fit=False,
        periodic_shift=True,
        basin=flag_params["basin"],
        input_length=flag_params["input_length"],
        output_length=flag_params["train_output_length"],
    )
)
tensor_context_inps_test, tensor_context_tgts_test = (
    standardized_batched_context_from_TCTracks(
        tc_tracks_test,
        flag_params["batch_size"],
        feature_list,
        scaler,
        context_length=flag_params["context_length"],
        time_lag=1,
        fit=False,
        periodic_shift=True,
        basin=flag_params["basin"],
        input_length=flag_params["input_length"],
        output_length=flag_params["train_output_length"],
    )
)
del tc_tracks_train
del tc_tracks_valid
del tc_tracks_test






optimizer = torch.optim.Adam(koopkernelmodel.parameters(), lr=flag_params["learning_rate"])
loss_koopkernel = KoopKernelLoss(koopkernelmodel.nystrom_data_Y, koopkernelmodel._kernel)


num_epochs = 5

for epoch_index, epoch in enumerate(range(num_epochs)):
    print(epoch_index)
    start_time = time.time()

    train_rmse = train_one_epoch(
        koopkernelmodel,
        optimizer,
        loss_koopkernel,
        tensor_context_inps_train,
        tensor_context_tgts_train,
    )
    eval_rmse, _, _ = eval_one_epoch(
        koopkernelmodel,
        loss_koopkernel,
        tensor_context_inps_valid,
        tensor_context_tgts_valid,
    )
