"""Generate tc_tracks test set for development purposes."""

import os
import pickle
import random

import numpy as np

from klearn_tcyclone.climada.utils import get_TCTrack_dict
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


# Datasets
tc_tracks_dict = get_TCTrack_dict(
    basins=[flag_params["basin"]],
    time_step_h=flag_params["time_step_h"],
    year_range=flag_params["year_range"],
)
tc_tracks = tc_tracks_dict[flag_params["basin"]]

current_file_dir_path = os.path.dirname(os.path.abspath(__file__))
tc_tracks_test_set_path = "../data/tc_tracks_test_set/"
tc_tracks_test_set_name = f"tc_tracks_basin{flag_params['basin']}_yrange{flag_params['year_range']}_tstep{flag_params['time_step_h']}"

# Save data
os.makedirs(tc_tracks_test_set_path, exist_ok=True)
with open(
    os.path.join(tc_tracks_test_set_path, tc_tracks_test_set_name + ".pickle"), "wb"
) as f:
    pickle.dump(tc_tracks, f)
