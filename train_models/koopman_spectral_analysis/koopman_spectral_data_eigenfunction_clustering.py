"""Trains UMAP model for eigenfunction clustering."""

import os
import pickle
from random import sample
from itertools import product

import umap
from sklearn.model_selection import train_test_split

from klearn_tcyclone.climada.tc_tracks_tools import BASINS_SELECTION
from klearn_tcyclone.climada.utils import get_TCTrack_dict
from klearn_tcyclone.data_utils import (
    load_model,
    standardized_context_dataset_from_TCTracks,
)
from klearn_tcyclone.kooplearn.spectral_analysis import (
    get_df_evecs,
)
from klearn_tcyclone.training_utils.training_utils import (
    extend_by_default_flag_values,
)

current_file_dir_path = os.getcwd()
path_training_results = os.path.join(
    current_file_dir_path,
    "../../train_models/",
    "training_results",
)

model_str = "Nystroem_RRR"


time_lag = 1
slide_by = 1
basins = BASINS_SELECTION
feature_list = [
    "lon",
    "lat",
    "max_sustained_wind",
    "radius_max_wind",
    "radius_oci",
    "central_pressure",
    "environmental_pressure",
]


flag_params = {
    "koopman_kernel_num_train_stops": 10,
    "year_range": [1980, 2021],
    "model": model_str,
    "koopman_kernel_length_scale": 10.0,
    "koopman_kernel_rank": 50,
    "koopman_kernel_num_centers": 800,
    "context_length": 16,
    "koopman_kernel_reduced_rank": False,
    "time_step_h": 1.0,
}
flag_params = extend_by_default_flag_values(flag_params)

model_config = {
    "length_scale": 10.0,
    "reduced_rank": False,
    "rank": 50,
    "num_centers": 600,
    "tikhonov_reg": 1e-8,
    "svd_solver": "arnoldi",
    "rng_seed": 42,
}


# Datasets
tc_tracks_dict = get_TCTrack_dict(
    basins=basins,
    time_step_h=flag_params["time_step_h"],
    year_range=flag_params["year_range"],
)
for basin in basins:
    tc_tracks_train, tc_tracks_test = train_test_split(
        tc_tracks_dict[basin].data, test_size=0.1, random_state=flag_params["seed"]
    )
    tc_tracks_dict[basin] = {
        "train": tc_tracks_train,
        "test": tc_tracks_test,
    }


# umap_n_neighbors = 600
# umap_min_dist = 0.2
# training_data_size = 30
top_k = 10

training_data_sizes = [50, 60, 70]
# training_data_sizes = [40, 60]
umap_n_neighborss = [600, 1200]
umap_min_dists = [0.1, 0.2]

for training_data_size, umap_n_neighbors, umap_min_dist in product(
    training_data_sizes, umap_n_neighborss, umap_min_dists
):
    tc_tracks_dict_sample = {}
    for basin in basins:
        tc_tracks_dict_sample[basin] = {
            "train": sample(tc_tracks_dict[basin]["train"], k=training_data_size),
        }

    models = {}
    scalers = {}
    for basin in basins:
        flag_params["basin"] = basin
        best_model_dict, _ = load_model(flag_params, path_training_results)
        models[basin] = best_model_dict["model"]
        scalers[basin] = best_model_dict["scaler"]

    contexts_sample = {}
    for basin in basins:
        tensor_context_train_standardized = standardized_context_dataset_from_TCTracks(
            tc_tracks_dict_sample[basin]["train"],
            feature_list=feature_list,
            scaler=scalers[basin],
            context_length=flag_params["context_length"],
            time_lag=time_lag,
            fit=False,
            periodic_shift=True,
            basin=basin,
        )

        contexts_sample[basin] = {
            "train": tensor_context_train_standardized,
        }

    df_evecs = get_df_evecs(
        {key: val["train"] for key, val in contexts_sample.items()},
        models=models,
        basins=basins,
        top_k=top_k,
    )

    umap_model = umap.UMAP(n_neighbors=umap_n_neighbors, min_dist=umap_min_dist)
    data = df_evecs.drop(columns=["basin_data"])
    embedding = umap_model.fit_transform(data.to_numpy())

    if model_config["reduced_rank"]:
        reduced_rank_str = "redrank"
    else:
        reduced_rank_str = ""

    file_name = "_".join(
        [
            f"cl{flag_params['context_length']}",
            f"tsteph{flag_params['time_step_h']}",
            f"nc{model_config['num_centers']}",
            f"tkreg{model_config['tikhonov_reg']}",
            reduced_rank_str,
            f"um_nneigh{umap_n_neighbors}",
            f"um_md{umap_min_dist}",
        ]
    )

    folder_name = "_".join(
        [
            "year_range",
            *map(str, flag_params["year_range"]),
            f"train_dsize{training_data_size}",
            f"topk{top_k}",
        ]
    )

    save_path = os.path.join(
        "../../data/",
        "koopman_spectral_analysis/",
        "eigenfunction_clustering/",
        folder_name,
    )
    # save_path = os.path.join(
    #     "E:/work/projects/kktrop_cyc/data/",
    #     "koopman_spectral_analysis/",
    #     "eigenfunction_clustering/",
    #     folder_name,
    # )

    os.makedirs(save_path, exist_ok=True)

    print("Save UMAP model.")
    with open(os.path.join(save_path, "umap_" + file_name + ".pickle"), "wb") as file:
        pickle.dump(umap_model, file)
