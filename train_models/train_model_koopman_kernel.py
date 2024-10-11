"""Training of models."""

import os
import random
import time
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import torch
from absl import app
from kooplearn.models import Kernel, NystroemKernel
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils import data

from klearn_tcyclone.climada.tc_tracks import TCTracks
from klearn_tcyclone.data_utils import (
    LinearScaler,
    characteristic_length_scale_from_TCTracks,
    context_dataset_from_TCTracks,
    standardize_TensorContextDataset,
)
from klearn_tcyclone.klearn_tcyclone import ModelBenchmark
from klearn_tcyclone.KNF.modules.eval_metrics import RMSE_TCTracks
from klearn_tcyclone.KNF.modules.models import Koopman
from klearn_tcyclone.KNF.modules.train_utils import (
    eval_epoch_koopman,
    train_epoch_koopman,
)
from klearn_tcyclone.knf_data_utils import TCTrackDataset
from klearn_tcyclone.models_utils import predict_time_series
from klearn_tcyclone.performance_benchmark import timer
from klearn_tcyclone.training_utils.args import FLAGS
from klearn_tcyclone.training_utils.training_utils import set_flags


def main(argv):
    random.seed(FLAGS.seed)  # python random generator
    np.random.seed(FLAGS.seed)  # numpy random generator

    # parameters from flag
    flag_params = set_flags(FLAGS=FLAGS)

    # feature_list = ["lat", "lon"]
    # feature_list = ["lat", "lon", "central_pressure"]
    feature_list = ["lat", "lon", "max_sustained_wind"]
    # feature_list = ["lat", "lon", "max_sustained_wind", "central_pressure"]


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device", device)

    scaler = LinearScaler()
    eval_metric = RMSE_TCTracks

    # Datasets
    tc_tracks = TCTracks.from_ibtracs_netcdf(
        provider="usa",
        year_range=flag_params["year_range"],
        basin="NA",
        correct_pres=False,
    )

    tc_tracks_train, tc_tracks_test = train_test_split(tc_tracks.data, test_size=0.1)


    # tensor_context_train = context_dataset_from_TCTracks(
    #     tc_tracks_train, feature_list=feature_list, context_length=flag_params["context_length"]
    # )
    # tensor_context_test = context_dataset_from_TCTracks(
    #     tc_tracks_test, feature_list=feature_list, context_length=flag_params["context_length"]
    # )
    # tensor_context_train_transformed = standardize_TensorContextDataset(
    #     tensor_context_train,
    #     scaler,
    #     fit=True,
    # )
    # tensor_context_test_transformed = standardize_TensorContextDataset(
    #     tensor_context_test,
    #     scaler,
    #     fit=False,
    # )

    # contexts = {
    #     "train": tensor_context_train_transformed,
    #     "test": tensor_context_test_transformed,
    # }

    # for split, ds in contexts.items():
    #     print(
    #         f"{split.capitalize()} contexts have shape {ds.shape}: {len(ds)} contexts of length {ds.context_length} with {ds.shape[2]} features each"
    #     )

    model_folder_path = (
        "Koopman_"
        + str(flag_params["dataset"])
        + "_model{}_glc{}".format(
            flag_params["model"], flag_params["global_local_combination"]
        )
    )
    model_name = "seed{}_jumps{}_freq{}_bz{}_lr{}_decay{}_dim{}_inp{}_pred{}_num{}_latdim{}_RevIN{}_insnorm{}_regrank{}_globalK{}_contK{}".format(  # noqa: E501, UP032
        flag_params["seed"],
        flag_params["jumps"],
        flag_params["data_freq"],
        flag_params["batch_size"],
        flag_params["learning_rate"],
        flag_params["decay_rate"],
        flag_params["input_dim"],
        flag_params["input_length"],
        flag_params["train_output_length"],
        flag_params["num_steps"],
        flag_params["latent_dim"],
        flag_params["use_revin"],
        flag_params["use_instancenorm"],
        flag_params["regularize_rank"],
        flag_params["add_global_operator"],
        flag_params["add_control"],
    )

    current_file_dir_path = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(
        current_file_dir_path,
        "training_results",
        flag_params["dataset"],
        model_folder_path,
    )

    results_file_name = os.path.join(results_dir, model_name + ".pth")

    reduced_rank = flag_params["koopman_kernel_reduced_rank"]
    rank = flag_params["koopman_kernel_rank"]
    num_centers = flag_params["koopman_kernel_num_centers"]
    tikhonov_reg = flag_params["tikhonov_reg"]

    # Instantiang the RBF kernel and its length scale as the median of the pairwise distances of the dataset
    length_scale = characteristic_length_scale_from_TCTracks(
        tc_tracks_train, feature_list, quantile=0.2
    )
    length_scale = flag_params["koopman_kernel_length_scale"]
    print("Length scale:", length_scale)

    kernel = RBF(length_scale=length_scale)

    if flag_params["model"] == "RRR":
        svd_solver = "arnoldi"
        model_class = Kernel
    elif flag_params["model"] == "Randomized_RRR":
        svd_solver = "randomized"
        model_class = Kernel
    elif flag_params["model"] == "Nystroem_RRR":
        svd_solver = "arnoldi"
        model_class = NystroemKernel

    model = model_class(
        kernel=kernel,
        reduced_rank=reduced_rank,
        svd_solver=svd_solver,
        tikhonov_reg=flag_params["tikhonov_reg"],
        rank=flag_params["koopman_kernel_rank"],
        rng_seed=flag_params["seed"],
    )

    num_train_stops = 5
    benchmark = ModelBenchmark(
        model,
        feature_list,
        tc_tracks_train,
        tc_tracks_test,
        scaler=scaler,
        context_length=flag_params["context_length"],
    )
    print(benchmark.get_info())
    benchmark.train_model(num_train_stops=num_train_stops)


if __name__ == "__main__":
    app.run(main)
