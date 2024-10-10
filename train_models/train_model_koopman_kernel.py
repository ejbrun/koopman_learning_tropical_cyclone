"""Training of models."""

import os
import random
import time

from klearn_tcyclone.klearn_tcyclone import ModelBenchmark

import numpy as np
import torch
from absl import app
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils import data

from klearn_tcyclone.climada.tc_tracks import TCTracks
from klearn_tcyclone.data_utils import (
    LinearScaler,
)
from klearn_tcyclone.KNF.modules.eval_metrics import RMSE_TCTracks
from klearn_tcyclone.KNF.modules.models import Koopman
from klearn_tcyclone.KNF.modules.train_utils import (
    eval_epoch_koopman,
    train_epoch_koopman,
)
from klearn_tcyclone.knf_data_utils import TCTrackDataset
from klearn_tcyclone.training_utils.args import FLAGS
from klearn_tcyclone.training_utils.training_utils import set_flags


from klearn_tcyclone.climada.tc_tracks import TCTracks
import numpy as np
from itertools import product

import matplotlib.pyplot as plt
from klearn_tcyclone.data_utils import context_dataset_from_TCTracks

from sklearn.model_selection import train_test_split
from klearn_tcyclone.data_utils import characteristic_length_scale_from_TCTracks
from kooplearn.models import Kernel, NystroemKernel
from sklearn.gaussian_process.kernels import RBF
from klearn_tcyclone.performance_benchmark import timer
from klearn_tcyclone.data_utils import standardize_TensorContextDataset, LinearScaler

from kooplearn.models import Kernel, NystroemKernel

from klearn_tcyclone.performance_benchmark import timer
from klearn_tcyclone.models_utils import predict_time_series

def main(argv):

    random.seed(FLAGS.seed)  # python random generator
    np.random.seed(FLAGS.seed)  # numpy random generator

    # parameters from flag
    flag_params = set_flags(FLAGS=FLAGS)

    # feature_list = ["lat", "lon"]
    # feature_list = ["lat", "lon", "central_pressure"]
    feature_list = ["lat", "lon", "max_sustained_wind"]
    # feature_list = ["lat", "lon", "max_sustained_wind", "central_pressure"]

    # these are not contained as flags
    output_dim = flag_params["input_dim"]
    num_feats = len(feature_list)
    learning_rate = flag_params["learning_rate"]
    # ---------------

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

    context_length = flag_params["context_length"]

    tensor_context_train = context_dataset_from_TCTracks(
        tc_tracks_train, feature_list=feature_list, context_length=context_length
    )
    tensor_context_test = context_dataset_from_TCTracks(
        tc_tracks_test, feature_list=feature_list, context_length=context_length
    )
    tensor_context_train_transformed = standardize_TensorContextDataset(
        tensor_context_train,
        scaler,
        fit=True,
    )
    tensor_context_test_transformed = standardize_TensorContextDataset(
        tensor_context_test,
        scaler,
        fit=False,
    )

    contexts = {
        "train": tensor_context_train_transformed,
        "test": tensor_context_test_transformed,
    }

    for split, ds in contexts.items():
        print(f"{split.capitalize()} contexts have shape {ds.shape}: {len(ds)} contexts of length {ds.context_length} with {ds.shape[2]} features each")


    model_folder_path = (
        "Koopman_"
        + str(flag_params["dataset"])
        + "_model{}_glc{}".format(flag_params["model"], flag_params["global_local_combination"])
    )
    model_name = (
        "seed{}_jumps{}_freq{}_bz{}_lr{}_decay{}_dim{}_inp{}_pred{}_num{}_latdim{}_RevIN{}_insnorm{}_regrank{}_globalK{}_contK{}".format(  # noqa: E501, UP032
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
    )

    current_file_dir_path = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(
        current_file_dir_path,
        "training_results",
        flag_params["dataset"],
        model_folder_path,
    )

    results_file_name = os.path.join(results_dir, model_name + ".pth")



    reduced_rank = True
    rank = 25
    num_centers = 250
    tikhonov_reg = 1e-6

    # Instantiang the RBF kernel and its length scale as the median of the pairwise distances of the dataset
    length_scale = characteristic_length_scale_from_TCTracks(tc_tracks_train, feature_list, quantile=0.2)
    length_scale = 50.0
    print("Length scale:", length_scale)

    kernel = RBF(length_scale=length_scale)
    # model = Kernel(kernel=kernel, reduced_rank=reduced_rank, tikhonov_reg=tikhonov_reg, rank = rank, svd_solver='arnoldi')
    model = Kernel(kernel=kernel, reduced_rank=reduced_rank, svd_solver='randomized', tikhonov_reg=tikhonov_reg, rank = rank, rng_seed=42)
    # model = NystroemKernel(kernel=kernel, reduced_rank=reduced_rank, svd_solver='arnoldi', tikhonov_reg=tikhonov_reg, rank = rank, num_centers=num_centers, rng_seed=42)
    # TODO set one of the models by a flag
    models = {
        'RRR': Kernel(kernel=kernel, reduced_rank=reduced_rank, tikhonov_reg=tikhonov_reg, rank = rank, svd_solver='arnoldi'),
        'Randomized-RRR': Kernel(kernel=kernel, reduced_rank=reduced_rank, svd_solver='randomized', tikhonov_reg=tikhonov_reg, rank = rank, rng_seed=42),
        'Nystroem-RRR': NystroemKernel(kernel=kernel, reduced_rank=reduced_rank, svd_solver='arnoldi', tikhonov_reg=tikhonov_reg, rank = rank, num_centers=num_centers, rng_seed=42),
    }


    # train_stops = np.logspace(2.5, 2.9, 5).astype(int)
    train_stops = np.logspace(2.5, 3.9, 5).astype(int)

    benchmark = ModelBenchmark(
        model,
        feature_list,
        tc_tracks_train,
        tc_tracks_test,
        scaler=scaler,
        context_length=context_length,
    )