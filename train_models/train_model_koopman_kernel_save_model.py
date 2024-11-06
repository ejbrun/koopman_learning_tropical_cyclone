"""Training of models."""

import logging
import os
import random

import numpy as np
import torch
from absl import app
from kooplearn.models import Kernel, NystroemKernel
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import train_test_split

from klearn_tcyclone.climada.tc_tracks import TCTracks
from klearn_tcyclone.data_utils import (
    LinearScaler,
    characteristic_length_scale_from_TCTracks,
)
from klearn_tcyclone.klearn_tcyclone import ModelBenchmark
from klearn_tcyclone.KNF.modules.eval_metrics import (
    RMSE_OneStep_TCTracks,
)
from klearn_tcyclone.training_utils.args import FLAGS
from klearn_tcyclone.training_utils.training_utils import set_flags


def main(argv):
    random.seed(FLAGS.seed)  # python random generator
    np.random.seed(FLAGS.seed)  # numpy random generator

    # parameters from flag
    flag_params = set_flags(FLAGS=FLAGS)

    # Logging and define save paths
    current_file_dir_path = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(
        current_file_dir_path,
        "training_results",
        "{}_yrange{}".format(
            flag_params["dataset"],
            "".join(map(str, flag_params["year_range"])),
        ),
        flag_params["model"],
    )
    logs_dir = os.path.join(
        current_file_dir_path,
        "logs",
        # flag_params["dataset"],
        "{}_yrange{}".format(
            flag_params["dataset"],
            "".join(map(str, flag_params["year_range"])),
        ),
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

    # Set remaining parameters
    # feature_list = ["lat", "lon"]
    feature_list = ["lat", "lon", "max_sustained_wind"]
    # feature_list = ["lat", "lon", "max_sustained_wind", "central_pressure"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device", device)

    scaler = LinearScaler()
    eval_metric = RMSE_OneStep_TCTracks

    # Datasets
    tc_tracks = TCTracks.from_ibtracs_netcdf(
        provider="usa",
        year_range=flag_params["year_range"],
        basin="NA",
        correct_pres=False,
    )

    # TODO also generate a validation set
    tc_tracks_train, tc_tracks_test = train_test_split(
        tc_tracks.data, test_size=0.1, random_state=flag_params["seed"]
    )

    model_name = "seed{}_kklnscale{}_kkrank{}_kkrdrank{}_kktkreg{}_kkncntr{}_kkntstops{}_kkcntlength{}".format(
        flag_params["seed"],
        flag_params["koopman_kernel_length_scale"],
        flag_params["koopman_kernel_rank"],
        flag_params["koopman_kernel_reduced_rank"],
        flag_params["tikhonov_reg"],
        flag_params["koopman_kernel_num_centers"],
        flag_params["koopman_kernel_num_train_stops"],
        flag_params["context_length"],
    )

    results_file_name = os.path.join(results_dir, model_name)

    # Instantiang the RBF kernel and its length scale as the
    # median of the pairwise distances of the dataset.
    # length_scale = characteristic_length_scale_from_TCTracks(
    #     tc_tracks_train, feature_list, quantile=0.2
    # )
    length_scale = flag_params["koopman_kernel_length_scale"]
    print("Length scale:", length_scale)

    kernel = RBF(length_scale=length_scale)
    model_params = {
        "kernel": kernel,
        "rank": flag_params["koopman_kernel_rank"],
        "reduced_rank": flag_params["koopman_kernel_reduced_rank"],
        "tikhonov_reg": flag_params["tikhonov_reg"],
        "rng_seed": flag_params["seed"],
    }

    if flag_params["model"] == "RRR":
        model_params["svd_solver"] = "arnoldi"
        model_class = Kernel
    elif flag_params["model"] == "Randomized_RRR":
        model_params["svd_solver"] = "randomized"
        model_class = Kernel
    elif flag_params["model"] == "Nystroem_RRR":
        model_params["svd_solver"] = "arnoldi"
        model_params["num_centers"] = flag_params["koopman_kernel_num_centers"]
        model_class = NystroemKernel

    model = model_class(**model_params)

    benchmark = ModelBenchmark(
        feature_list,
        tc_tracks_train,
        tc_tracks_test,
        scaler=scaler,
        context_length=flag_params["context_length"],
    )
    logger.info(benchmark.get_info())
    _ = benchmark.train_model(
        model=model,
        eval_metric=eval_metric,
        num_train_stops=flag_params["koopman_kernel_num_train_stops"],
        save_model="best",
        save_results=True,
        save_path=results_file_name,
    )


if __name__ == "__main__":
    app.run(main)
