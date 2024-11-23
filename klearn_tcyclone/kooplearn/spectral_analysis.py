"""Functions for spectral analysis and consistency analysis."""

import numpy as np
from numpy.typing import NDArray
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import train_test_split

from klearn_tcyclone.data_utils import (
    LinearScaler,
    context_dataset_from_TCTracks,
    standardize_TensorContextDataset,
)
from klearn_tcyclone.kooplearn.models.nystroem import NystroemKernelCustom


def get_top_k_ev_below_zero(vec: NDArray, k: int) -> NDArray:
    assert np.ndim(vec) == 1, "'vec' must be a 1D array"
    assert k > 0, "k should be greater than 0"

    eig_below_zero_real = vec[vec.real < 1.0]
    sort_perm = np.flip(np.argsort(eig_below_zero_real.real))  # descending order
    indices = sort_perm[:k]
    values = eig_below_zero_real[indices]
    return values


# def train_model_evals(
#     tc_tracks_train,
#     tc_tracks_test,
#     context_length,
#     time_lag,
#     feature_list=["lon", "lat", "max_sustained_wind"],
#     top_k: int = 4,
# ):
#     tensor_context_train = context_dataset_from_TCTracks(
#         tc_tracks_train,
#         feature_list=feature_list,
#         context_length=context_length,
#         time_lag=time_lag,
#     )
#     tensor_context_test = context_dataset_from_TCTracks(
#         tc_tracks_test,
#         feature_list=feature_list,
#         context_length=context_length,
#         time_lag=time_lag,
#     )
#     contexts = {
#         "train": tensor_context_train,
#         "test": tensor_context_test,
#     }
#     scaler = LinearScaler()
#     tensor_context_train_transformed = standardize_TensorContextDataset(
#         tensor_context_train,
#         scaler,
#         fit=True,
#     )
#     tensor_context_test_transformed = standardize_TensorContextDataset(
#         tensor_context_test,
#         scaler,
#         fit=False,
#     )

#     contexts = {
#         "train": tensor_context_train_transformed,
#         "test": tensor_context_test_transformed,
#     }

#     ### Spectral analysis
#     # reduced_rank = True
#     reduced_rank = False
#     rank = 25
#     num_centers = 1300
#     tikhonov_reg = 1e-8

#     # Instantiang the RBF kernel and its length scale as the median of the pairwise distances of the dataset
#     length_scale = 10.0
#     kernel = RBF(length_scale=length_scale)
#     nys_rrr = NystroemKernel(
#         kernel=kernel,
#         reduced_rank=reduced_rank,
#         svd_solver="arnoldi",
#         tikhonov_reg=tikhonov_reg,
#         rank=rank,
#         num_centers=num_centers,
#         rng_seed=42,
#     )
#     nys_rrr = nys_rrr.fit(contexts["train"])

#     # X_pred = nys_rrr.predict(contexts["test"])  # Here we must pass the `X` part of the context
#     # X_true = contexts["test"].lookforward(nys_rrr.lookback_len)# This is the `Y` part of the test context
#     # rmse_onestep = np.sqrt(np.mean((X_pred - X_true)**2))

#     evals = nys_rrr.eig()
#     # evals = evals[topk(np.abs(evals), top_k).indices]

#     return evals


def train_model(
    tc_tracks_train,
    tc_tracks_test,
    context_length: int,
    time_lag: int,
    slide_by: int,
    model_config: dict | None = None,
    feature_list: list[str] | None = None,
    top_k: int = 5,
):
    if feature_list is None:
        feature_list = ["lon", "lat", "max_sustained_wind"]

    tensor_context_train = context_dataset_from_TCTracks(
        tc_tracks_train,
        feature_list=feature_list,
        context_length=context_length,
        time_lag=time_lag,
    )
    tensor_context_test = context_dataset_from_TCTracks(
        tc_tracks_test,
        feature_list=feature_list,
        context_length=context_length,
        time_lag=time_lag,
    )
    contexts = {
        "train": tensor_context_train,
        "test": tensor_context_test,
    }
    scaler = LinearScaler()
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

    ### Spectral analysis
    if model_config is None:
        model_config = {}
        model_config["length_scale"] = 10.0
        model_config["reduced_rank"] = False
        model_config["rank"] = 25
        model_config["num_centers"] = 600
        model_config["tikhonov_reg"] = 1e-8
        model_config["svd_solver"] = "arnoldi"
        model_config["rng_seed"] = 42

    # Instantiang the RBF kernel and its length scale as the median of the pairwise distances of the dataset
    kernel = RBF(length_scale=model_config["length_scale"])
    nys_rrr = NystroemKernelCustom(
        kernel=kernel,
        reduced_rank=model_config["reduced_rank"],
        svd_solver=model_config["svd_solver"],
        tikhonov_reg=model_config["tikhonov_reg"],
        rank=model_config["rank"],
        num_centers=model_config["num_centers"],
        rng_seed=model_config["rng_seed"],
    )
    nys_rrr = nys_rrr.fit(contexts["train"], slide_by=slide_by)

    X_pred = nys_rrr.predict(
        contexts["test"]
    )  # Here we must pass the `X` part of the context
    # FIXME the following is probably not the correct Y part if slide_by != 1
    X_true = contexts["test"].lookforward(
        nys_rrr.lookback_len
    )  # This is the `Y` part of the test context
    rmse_onestep = np.sqrt(np.mean((X_pred - X_true) ** 2))

    evals = nys_rrr.eig()
    evals_top_k = get_top_k_ev_below_zero(evals, k=top_k)
    # evals_top_k = evals[topk(np.abs(evals.real), top_k).indices]
    # evals_top_k = evals[topk(np.abs(evals), top_k).indices]

    tscales = -1 / np.log(evals_top_k.real.clip(1e-8, 1))
    tscales_real = tscales * time_lag * slide_by

    return evals, rmse_onestep, tscales_real


# def time_lag_scaling_evals(
#     tc_tracks,
#     time_lags: list[int],
#     context_length: int = 2,
#     model_config: dict | None = None,
#     feature_list: list[str] = ["lon", "lat", "max_sustained_wind"],
# ):
#     # # Load TCTracks and resample on equal time step.
#     # tc_tracks = TCTracks.from_ibtracs_netcdf(provider='official', year_range=(2000, 2021), basin=basin)
#     # print('Number of tracks:', tc_tracks.size)
#     # tc_tracks.equal_timestep(time_step_h=time_step_h)
#     # assert check_time_steps_TCTracks(tc_tracks, time_step_h)

#     tc_tracks_train, tc_tracks_test = train_test_split(tc_tracks.data, test_size=0.1)

#     # Train model for varying time_lags.
#     evals_d = {}
#     for time_lag in time_lags:
#         print(f"Train with time_lag {time_lag}.")
#         try:
#             evals, _, _ = train_model(
#                 tc_tracks_train,
#                 tc_tracks_test,
#                 context_length,
#                 time_lag=time_lag,
#                 slide_by=1,
#                 model_config=model_config,
#                 feature_list=feature_list,
#             )
#         except:
#             evals = (
#                 None,
#                 None,
#             )
#             print("Not enough time points in some trajectories for given time_lag.")
#         evals_d[time_lag] = evals

#     return evals_d


def time_lag_scaling(
    tc_tracks,
    time_lags: list[int],
    context_length: int = 2,
    top_k: int = 5,
    model_config: dict | None = None,
    feature_list: list[str] | None = None,
) -> tuple[dict, dict, dict]:
    if feature_list is None:
        feature_list = ["lon", "lat", "max_sustained_wind"]

    tc_tracks_train, tc_tracks_test = train_test_split(tc_tracks.data, test_size=0.1)

    # Train model for varying time_lags.
    evals_d = {}
    error_d = {}
    tscale_d = {}
    for time_lag in time_lags:
        print(f"Train with time_lag {time_lag}.")
        # try:
        evals, error, tscale = train_model(
            tc_tracks_train,
            tc_tracks_test,
            context_length,
            time_lag=time_lag,
            slide_by=1,
            model_config=model_config,
            feature_list=feature_list,
            top_k=top_k,
        )
        # except:
        #     evals = (None, None)
        #     error, tscale = None, np.array(top_k * [None])
        #     print("Not enough time points in some trajectories for given time_lag.")
        error_d[time_lag] = error
        tscale_d[time_lag] = tscale
        evals_d[time_lag] = evals

    return evals_d, error_d, tscale_d


def slide_by_scaling(
    tc_tracks,
    slide_bys: list[int],
    context_length: int = 2,
    top_k: int = 5,
    model_config: dict | None = None,
    feature_list: list[str] | None = None,
) -> tuple[dict, dict, dict]:
    if feature_list is None:
        feature_list = ["lon", "lat", "max_sustained_wind"]

    tc_tracks_train, tc_tracks_test = train_test_split(tc_tracks.data, test_size=0.1)

    # Train model for varying slide_bys.
    evals_d = {}
    error_d = {}
    tscale_d = {}
    for slide_by in slide_bys:
        print(f"Train with slide_by {slide_by}.")
        # try:
        evals, error, tscale = train_model(
            tc_tracks_train,
            tc_tracks_test,
            context_length,
            time_lag=1,
            slide_by=slide_by,
            model_config=model_config,
            feature_list=feature_list,
            top_k=top_k,
        )
        # except:
        #     evals = (None, None)
        #     error, tscale = None, np.array(top_k * [None])
        #     print("Not enough time points in some trajectories for given slide_by.")
        error_d[slide_by] = error
        tscale_d[slide_by] = tscale
        evals_d[slide_by] = evals

    return evals_d, error_d, tscale_d
