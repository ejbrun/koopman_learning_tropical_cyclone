"""Functions for spectral analysis and consistency analysis."""

import numpy as np
import pandas as pd
from itertools import product
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


def get_top_k_ev_and_indices_below_zero(evals: NDArray, k: int) -> NDArray:
    assert np.ndim(evals) == 1, "'vec' must be a 1D array"
    assert k > 0, "k should be greater than 0"

    indices_below_zero_real = np.where(evals.real < 1.0)[0]
    evals_below_zero_real = evals[indices_below_zero_real]

    sort_perm = np.flip(np.argsort(evals_below_zero_real.real))  # descending order
    indices_k = indices_below_zero_real[sort_perm[:k]]
    evals_k = evals[indices_k]
    return indices_k, evals_k


def train_model(
    tc_tracks_train,
    tc_tracks_test,
    basin: str,
    context_length: int,
    time_lag: int,
    slide_by: int,
    model_config: dict | None = None,
    feature_list: list[str] | None = None,
    top_k: int = 5,
    **backend_kw,
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
    if tensor_context_train.data.shape[0] == 0:
        print(
            f"Train tensor context is empty, for time_lag={time_lag}, slide_by={slide_by} and context_length={context_length}."
        )
        return None, None, np.array(top_k * [None])
    else:
        tensor_context_train_transformed = standardize_TensorContextDataset(
            tensor_context_train,
            scaler,
            fit=True,
            periodic_shift=True,
            basin=basin,
            **backend_kw,
        )
        tensor_context_test_transformed = standardize_TensorContextDataset(
            tensor_context_test,
            scaler,
            fit=False,
            periodic_shift=True,
            basin=basin,
            **backend_kw,
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
    _, evals_k = get_top_k_ev_and_indices_below_zero(evals, k=top_k)
    # evals_k = get_top_k_ev_below_zero(evals, k=top_k)

    tscales = -1 / np.log(evals_k.real.clip(1e-8, 1))
    tscales_real = tscales * time_lag * slide_by

    return evals, rmse_onestep, tscales_real


def time_lag_scaling(
    tc_tracks,
    basin: str,
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
        evals, error, tscale = train_model(
            tc_tracks_train,
            tc_tracks_test,
            basin=basin,
            context_length=context_length,
            time_lag=time_lag,
            slide_by=1,
            model_config=model_config,
            feature_list=feature_list,
            top_k=top_k,
        )
        error_d[time_lag] = error
        tscale_d[time_lag] = tscale
        evals_d[time_lag] = evals

    return evals_d, error_d, tscale_d


def slide_by_scaling(
    tc_tracks,
    basin: str,
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
            basin=basin,
            context_length=context_length,
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


def get_df_evecs(context_dict, models, basins: list[str], top_k: int) -> pd.DataFrame:
    evecs_right = {}
    for basin_data, basin_model in product(basins, repeat=2):
        evals, evec_right = models[basin_model].eig(
            eval_right_on=context_dict[basin_data]
        )

        indices_k, _ = get_top_k_ev_and_indices_below_zero(evals, top_k)
        evec_right = evec_right[:, indices_k]

        evecs_right[(basin_data, basin_model)] = evec_right

    df_evecs = pd.DataFrame()
    for basin_data in basins:
        data_df = pd.concat(
            [
                pd.DataFrame(
                    evecs_right[(basin_data, basin_model)].real,
                    columns=[
                        f"evec_re_m{idx}_{i}"
                        for i in range(evecs_right[(basin_data, basin_model)].shape[1])
                    ],
                )
                for idx, basin_model in enumerate(basins)
            ]
            + [
                pd.DataFrame(
                    evecs_right[(basin_data, basin_model)].imag,
                    columns=[
                        f"evec_im_m{idx}_{i}"
                        for i in range(evecs_right[(basin_data, basin_model)].shape[1])
                    ],
                )
                for idx, basin_model in enumerate(basins)
            ],
            axis=1,
            ignore_index=False,
        )
        data_df["basin_data"] = [basin_data] * data_df.shape[0]
        df_evecs = pd.concat([df_evecs, data_df], axis=0, ignore_index=True)

    return df_evecs
