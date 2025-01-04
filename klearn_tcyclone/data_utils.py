"""Utils for data processing."""

import glob
import os
from collections.abc import Callable
from typing import Union

import numpy as np
import pandas as pd
import torch
import xarray as xr
from kooplearn.data import TensorContextDataset, TrajectoryContextDataset
from numpy.typing import NDArray
from scipy.spatial.distance import pdist
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from xarray import Dataset

from klearn_tcyclone.climada.tc_tracks import TCTracks
from klearn_tcyclone.climada.tc_tracks_tools import BASIN_SHIFT_SWITCHER


def data_array_list_from_TCTracks(
    tc_tracks: TCTracks | list[Dataset],
    feature_list: list[str],
    # tc_tracks: Union[TCTracks, list[Dataset]], feature_list: list[str]
) -> list[NDArray]:
    """Create data array list from TCTracks.

    Args:
        tc_tracks (Union[TCTracks, list[Dataset]]): TCTRacks dataset.
        feature_list (list[str]): List of features that are extracted.

    Returns:
        list[NDArray]: List of data arrays.
    """
    if isinstance(tc_tracks, TCTracks):
        tc_data = tc_tracks.data
    elif isinstance(tc_tracks, list):
        tc_data = tc_tracks
    else:
        raise Exception("tc_track is of wrong type.")

    data_array_list = []
    for tc in tc_data:
        data_array = np.array([tc[key].data for key in feature_list]).transpose()

        data_array_list.append(data_array)

    return data_array_list


def context_dataset_from_TCTracks(
    tc_tracks: TCTracks,
    feature_list: list[str],
    context_length: int = 2,
    time_lag: int = 1,
    backend: str = "auto",
    verbose: int = 0,
    **backend_kw,
) -> TensorContextDataset:
    """Generate context dataset from TCTRacks.

    Args:
        tc_tracks (TCTracks): _description_
        feature_list (list[str]): _description_
        context_length (int, optional): Length of the context window. Default to ``2``.
        time_lag (int, optional): Time lag, i.e. stride, between successive context
            windows. Default to ``1``.
        backend (str, optional): Specifies the backend to be used
            (``'numpy'``, ``'torch'``). If set to ``'auto'``, will use the same backend
            of the trajectory. Default to ``'auto'``.
        verbose (int, optional): Sets verbosity level. Default to 0 (no output).
        **backend_kw (dict, optional): Keyword arguments to pass to the backend.
            For example, if ``'torch'``, it is possible to specify the device of the
            tensor.

    Returns:
        TensorContextDataset: _description_
    """
    data_array_list = data_array_list_from_TCTracks(
        tc_tracks=tc_tracks, feature_list=feature_list
    )
    context_data_array = np.empty((0, context_length, len(feature_list)))
    for idx, data_array in enumerate(data_array_list):
        if data_array.shape[0] > context_length * time_lag:
            context_data_array = np.concatenate(
                [
                    context_data_array,
                    TrajectoryContextDataset(
                        data_array, context_length, time_lag, backend, **backend_kw
                    ).data,
                ],
                axis=0,
            )
        else:
            if verbose > 0:
                print(
                    f"""Data entry {idx} has been removed since it is shorter than the 
                    context_length {context_length} times time_lag {time_lag}."""
                )

    tensor_context_dataset = TensorContextDataset(
        context_data_array, backend, **backend_kw
    )
    return tensor_context_dataset


def TCTracks_from_TensorContextDataset(
    tensor_context: TensorContextDataset, feature_list: list[str]
):
    new_xarr_list = []
    len_time_series = tensor_context.shape[1]
    n_features = tensor_context.shape[2]
    assert len(feature_list) == n_features
    is_admissible = (
        "lat" in feature_list
        and "lon" in feature_list
        and "max_sustained_wind" in feature_list
    )
    assert (
        is_admissible
    ), "'lon', 'lat' and 'max_sustained_wind' must be in feature_list."

    for t_context in tensor_context:
        time_data = np.linspace(0, 1, len_time_series)
        data_helper = {}
        data_helper["time"] = time_data
        for idx, key in enumerate(feature_list):
            data_helper[key] = t_context[:, idx]

        data_vars = {
            # 'radius_max_wind': ('time', track_ds.rmw.data),
            # 'radius_oci': ('time', track_ds.roci.data),
            "max_sustained_wind": ("time", data_helper["max_sustained_wind"]),
            # 'central_pressure': ('time', track_ds.pres.data),
            # 'environmental_pressure': ('time', track_ds.poci.data),
        }
        coords = {
            "time": ("time", data_helper["time"]),
            "lat": ("time", data_helper["lat"]),
            "lon": ("time", data_helper["lon"]),
        }
        attrs = {
            # 'max_sustained_wind_unit': 'kn',
            # 'central_pressure_unit': 'mb',
            "orig_event_flag": True,
            # 'data_provider': provider_str,
            # 'category': category[i_track],
        }

        xarr = xr.Dataset(data_vars, coords=coords, attrs=attrs)
        new_xarr_list.append(xarr)

    return TCTracks(new_xarr_list)


def generate_reduced_tc_tracks(tc_tracks: TCTracks) -> TCTracks:
    new_xarr_list = []
    for tc_track in tc_tracks.data:
        time_data = tc_track["time"].data
        lat_data = tc_track["lat"].data
        lon_data = tc_track["lon"].data
        max_sustained_wind_data = tc_track["max_sustained_wind"].data

        data_vars = {
            # 'radius_max_wind': ('time', track_ds.rmw.data),
            # 'radius_oci': ('time', track_ds.roci.data),
            "max_sustained_wind": ("time", max_sustained_wind_data),
            # 'central_pressure': ('time', track_ds.pres.data),
            # 'environmental_pressure': ('time', track_ds.poci.data),
        }
        coords = {
            "time": ("time", time_data),
            "lat": ("time", lat_data),
            "lon": ("time", lon_data),
        }
        attrs = {
            # 'max_sustained_wind_unit': 'kn',
            # 'central_pressure_unit': 'mb',
            "orig_event_flag": True,
            # 'data_provider': provider_str,
            # 'category': category[i_track],
        }

        xarr = xr.Dataset(data_vars, coords=coords, attrs=attrs)
        # print(type(tc_track), tc_track.orig_event_flag)
        # if tc_track.orig_event_flag:
        #     xarr.orig_event_flat = True
        # else:
        #     xarr.orig_event_flat = False
        new_xarr_list.append(xarr)

    return TCTracks(new_xarr_list)


def characteristic_length_scale_from_TCTracks(
    tc_tracks: Union[TCTracks, list], feature_list: list[str], quantile: int = 0.5
) -> float:
    """Compute characteristic length scale from TCTracks data.

    Args:
        tc_tracks (Union[TCTracks, list]): _description_
        feature_list (list[str]): _description_
        quantile (int, optional): _description_. Defaults to 0.5.

    Returns:
        float: _description_
    """
    data_array_list = data_array_list_from_TCTracks(tc_tracks, feature_list)
    quantiles = [
        np.quantile(pdist(data_array), quantile) for data_array in data_array_list
    ]
    mean_quantile = np.mean(quantiles)
    return mean_quantile


def linear_transform(
    min_vec: NDArray, max_vec: NDArray, target_min_vec: NDArray, target_max_vec: NDArray
) -> Callable:
    """Generate a multidimensional linear transformation.

    The transformation mapsp the rectangle with corners given by min_vec and max_vec to
    the rectangle with corners given by target_min_vec and target_max_vec.

    Args:
        min_vec (NDArray): Lower corner of the input rectangle.
        max_vec (NDArray): Upper corner of the intput rectangle.
        target_min_vec (NDArray): Lower corner of the target rectangle.
        target_max_vec (NDArray): Upper corner of the target rectangle.

    Returns:
        Callable: The linear transformation as a Callable.
    """

    def fun(data):
        scaling_factor = (target_max_vec - target_min_vec) / (max_vec - min_vec)
        diffs_to_min = data - min_vec
        scaled_diffs_to_min_vec = scaling_factor * diffs_to_min
        return scaled_diffs_to_min_vec + target_min_vec

    return fun


class LinearScalerError(Exception):
    """A custom exception used to report errors in use of Timer class"""


class LinearScaler:
    """MinMaxScaler von sklearn is very similar.

    However, MinMaxScaler does not allow to scale different dimensions to different
    intervals. This functionality might be useful down the road.
    """

    def __init__(self, target_min_vec=None, target_max_vec=None):
        self.target_min_vec = target_min_vec
        self.target_max_vec = target_max_vec
        self.min_vec = None
        self.max_vec = None

    def _linear_transform(self, data):
        scaling_factor = (self.target_max_vec - self.target_min_vec) / (
            self.max_vec - self.min_vec
        )
        diffs_to_min = data - self.min_vec
        scaled_diffs_to_min_vec = scaling_factor * diffs_to_min
        return scaled_diffs_to_min_vec + self.target_min_vec

    def _linear_transform_2(
        self, data, input_min_vec, input_max_vec, target_min_vec, target_max_vec
    ):
        scaling_factor = (target_max_vec - target_min_vec) / (
            input_max_vec - input_min_vec
        )
        diffs_to_min = data - input_min_vec
        scaled_diffs_to_min_vec = scaling_factor * diffs_to_min
        return scaled_diffs_to_min_vec + target_min_vec

    def transform(self, data):
        if self.min_vec is None or self.max_vec is None:
            raise LinearScalerError(
                "Cannot transform. You first have to call .fit_transform()."
            )

        return self._linear_transform(data)

    def inverse_transform(self, data):
        if self.min_vec is None or self.max_vec is None:
            raise LinearScalerError(
                "Cannot inverse-transform. You first have to call .fit_transform()."
            )
        inverse_transformed_data = self._linear_transform_2(
            data,
            input_min_vec=self.target_min_vec,
            input_max_vec=self.target_max_vec,
            target_min_vec=self.min_vec,
            target_max_vec=self.max_vec,
        )
        return inverse_transformed_data

    def fit_transform(self, data):
        if self.target_min_vec is None:
            self.target_min_vec = np.array([-1.0 for _ in range(data.shape[-1])])
        if self.target_max_vec is None:
            self.target_max_vec = np.array([1.0 for _ in range(data.shape[-1])])
        self.max_vec = np.max(data, axis=0)
        self.min_vec = np.min(data, axis=0)
        return self.transform(data)


def standardize_TensorContextDataset(
    tensor_context: TensorContextDataset,
    scaler: StandardScaler | MinMaxScaler | LinearScaler,
    fit: bool = True,
    periodic_shift: bool = False,
    basin: str | None = None,
    backend: str = "auto",
    **backend_kw,
) -> TensorContextDataset:
    """Standardizes a TensorContextDataset.

    Data standardization is performed by the scaler. Often used scalers are the
    standard scaler, which scales each feature to zero mean and unit variance, or global
    linear scaler, which transform the data by a affine linear transformation to a
    target rectangular domain.

    TODO At the moment the TensorContextDataset is standardized, by flattening into an
    array of shape (-1, n_features). An alternative would be to standardize the
    data_array_list (output of data_array_list_from_TCTracks), by concatenating all
    arrays of shape (-1, n_features) in data_array_list. This latter approach is
    implemented for the standardization of the KNF-adjusted dataset. The disadvantage
    of the former approach is that there might be some bias induced towards data points
    in the middle of the time series contained in data_array_list. This is because all
    these time series are sampled with a slicing window to homogenize the format into a
    dense array of time series of equal length, which is needed as input to the models.

    Args:
        tensor_context (TensorContextDataset): _description_
        scaler (StandardScaler | MinMaxScaler | LinearScaler): _description_
        fit (bool, optional): _description_. Defaults to True.
        backend (str, optional): _description_. Defaults to "auto".
        **backend_kw (Any): _description_

    Returns:
        TensorContextDataset: _description_
    """
    if periodic_shift:
        if basin is None:
            raise Exception("If periodic_shift is True, basin must be specified.")
        tensor_context = periodic_shift_TensorContextDataset(
            tensor_context, basin=basin, **backend_kw
        )

    if fit:
        data_transformed = scaler.fit_transform(
            tensor_context.data.reshape(
                (
                    tensor_context.shape[0] * tensor_context.shape[1],
                    tensor_context.shape[2],
                )
            )
        ).reshape(tensor_context.shape)
    else:
        data_transformed = scaler.transform(
            tensor_context.data.reshape(
                (
                    tensor_context.shape[0] * tensor_context.shape[1],
                    tensor_context.shape[2],
                )
            )
        ).reshape(tensor_context.shape)
    tensor_context_transformed = TensorContextDataset(
        data_transformed, backend, **backend_kw
    )
    return tensor_context_transformed


def concatenate_time_series_list(time_series_list: list[NDArray]):
    concatenated_time_series = np.concatenate(time_series_list, axis=0)
    return concatenated_time_series


def time_series_list_from_concatenated_time_series(
    concatenated_time_series: NDArray, length_list: list[int]
):
    time_series_list = []
    counter = 0
    for length in length_list:
        time_series_list.append(concatenated_time_series[counter : counter + length])
        counter = counter + length

    return time_series_list


def standardize_time_series_list(
    time_series_list: list[NDArray],
    scaler: Union[StandardScaler, MinMaxScaler, LinearScaler],
    fit: bool = True,
) -> list[NDArray]:
    """Standardizes a TensorContextDataset.

    Data standardization is performed by the scaler. Often used scalers are the
    standard scaler, which scales each feature to zero mean and unit variance, or global
    linear scaler, which transform the data by a affine linear transformation to a
    target rectangular domain.

    Args:
        time_series_list (TensorContextDataset): _description_
        scaler (StandardScaler | MinMaxScaler | LinearScaler): _description_
        fit (bool, optional): _description_. Defaults to True.

    Returns:
        TensorContextDataset: _description_
    """
    length_list = [len(da) for da in time_series_list]
    concatenated_time_series = concatenate_time_series_list(time_series_list)
    if fit:
        rescaled_concatenated_time_series = scaler.fit_transform(
            concatenated_time_series
        )
    else:
        rescaled_concatenated_time_series = scaler.transform(concatenated_time_series)

    rescaled_time_series_list = time_series_list_from_concatenated_time_series(
        rescaled_concatenated_time_series, length_list
    )
    return rescaled_time_series_list


def periodic_identification(
    data: NDArray, limit_min: float, limit_max: float
) -> NDArray:
    def _p_id(x, limit_min, limit_max):
        dist = limit_max - limit_min
        if x >= limit_max:
            x = x - dist
        if x < limit_min:
            x = x + dist
        return x

    _p_id_vec = np.vectorize(_p_id, excluded=["limit_min", "limit_max"])
    return _p_id_vec(data, limit_min, limit_max)


def periodic_shift(
    data: NDArray,
    shift: float,
    dim: int,
    limits: tuple[float, float],
) -> NDArray:
    if not len(data.shape) == 3:
        raise Exception("Data must be 3-dimensional array.")

    if shift == 0:
        return data

    data_c = data.copy()
    data_c[:, :, dim] = data_c[:, :, dim] + shift
    data_c[:, :, dim] = periodic_identification(
        data_c[:, :, dim], limit_min=limits[0], limit_max=limits[1]
    )
    return data_c


def periodic_shift_TensorContextDataset(
    tensor_context_dataset: TensorContextDataset,
    shift: float | None = None,
    dim: int | None = None,
    limits: tuple[float, float] | None = None,
    basin: str | None = None,
    backend: str = "auto",
    verbose: str = False,
    **backend_kw,
) -> TensorContextDataset:
    if basin is not None:
        if basin not in BASIN_SHIFT_SWITCHER.keys():
            raise Exception("Not a valid value for basin.")
        shift = BASIN_SHIFT_SWITCHER[basin]
        dim = 0
        limits = (-180, 180)
        if verbose:
            print(basin, shift)
    else:
        if shift is None:
            raise Exception("If basin is not specified, shifts needs to be specified.")
        if dim is None:
            dim = 0
        if limits is None:
            limits = (-180, 180)
    data = tensor_context_dataset.data
    data_shifted = periodic_shift(data, shift, dim, limits)
    tensor_context_dataset_shifted = TensorContextDataset(
        data_shifted, backend, **backend_kw
    )

    return tensor_context_dataset_shifted


def standardized_context_dataset_from_TCTracks(
    tc_tracks: TCTracks,
    feature_list: list[str],
    scaler: StandardScaler | MinMaxScaler | LinearScaler,
    context_length: int = 2,
    time_lag: int = 1,
    fit: bool = True,
    periodic_shift: bool = True,
    basin: str | None = None,
    verbose: int = 0,
    backend: str = "auto",
    **backend_kw,
) -> TensorContextDataset:
    """Generates standardized TensorContextDataset from TCTracks.

    Args:
        tc_tracks (TCTracks): _description_
        feature_list (list[str]): _description_
        scaler (StandardScaler | MinMaxScaler | LinearScaler): _description_
        context_length (int, optional): _description_. Defaults to 2.
        time_lag (int, optional): _description_. Defaults to 1.
        fit (bool, optional): _description_. Defaults to True.
        periodic_shift (bool, optional): _description_. Defaults to True.
        basin (str | None, optional): _description_. Defaults to None.
        verbose (int, optional): _description_. Defaults to 1.
        backend (str, optional): _description_. Defaults to "auto".

    Returns:
        TensorContextDataset: _description_
    """
    tensor_context = context_dataset_from_TCTracks(
        tc_tracks,
        feature_list=feature_list,
        context_length=context_length,
        time_lag=time_lag,
        verbose=verbose,
        **backend_kw
    )  
    standardized_tensor_context = standardize_TensorContextDataset(
        tensor_context,
        scaler=scaler,
        fit=fit,
        periodic_shift=periodic_shift,
        basin=basin,
        backend=backend,
        **backend_kw,
    )
    return standardized_tensor_context


def get_parameters_from_files(folder_path: str, keys: list[str]):
    file_list = glob.glob(os.path.join(folder_path, "*train_steps*.pth"))

    index_identifier = {
        "train_steps": ("train_steps", "_results.pth"),
        "koopman_kernel_rank": ("kkrank", "_kkrdrank"),
        "context_length": ("kkcntlength", "_kktsteph"),
    }

    param_dict = {}
    for key in keys:
        param_dict[key] = []
    # for key in keys:
    #     start_pattern, stop_pattern = index_identifier[key]
    #     param_list = []

    for file_name in file_list:
        for key in keys:
            start_pattern, stop_pattern = index_identifier[key]
            index_start = file_name.find(start_pattern)
            index_end = file_name.find(stop_pattern)
            param_dict[key].append(
                int(file_name[index_start + len(start_pattern) : index_end])
            )

    df_params = pd.DataFrame()
    for key in keys:
        df_params[key] = param_dict[key]

    return df_params


def load_model_results(flag_params: dict, path_training_results: str, keys: list[str]):
    import_dir = os.path.join(
        path_training_results,
        "{}_yrange{}_basin{}".format(
            flag_params["dataset"],
            "".join(map(str, flag_params["year_range"])),
            flag_params["basin"],
        ),
        flag_params["model"],
    )
    model_name = "seed{}_kklnscale{}_kkrank{}_kkrdrank{}_kktkreg{}_kkncntr{}_kkntstops{}_kkcntlength{}_kktsteph{}".format(  # noqa: E501
        flag_params["seed"],
        flag_params["koopman_kernel_length_scale"],
        flag_params["koopman_kernel_rank"],
        flag_params["koopman_kernel_reduced_rank"],
        flag_params["tikhonov_reg"],
        flag_params["koopman_kernel_num_centers"],
        flag_params["koopman_kernel_num_train_stops"],
        flag_params["context_length"],
        flag_params["time_step_h"],
    )

    results_file_name = os.path.join(import_dir, model_name)
    df = get_parameters_from_files(import_dir, keys)
    for key in [key for key in keys if key != "train_steps"]:
        df = df[df[key] == flag_params[key]]
    train_steps = df.sort_values(by="train_steps")["train_steps"].to_list()

    load_keys = ["scaler", "eval_rmse", "train_stop", "fit_time"]
    results = {}
    for step in train_steps:
        res = torch.load(results_file_name + f"_train_steps{step}_results.pth")
        results[step] = {}
        for key in load_keys:
            results[step][key] = res[key]

    return results


def load_model(flag_params: dict, path_training_results: str):
    import_dir = os.path.join(
        path_training_results,
        "{}_yrange{}_basin{}".format(
            flag_params["dataset"],
            "".join(map(str, flag_params["year_range"])),
            flag_params["basin"],
        ),
        flag_params["model"],
    )
    model_name = "seed{}_kklnscale{}_kkrank{}_kkrdrank{}_kktkreg{}_kkncntr{}_kkntstops{}_kkcntlength{}_kktsteph{}".format(  # noqa: E501
        flag_params["seed"],
        flag_params["koopman_kernel_length_scale"],
        flag_params["koopman_kernel_rank"],
        flag_params["koopman_kernel_reduced_rank"],
        flag_params["tikhonov_reg"],
        flag_params["koopman_kernel_num_centers"],
        flag_params["koopman_kernel_num_train_stops"],
        flag_params["context_length"],
        flag_params["time_step_h"],
    )

    results_file_name = os.path.join(import_dir, model_name)
    best_model_dict = torch.load(results_file_name + "_best_model.pth")
    best_results_dict = torch.load(results_file_name + "_best_results.pth")

    return best_model_dict, best_results_dict
