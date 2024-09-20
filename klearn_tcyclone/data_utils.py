"""Utils for data processing."""

import warnings
from climada.hazard import TCTracks
import numpy as np
from xarray import Dataset
from kooplearn.data import TrajectoryContextDataset, TensorContextDataset
from scipy.spatial.distance import pdist
from typing import Union
from numpy.typing import NDArray
from collections.abc import Callable
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def data_array_list_from_TCTracks(
    tc_tracks: Union[TCTracks, list[Dataset]], feature_list: list[str]
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
        if data_array.shape[0] >= context_length:
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
                    context_length."""
                )

    tensor_context_dataset = TensorContextDataset(
        context_data_array, backend, **backend_kw
    )
    return tensor_context_dataset


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

    def __init__(
        self, target_min_vec=np.array([-1.0, -1.0]), target_max_vec=np.array([1.0, 1.0])
    ):
        self.target_min_vec = np.array(target_min_vec)
        self.target_max_vec = np.array(target_max_vec)
        self.min_vec = None
        self.max_vec = None

    def _linear_transform(self, data):
        scaling_factor = (self.target_max_vec - self.target_min_vec) / (
            self.max_vec - self.min_vec
        )
        diffs_to_min = data - self.min_vec
        scaled_diffs_to_min_vec = scaling_factor * diffs_to_min
        return scaled_diffs_to_min_vec + self.target_min_vec

    def transform(self, data):
        if self.min_vec is None or self.max_vec is None:
            raise LinearScalerError(
                "Cannot transform. You first have to call .fit_transform()."
            )

        return self._linear_transform(data)

    def fit_transform(self, data):
        self.max_vec = np.max(data, axis=0)
        self.min_vec = np.min(data, axis=0)
        return self.transform(data)


def standardize_TensorContextDataset(
    tensor_context: TensorContextDataset,
    scaler: Union[StandardScaler, MinMaxScaler, LinearScaler],
    fit: bool = True,
    backend: str = "auto",
    **backend_kw,
) -> TensorContextDataset:
    """Standardizes a TensorContextDataset.

    Data standardization is performed by the scaler. Often used scalers are the
    standard scaler, which scales each feature to zero mean and unit variance, or global
    linear scaler, which transform the data by a affine linear transformation to a
    target rectangular domain.

    Args:
        tensor_context (TensorContextDataset): _description_
        scaler (StandardScaler | MinMaxScaler | LinearScaler): _description_
        fit (bool, optional): _description_. Defaults to True.
        backend (str, optional): _description_. Defaults to "auto".
        **backend_kw (Any): _description_

    Returns:
        TensorContextDataset: _description_
    """
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
