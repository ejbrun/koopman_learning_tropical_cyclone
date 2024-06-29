"""Utils for data processing."""

import warnings
from climada.hazard import TCTracks
import numpy as np
from kooplearn.data import TrajectoryContextDataset, TensorContextDataset


def data_array_list_from_TCTracks(tc_tracks: TCTracks, feature_list: list[str]) -> list:
    """Create data array list from TCTracks.

    Args:
        tc_tracks (TCTracks): _description_
        feature_list (list[str]): _description_

    Returns:
        list: _description_
    """
    tc_data = tc_tracks.data

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
            warnings.warn(
                f"""Data entry {idx} has been removed since it is shorter than the 
                context_length.""",
                stacklevel=2,
            )

    tensor_context_dataset = TensorContextDataset(
        context_data_array, backend, **backend_kw
    )
    return tensor_context_dataset
