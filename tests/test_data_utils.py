"""Tests for utils.py."""

from klearn_tcyclone.data_utils import (
    data_array_list_from_TCTracks,
    context_dataset_from_TCTracks,
)
from climada.hazard import TCTracks
import numpy as np
import pytest
from klearn_tcyclone.data_utils import linear_transform
from numpy.testing import assert_allclose


def test_data_array_list_from_TCTracks():
    """Test for data_array_list_from_TCTracks."""
    tc_tracks: TCTracks = TCTracks.from_ibtracs_netcdf(
        provider="usa", year_range=(2000, 2001), basin="EP", correct_pres=False
    )
    n_tracks = len(tc_tracks.data)
    feature_list = ["lat", "lon", "max_sustained_wind", "central_pressure"]
    n_features = len(feature_list)

    data_array_list = data_array_list_from_TCTracks(
        tc_tracks=tc_tracks, feature_list=feature_list
    )
    assert (
        len(data_array_list) == n_tracks
    ), "Length of data_array_list must be equal to n_tracks."

    assert np.all(
        [data_array.shape[1] == n_features for data_array in data_array_list]
    ), "The dimension of the first axis of data_array must be equal to n_features."


@pytest.mark.parametrize("context_length", [2, 4, 7, 12, 42])
def test_context_dataset_from_TCTracks(context_length):
    """Test for context_dataset_from_TCTracks."""
    tc_tracks: TCTracks = TCTracks.from_ibtracs_netcdf(
        provider="usa", year_range=(2000, 2001), basin="EP", correct_pres=False
    )
    feature_list = ["lat", "lon", "max_sustained_wind", "central_pressure"]

    data_array_list = data_array_list_from_TCTracks(
        tc_tracks=tc_tracks, feature_list=feature_list
    )

    n_entries = len([x for x in data_array_list if x.shape[0] >= context_length])
    n_data = np.sum(
        [
            data_array.shape[0]
            for data_array in data_array_list
            if data_array.shape[0] >= context_length
        ]
    )
    len_tensor_context_check = n_data - n_entries * (context_length - 1)

    tensor_context_dataset = context_dataset_from_TCTracks(
        tc_tracks, feature_list=feature_list, context_length=context_length
    )
    len_tensor_context = tensor_context_dataset.shape[0]

    assert (
        len_tensor_context == len_tensor_context_check
    ), "Length of tensor context is wrong."


def test_linear_transform():
    """Test for linear_transform."""
    min_vec = np.array([1,1])
    max_vec = np.array([4,5])
    target_min_vec = np.array([-1,-1])
    target_max_vec = np.array([1,1])
    data = np.array([[1,1], [1,5], [4,1], [4,5], [1,3]])

    fun = linear_transform(min_vec, max_vec, target_min_vec, target_max_vec)
    transformed_data = fun(data)
    target = np.array([
        [-1,-1],
        [-1,1],
        [1,-1],
        [1,1],
        [-1,0],
    ])
    assert_allclose(transformed_data, target)
