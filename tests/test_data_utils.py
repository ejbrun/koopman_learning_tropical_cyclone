"""Tests for utils.py."""

import numpy as np
import pytest
from kooplearn.data import TensorContextDataset
from numpy.testing import assert_allclose

from klearn_tcyclone.climada.tc_tracks import TCTracks
from klearn_tcyclone.data_utils import (
    LinearScaler,
    concatenate_time_series_list,
    context_dataset_from_TCTracks,
    data_array_list_from_TCTracks,
    linear_transform,
    periodic_shift,
    periodic_shift_TensorContextDataset,
    standardize_time_series_list,
    time_series_list_from_concatenated_time_series,
)


def test_data_array_list_from_TCTracks():
    """Test for data_array_list_from_TCTracks."""
    tc_tracks: TCTracks = TCTracks.from_ibtracs_netcdf(
        provider="usa", year_range=(2000, 2001), basin="EP", correct_pres=False
    )
    n_tracks = len(tc_tracks.data)
    feature_list = ["lon", "lat", "max_sustained_wind", "central_pressure"]
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
    feature_list = ["lon", "lat", "max_sustained_wind", "central_pressure"]

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
    min_vec = np.array([1, 1])
    max_vec = np.array([4, 5])
    target_min_vec = np.array([-1, -1])
    target_max_vec = np.array([1, 1])
    data = np.array([[1, 1], [1, 5], [4, 1], [4, 5], [1, 3]])

    fun = linear_transform(min_vec, max_vec, target_min_vec, target_max_vec)
    transformed_data = fun(data)
    target = np.array(
        [
            [-1, -1],
            [-1, 1],
            [1, -1],
            [1, 1],
            [-1, 0],
        ]
    )
    assert_allclose(transformed_data, target)


def test_concatenate_time_series_list():
    """Test for test_concatenate_time_series_list."""
    time_series_list = [
        np.random.normal(0, 1, (length, 2)) for length in np.random.randint(10, 20, 5)
    ]
    concatenated_time_series = concatenate_time_series_list(time_series_list)
    length_list = [len(da) for da in time_series_list]

    assert concatenated_time_series.shape == (np.sum(length_list), 2)


def test_time_series_list_from_concatenated_time_series():
    """Test for test_time_series_list_from_concatenated_time_series."""
    time_series_list = [
        np.random.normal(0, 1, (length, 2)) for length in np.random.randint(10, 20, 5)
    ]
    concatenated_time_series = concatenate_time_series_list(time_series_list)
    length_list = [len(da) for da in time_series_list]
    time_series_list_test = time_series_list_from_concatenated_time_series(
        concatenated_time_series, length_list
    )
    is_equal = [
        np.all(time_series_list[idx] == time_series_list_test[idx])
        for idx in range(len(time_series_list))
    ]
    assert np.all(is_equal)


def test_standardize_time_series_list():
    """Test for test_standardize_time_series_list."""
    time_series_list = [
        np.random.normal(0, 1, (length, 2)) for length in np.random.randint(10, 20, 5)
    ]
    scaler = LinearScaler()
    rescaled_time_series_list = standardize_time_series_list(
        time_series_list, scaler=scaler, fit=True
    )
    is_equal_shape = [
        time_series_list[idx].shape == rescaled_time_series_list[idx].shape
        for idx in range(len(time_series_list))
    ]
    assert np.all(is_equal_shape)


def test_periodic_shift_0():
    """Test for periodic_shift."""
    data = np.random.uniform(-1, 1, (120, 12, 4))
    data_shifted = periodic_shift(data, shift=0.5, dim=0, limits=(-1, 1))
    data_backshifted = periodic_shift(data_shifted, shift=-0.5, dim=0, limits=(-1, 1))
    dist = np.linalg.norm(data - data_backshifted)
    assert dist < 1e-12


def test_periodic_shift_1():
    """Test for periodic_shift."""
    x_space = np.linspace(-1, 1, 11)
    data = np.array(
        [
            [
                x_space,
                x_space,
            ],
            [
                x_space,
                1 + x_space,
            ],
        ]
    ).transpose((0, 2, 1))
    data_shifted = periodic_shift(data, shift=1, dim=0, limits=(-1, 1))
    data_true = np.array(
        [
            [
                np.concatenate([np.linspace(0, 0.8, 5), np.linspace(-1, 0, 6)]),
                x_space,
            ],
            [
                np.concatenate([np.linspace(0, 0.8, 5), np.linspace(-1, 0, 6)]),
                1 + x_space,
            ],
        ]
    ).transpose((0, 2, 1))
    dist = np.linalg.norm(data_true - data_shifted)
    assert dist < 1e-12


def test_periodic_shift_TensorContextDataset_0():
    """Test for periodic_shift_TensorContextDataset."""
    data = np.random.uniform(-1, 1, (120, 12, 4))
    tcd = TensorContextDataset(data)
    tcd_shifted = periodic_shift_TensorContextDataset(
        tcd, shift=1.0, dim=0, limits=(-1, 1)
    )
    data_shifted = periodic_shift(data, shift=1.0, dim=0, limits=(-1, 1))
    data_tcd_shifted = tcd_shifted.data
    dist = np.linalg.norm(data_shifted - data_tcd_shifted)
    assert dist < 1e-12


def test_periodic_shift_TensorContextDataset_1():
    """Test for periodic_shift_TensorContextDataset."""
    data = np.random.uniform(-1, 1, (120, 12, 4))
    tcd = TensorContextDataset(data)
    tcd_shifted = periodic_shift_TensorContextDataset(
        tcd, shift=1.0, dim=0, limits=(-1, 1)
    )
    tcd_backshifed = periodic_shift_TensorContextDataset(
        tcd_shifted, shift=-1.0, dim=0, limits=(-1, 1)
    )
    dist = np.linalg.norm(tcd.data - tcd_backshifed.data)
    assert dist < 1e-12
