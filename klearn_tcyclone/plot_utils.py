"""Utils for plotting."""

from random import sample

import numpy as np
from matplotlib.axes._axes import Axes

from klearn_tcyclone.data_utils import data_array_list_from_TCTracks
from klearn_tcyclone.knf_model_utils import predict_koopman_model
from klearn_tcyclone.knf_data_utils import TCTrackDataset


def plot_feature(tc_tracks, feature, ax: Axes, n_tracks: int = 10):
    data_array_list = data_array_list_from_TCTracks(
        tc_tracks=tc_tracks, feature_list=["time", feature]
    )

    data_array_list = sample(data_array_list, k=n_tracks)
    for data_array in data_array_list[:n_tracks]:
        ax.plot(range(len(data_array[:, 1])), data_array[:, 1])
        # ax.plot(data_array[:,0], data_array[:,1])


def plot_features(tc_tracks, features, ax: Axes, n_tracks: int = 10):
    data_array_list = data_array_list_from_TCTracks(
        tc_tracks=tc_tracks, feature_list=features
    )

    data_array_list = sample(data_array_list, k=n_tracks)
    for idx_feature, feature in enumerate(features):
        for data_array in data_array_list[:n_tracks]:
            ax[idx_feature].plot(
                range(len(data_array[:, idx_feature])), data_array[:, idx_feature]
            )
            ax[idx_feature].set_ylabel(feature)
            # ax.plot(data_array[:,0], data_array[:,1])


def plot_TCTrackDataset_item(
    tc_tracks_dataset: TCTrackDataset,
    idx: int,
    dimension: int,
    ax: Axes,
    color: str,
    t_min: float = 0,
    t_max: float = 1,
):
    inp, tgt = tc_tracks_dataset[idx]
    time = np.linspace(t_min, t_max, len(inp) + len(tgt))
    time_inp = time[: len(inp)]
    time_tgt = time[len(inp) :]
    ax.plot(time_inp, inp[:, dimension], marker="x", color=color)
    ax.plot(time_tgt, tgt[:, dimension], marker="^", color=color)


def plot_TCTrackDataset_item_2D(
    tc_tracks_dataset: TCTrackDataset,
    idx: int,
    ax: Axes,
    color: str,
    dimensions: tuple[int, int] = (0, 1),
):
    inp, tgt = tc_tracks_dataset[idx]
    ax.plot(inp[:, dimensions[0]], inp[:, dimensions[1]], marker="x", color=color)
    ax.plot(tgt[:, dimensions[0]], tgt[:, dimensions[1]], marker="^", color=color)


def plot_TCTrackDataset_item_2D_with_prediction(
    tc_tracks_dataset: TCTrackDataset,
    idx: int,
    prediction_steps,
    model,
    ax: Axes,
    color: str,
    dimensions: tuple[int, int] = (0, 1),
    num_steps: int = None,
):
    plot_TCTrackDataset_item_2D(tc_tracks_dataset, idx, ax, color, dimensions)
    time_series, _ = tc_tracks_dataset[idx]
    predictions = predict_koopman_model(
        time_series, prediction_steps, model, num_steps=num_steps
    )
    predictions = predictions.cpu().data.numpy()
    ax.plot(
        predictions[:, dimensions[0]],
        predictions[:, dimensions[1]],
        marker="s",
        color=color,
    )
