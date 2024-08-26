"""Utils for plotting."""

from klearn_tcyclone.data_utils import data_array_list_from_TCTracks
from random import sample


def plot_feature(tc_tracks, feature, ax, n_tracks: int = 10):
    data_array_list = data_array_list_from_TCTracks(
        tc_tracks=tc_tracks, feature_list=["time", feature]
    )

    data_array_list = sample(data_array_list, k=n_tracks)
    for data_array in data_array_list[:n_tracks]:
        ax.plot(range(len(data_array[:, 1])), data_array[:, 1])
        # ax.plot(data_array[:,0], data_array[:,1])


def plot_features(tc_tracks, features, ax, n_tracks: int = 10):
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
