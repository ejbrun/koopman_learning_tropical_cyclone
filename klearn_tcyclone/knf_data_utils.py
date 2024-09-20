"""Utils for Koopman Neural Forecaster datat handling."""

from climada.hazard import TCTracks
import numpy as np
import torch
from xarray import Dataset
from typing import Union
from klearn_tcyclone.data_utils import data_array_list_from_TCTracks


class TCTrackDataset(torch.utils.data.Dataset):
    """Dataset class for NBA player trajectory data.

    Attributes:
    ----------
    input_length: int
        num of input steps.
    output_length: int
        forecasting horizon.
    tc_tracks: Union[TCTracks, list[Dataset]]
        TCTracks data from which the torch dataset is constructed.
    feature_list: list[str]
        List of features from the TCTracks data.
    mode: str = "train"
        train, validation or test.
    jumps: int = 1
        number of skipped steps between two sliding windows.
    freq: None=None
        Not needed, only there for consistency with other datasets.
    """

    def __init__(
        self,
        input_length: int,  # num of input steps
        output_length: int,  # forecasting horizon
        tc_tracks: Union[TCTracks, list[Dataset]],
        feature_list: list[str],
        mode: str = "train",  # train, validation or test
        jumps: int = 1,  # number of skipped steps between two sliding windows
        freq: None=None,
    ):
        self.input_length = input_length
        self.output_length = output_length
        self.mode = mode

        data_array_list = data_array_list_from_TCTracks(
            tc_tracks=tc_tracks, feature_list=feature_list
        )
        self.data = data_array_list

        if mode == "test":
            # change the input length (<100) will not affect the target output
            self.ts_indices = []
            for i, item in enumerate(self.test_lsts):
                for j in range(100, len(item) - output_length, output_length):
                    # for i in range(len(self.data)):
                    #     for j in range(50, 300 - output_length, 50):
                    self.ts_indices.append((i, j))
        elif mode == "train" or "valid":
            # shuffle slices before split
            np.random.seed(123)
            self.ts_indices = []
            for i, item in enumerate(self.data):
                for j in range(0, len(item) - input_length - output_length, jumps):
                    self.ts_indices.append((i, j))
            np.random.shuffle(self.ts_indices)

            # 90%-10% train-validation split
            train_valid_split = int(len(self.ts_indices) * 0.9)
            if mode == "train":
                self.ts_indices = self.ts_indices[:train_valid_split]
            elif mode == "valid":
                self.ts_indices = self.ts_indices[train_valid_split:]
        else:
            raise ValueError("Mode can only be one of train, valid, test")

    def __len__(self):
        return len(self.ts_indices)

    def __getitem__(self, index):
        if self.mode == "test":
            i, j = self.ts_indices[index]
            x = self.data[i][j - self.input_length : j]
            y = self.data[i][j : j + self.output_length]
        else:
            i, j = self.ts_indices[index]
            x = self.data[i][j : j + self.input_length]
            y = self.data[i][
                j + self.input_length : j + self.input_length + self.output_length
            ]
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()
