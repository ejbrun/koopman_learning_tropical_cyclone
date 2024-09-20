"""Utils for Koopman Neural Forecaster datat handling."""

import torch
import numpy as np

from numpy.typing import NDArray


class TCTrackDataset(torch.utils.data.Dataset):
    """Dataset class for NBA player trajectory data."""

    def __init__(
        self,
        input_length: int,  # num of input steps
        output_length: int,  # forecasting horizon
        data_array_list: NDArray,
        mode: str = "train",  # train, validation or test
        jumps: int = 1,  # number of skipped steps between two sliding windows
        freq=None,
    ):
        self.input_length = input_length
        self.output_length = output_length
        self.mode = mode
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
