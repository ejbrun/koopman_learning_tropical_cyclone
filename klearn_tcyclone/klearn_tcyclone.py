"""Model benchmark class."""

import logging
from typing import Union

# import numpy as np
import torch
from numpy import log10, logspace

from klearn_tcyclone.data_utils import (
    context_dataset_from_TCTracks,
    standardize_TensorContextDataset,
)
from klearn_tcyclone.models_utils import runner

logger = logging.getLogger("ModelBenchmark")


class ModelBenchmarkError(Exception):
    """A custom exception used to report errors in use of Timer class."""


class ModelBenchmark:
    """A class to represent a person.

    TODO fix docstring
    Attributes:
    ----------
    model : Any
        The model that is trained and benchmarked.
    features : list
        List of features (of the tropical cylone dataset) on which the training is
            performed.
    scaler : Any
        Scaler or method for data standardization.
    context_length : int
        Context length of the context windows (the Kooplearn data points)
    results : Union[dict, list[dict]]
        list of results.

    Methods:
    -------
    info(additional=""):
        Prints the person's name and age.
    """

    def __init__(
        self,
        # model,
        features,
        tc_tracks_train,
        tc_tracks_test,
        basin: str,
        scaler=None,
        context_length: int = 42,
        time_lag: int | None = None,
    ) -> None:
        # self.model = model
        self.features = features
        self.basin = basin
        self.scaler = scaler
        self.context_length = context_length
        if time_lag is None:
            self.time_lag = 1
        else:
            self.time_lag = time_lag
        self._set_tensor_contexts(tc_tracks_train, tc_tracks_test)
        self._results = None

    @property
    def results(
        self,
    ) -> Union[dict, list[dict]]:  # TODO replace Unions by | and update python version
        """Benchmark results.

        Raises:
            ModelBenchmarkError: _description_

        Returns:
            Union[dict, list[dict]]: List of dictionaries containing the results.
        """
        if self._results is None:
            raise ModelBenchmarkError(
                "No results yet. You first have to call .train_model()."
            )
        else:
            return self._results

    def _set_tensor_contexts(self, tc_tracks_train, tc_tracks_test) -> None:
        tensor_context_train = context_dataset_from_TCTracks(
            tc_tracks_train,
            feature_list=self.features,
            context_length=self.context_length,
            time_lag=self.time_lag,
        )
        tensor_context_test = context_dataset_from_TCTracks(
            tc_tracks_test,
            feature_list=self.features,
            context_length=self.context_length,
            time_lag=self.time_lag,
        )
        self.tensor_context_train = tensor_context_train
        self.tensor_context_test = tensor_context_test

    def _standardize_data(self):
        if self.scaler is not None:
            self.tensor_context_train = standardize_TensorContextDataset(
                self.tensor_context_train,
                self.scaler,
                fit=True,
                periodic_shift=True,
                basin=self.basin,
            )
            self.tensor_context_test = standardize_TensorContextDataset(
                self.tensor_context_test,
                self.scaler,
                fit=False,
                periodic_shift=True,
                basin=self.basin,
            )

    def get_info(self):
        """Prints information about the data."""
        print(
            " ".join(
                [
                    f"Train contexts have shape {self.tensor_context_train.shape}:",
                    f"{len(self.tensor_context_train)} contexts of length",
                    f"{self.tensor_context_train.context_length} with",
                    f"{self.tensor_context_train.shape[2]} features each.",
                ]
            )
        )
        print(
            " ".join(
                [
                    f"Test contexts have shape {self.tensor_context_test.shape}:",
                    f"{len(self.tensor_context_test)} contexts of length",
                    f"{self.tensor_context_test.context_length} with",
                    f"{self.tensor_context_test.shape[2]} features each.",
                ]
            )
        )

    def train_model(
        self,
        model,
        eval_metric,
        train_stops: int | list[int] | None = None,
        num_train_stops: int | None = None,
        save_model: str | bool = False,
        save_results: bool = False,
        save_path: str | None = None,
    ) -> dict | list[dict]:
        """Model training.

        Args:
            train_stops (Union[int, list[int]]): Maximum number of data points (context
                windows) that are taken into account.

        Returns:
            Union[dict, list[dict]]: Results of the model training.
        """
        if save_model == "all":
            save_model = True

        self._standardize_data()
        tensor_contexts = {
            "train": self.tensor_context_train,
            "test": self.tensor_context_test,
        }

        if train_stops is None:
            training_data_size = tensor_contexts["train"].shape[0]
            # TODO check what happens if for each stop in training_stops, the model is initialized from scratch
            train_stops = logspace(
                2.5, log10(training_data_size), num_train_stops
            ).astype(int)

        best_eval_rmse = 1e6
        results = []
        for stop in train_stops:
            logger.info(f"\nModel training: Training points: {stop}")
            print(f"\nModel training: Training points: {stop}")
            results.append(runner(model, tensor_contexts, stop))

            RMSE_onestep_train_error = eval_metric(
                results[-1]["X_pred_train"], results[-1]["X_true_train"]
            )
            RMSE_onestep_test_error = eval_metric(
                results[-1]["X_pred_test"], results[-1]["X_true_test"]
            )

            print_str = " ".join(
                [
                    r"Fitting of model took {:.2f}s".format(results[-1]["fit_time"]),
                    r"with train RMSE of {:.5f} and test RMSE of {:.5f}.".format(  # noqa: UP032
                        RMSE_onestep_train_error, RMSE_onestep_test_error
                    ),
                ]
            )
            logger.info(print_str)
            print(print_str)
            
            eval_rmse = RMSE_onestep_test_error
            if save_results:
                save_results = {
                    "scaler": self.scaler,
                    "eval_rmse": eval_rmse,
                    "train_stop": results[-1]["train_stop"],
                    "fit_time": results[-1]["fit_time"],
                }
                torch.save(
                    save_results,
                    save_path + f"_train_steps{stop}" + "_results.pth",
                )
            if save_model is True:
                torch.save(
                    {"model": model, "scaler": self.scaler},
                    save_path + f"_train_steps{stop}" + "_model.pth",
                )
            if eval_rmse < best_eval_rmse:
                best_eval_rmse = eval_rmse
                if save_results:
                    torch.save(
                        save_results,
                        save_path + "_best" + "_results.pth",
                    )
                if save_model == "best" or save_model is True:
                    torch.save(
                        {"model": model, "scaler": self.scaler},
                        save_path + "_best" + "_model.pth",
                    )

        self._results = results
        return results
