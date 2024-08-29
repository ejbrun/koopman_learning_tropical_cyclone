
from typing import Union
from klearn_tcyclone.data_utils import (
    context_dataset_from_TCTracks,
    standardize_TensorContextDataset,
)
from klearn_tcyclone.models_utils import runner


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
        model,
        features,
        tc_tracks_train,
        tc_tracks_test,
        scaler=None,
        context_length: int = 42,
    ) -> None:
        self.model = model
        self.features = features
        self.scaler = scaler
        self.context_length = context_length
        self._set_tensor_contexts(tc_tracks_train, tc_tracks_test)
        self._results = None

    @property
    def results(self)->Union[dict, list[dict]]: #TODO replace Unions by | and update python version
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
        )
        tensor_context_test = context_dataset_from_TCTracks(
            tc_tracks_test,
            feature_list=self.features,
            context_length=self.context_length,
        )
        self.tensor_context_train = tensor_context_train
        self.tensor_context_test = tensor_context_test

    def _standardize_data(self):
        if self.scaler is not None:
            self.tensor_context_train = standardize_TensorContextDataset(
                self.tensor_context_train,
                self.scaler,
                fit=True,
            )
            self.tensor_context_test = standardize_TensorContextDataset(
                self.tensor_context_test,
                self.scaler,
                fit=False,
            )

    def get_info(self):
        """Prints information about the data."""
        print(
            " ".join([
                f"Train contexts have shape {self.tensor_context_train.shape}:",
                f"{len(self.tensor_context_train)} contexts of length",
                f"{self.tensor_context_train.context_length} with",
                f"{self.tensor_context_train.shape[2]} features each.",
            ])
        )
        print(
            " ".join([
                f"Test contexts have shape {self.tensor_context_test.shape}:",
                f"{len(self.tensor_context_test)} contexts of length",
                f"{self.tensor_context_test.context_length} with",
                f"{self.tensor_context_test.shape[2]} features each.",
            ])
        )

    def train_model(self, train_stops: Union[int, list[int]]) -> Union[dict, list[dict]]:
        """Model training.

        Args:
            train_stops (Union[int, list[int]]): Maximum number of data points (context
                windows) that are taken into account.

        Returns:
            Union[dict, list[dict]]: Results of the model training.
        """
        self._standardize_data()
        tensor_contexts = {
            "train": self.tensor_context_train,
            "test": self.tensor_context_test,
        }

        if isinstance(train_stops, int):
            print(f"\nModel training: Training points: {train_stops}")
            results = runner(self.model, tensor_contexts, train_stops)
        else:
            results = []
            for stop in train_stops:
                print(f"\nModel training: Training points: {stop}")
                results.append(runner(self.model, tensor_contexts, stop))

        self._results = results
        return results
