from klearn_tcyclone.data_utils import (
    context_dataset_from_TCTracks,
    standardize_TensorContextDataset,
)
from klearn_tcyclone.models_utils import runner


class ModelBenchmarkError(Exception):
    """A custom exception used to report errors in use of Timer class"""


class ModelBenchmark:
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
        # self.tc_tracks_train = tc_tracks_train
        # self.tc_tracks_test = tc_tracks_test
        self.scaler = scaler
        self.context_length = context_length
        self._set_tensor_contexts(tc_tracks_train, tc_tracks_test)
        self._results = None

    @property
    def results(self):
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
            " ".join(
                f"Train contexts have shape {self.tensor_context_train.shape}:",
                f"{len(self.tensor_context_train)} contexts of length",
                f"{self.tensor_context_train.context_length} with",
                f"{self.tensor_context_train.shape[2]} features each.",
            )
        )
        print(
            " ".join(
                f"Test contexts have shape {self.tensor_context_test.shape}:",
                f"{len(self.tensor_context_test)} contexts of length",
                f"{self.tensor_context_test.context_length} with",
                f"{self.tensor_context_test.shape[2]} features each.",
            )
        )

    def train_model(self, train_stops: int | list[int]) -> dict | list[dict]:
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
