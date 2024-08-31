"""Utils for models."""

from klearn_tcyclone.performance_benchmark import timer
import numpy as np
from numpy.typing import NDArray
from kooplearn.data import TensorContextDataset


def runner(model, contexts, stop: int) -> dict:
    """Runs the model training.

    Args:
        model (_type_): _description_
        contexts (_type_): _description_
        stop (int): _description_

    Returns:
        dict: _description_
    """
    results = {}

    # Model fitting
    model, fit_time = timer(model.fit)(contexts["train"][:stop])

    # One-step prediction
    X_pred_train = model.predict(contexts["train"])
    X_true_train = contexts["train"].lookforward(model.lookback_len)
    X_pred_test = model.predict(contexts["test"])
    X_true_test = contexts["test"].lookforward(model.lookback_len)

    results["train_stop"] = stop
    results["fit_time"] = fit_time
    results["RMSE_onestep_train_error"] = np.sqrt(
        np.mean((X_pred_train - X_true_train) ** 2)
    )
    results["RMSE_onestep_test_error"] = np.sqrt(
        np.mean((X_pred_test - X_true_test) ** 2)
    )

    print_str = " ".join(
        [
            r"Fitting of model took {:.2f}s".format(results["fit_time"]),
            r"with train RMSE of {:.5f} and test RMSE of {:.5f}.".format(
                results["RMSE_onestep_train_error"], results["RMSE_onestep_test_error"]
            ),
        ]
    )
    print(print_str)

    return results


def predict_context_shift(
    model, initial_context: TensorContextDataset
) -> TensorContextDataset:
    """Predict the next context windows based on the initial_context_window.

    The predicted context is shifted by one time step to the future.

    Args:
        model (_type_): _description_
        initial_context (TensorContextDataset): _description_

    Returns:
        TensorContextDataset: _description_
    """
    next_step = model.predict(initial_context)
    helper_array = np.concatenate([initial_context.data[:, 1:], next_step], axis=1)
    next_context = TensorContextDataset(helper_array)
    return next_context


def predict_time_series(
    model, initial_context: TensorContextDataset, n_steps: int
) -> NDArray:
    """Predict time series based on initial context window.

    Args:
        model (_type_): _description_
        initial_context (TensorContextDataset): _description_
        n_steps (int): _description_

    Returns:
        NDArray: _description_
    """
    time_series_data = np.empty(shape=(0,))
    time_series_data = np.concatenate([time_series_data, [1, 2, 3]])
    time_series_data = []
    current_context = initial_context
    for _ in range(n_steps):
        current_context = predict_context_shift(model, current_context)
        time_series_data.append(current_context.data[0, -1])
    time_series_data = np.array(time_series_data)
    return time_series_data