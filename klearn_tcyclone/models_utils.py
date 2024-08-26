"""Utils for models."""

from klearn_tcyclone.performance_benchmark import timer
import numpy as np


def runner(model, contexts, stop) -> dict:
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
        r"Fitting of model took {:.2f}s".format(results["fit_time"]),
        r"with train RMSE of {:.5f} and test RMSE of {:.5f}.".format(
            results["RMSE_onestep_train_error"], results["RMSE_onestep_test_error"]
        ),
    )
    print(print_str)

    return results
