"""Utils for models."""

from typing import Union

import numpy as np
from kooplearn.data import TensorContextDataset
from kooplearn.models import Kernel, NystroemKernel
from numpy.typing import NDArray

from klearn_tcyclone.performance_benchmark import timer


def runner(model: Union[Kernel, NystroemKernel], contexts, stop: int) -> dict:
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
    results["X_pred_train"] = X_pred_train
    results["X_true_train"] = X_true_train
    results["X_pred_test"] = X_pred_test
    results["X_true_test"] = X_true_test

    # results["RMSE_onestep_train_error"] = np.sqrt(
    #     np.mean((X_pred_train - X_true_train) ** 2)
    # )
    # results["RMSE_onestep_test_error"] = np.sqrt(
    #     np.mean((X_pred_test - X_true_test) ** 2)
    # )

    # print_str = " ".join(
    #     [
    #         r"Fitting of model took {:.2f}s".format(results["fit_time"]),
    #         r"with train RMSE of {:.5f} and test RMSE of {:.5f}.".format(
    #             results["RMSE_onestep_train_error"], results["RMSE_onestep_test_error"]
    #         ),
    #     ]
    # )
    # print(print_str)

    return results


def predict_context_shift(
    model: Union[Kernel, NystroemKernel], initial_context: TensorContextDataset
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
    model: Kernel | NystroemKernel,
    initial_context: TensorContextDataset,
    n_steps: int,
    context_length: int,
) -> NDArray:
    """Predict time series based on initial context window.

    Args:
        model (_type_): _description_
        initial_context (TensorContextDataset): _description_
        n_steps (int): _description_
        context_length (int): _description_

    Returns:
        NDArray: _description_
    """
    time_series_data = []
    current_context = initial_context[:, -context_length:]
    for _ in range(n_steps):
        current_context = predict_context_shift(model, current_context)
        time_series_data.append(current_context.data[:, -1])
    time_series_data = np.array(time_series_data).transpose((1, 0, 2))
    return time_series_data


def get_model_name(flag_params: dict) -> str:
    """Get model name.

    Args:
        flag_params (dict): _description_

    Raises:
        Exception: _description_

    Returns:
        str: _description_
    """
    if flag_params["model"] == "KNF":
        model_name = "seed{}_jumps{}_freq{}_poly{}_sin{}_exp{}_bz{}_lr{}_decay{}_dim{}_inp{}_pred{}_num{}_enchid{}_dechid{}_trm{}_conhid{}_enclys{}_declys{}_trmlys{}_conlys{}_latdim{}_RevIN{}_insnorm{}_regrank{}_globalK{}_contK{}".format(  # noqa: E501, UP032
            flag_params["seed"],
            flag_params["jumps"],
            flag_params["data_freq"],
            flag_params["num_poly"],
            flag_params["num_sins"],
            flag_params["num_exp"],
            flag_params["batch_size"],
            flag_params["learning_rate"],
            flag_params["decay_rate"],
            flag_params["input_dim"],
            flag_params["input_length"],
            flag_params["train_output_length"],
            flag_params["num_steps"],
            flag_params["hidden_dim"],
            flag_params["hidden_dim"],
            flag_params["transformer_dim"],
            flag_params["control_hidden_dim"],
            flag_params["num_layers"],
            flag_params["num_layers"],
            flag_params["transformer_num_layers"],
            flag_params["control_num_layers"],
            flag_params["latent_dim"],
            flag_params["use_revin"],
            flag_params["use_instancenorm"],
            flag_params["regularize_rank"],
            flag_params["add_global_operator"],
            flag_params["add_control"],
        )
    elif flag_params["model"] == "koopkernelseq":
        model_name = "seed{}_jumps{}_freq{}_bz{}_lr{}_decay{}_dim{}_inp{}_pred{}_num{}_kknc{}_kkls{}_ctxm{}_mko{}_mv{}_ncw{}".format(  # noqa: E501, UP032
            flag_params["seed"],
            flag_params["jumps"],
            flag_params["data_freq"],
            flag_params["batch_size"],
            flag_params["learning_rate"],
            flag_params["decay_rate"],
            flag_params["input_dim"],
            flag_params["input_length"],
            flag_params["train_output_length"],
            flag_params["num_steps"],
            flag_params["koopman_kernel_num_centers"],
            flag_params["koopman_kernel_length_scale"],
            flag_params["context_mode"],
            flag_params["mask_koopman_operator"],
            flag_params["mask_version"],
            flag_params["use_nystroem_context_window"],
        )
    else:
        raise Exception("Wrong model_str.")

    return model_name


def get_model_name_old(flag_params: dict) -> str:
    """Get model name.

    Args:
        flag_params (dict): _description_

    Raises:
        Exception: _description_

    Returns:
        str: _description_
    """
    if flag_params["model"] == "KNF":
        encoder_hidden_dim = flag_params["hidden_dim"]
        decoder_hidden_dim = flag_params["hidden_dim"]
        encoder_num_layers = flag_params["num_layers"]
        decoder_num_layers = flag_params["num_layers"]
        model_name = "seed{}_jumps{}_freq{}_poly{}_sin{}_exp{}_bz{}_lr{}_decay{}_dim{}_inp{}_pred{}_num{}_enchid{}_dechid{}_trm{}_conhid{}_enclys{}_declys{}_trmlys{}_conlys{}_latdim{}_RevIN{}_insnorm{}_regrank{}_globalK{}_contK{}".format(  # noqa: E501, UP032
            flag_params["seed"],
            flag_params["jumps"],
            flag_params["data_freq"],
            flag_params["num_poly"],
            flag_params["num_sins"],
            flag_params["num_exp"],
            flag_params["batch_size"],
            flag_params["learning_rate"],
            flag_params["decay_rate"],
            flag_params["input_dim"],
            flag_params["input_length"],
            flag_params["train_output_length"],
            flag_params["num_steps"],
            encoder_hidden_dim,
            decoder_hidden_dim,
            flag_params["transformer_dim"],
            flag_params["control_hidden_dim"],
            encoder_num_layers,
            decoder_num_layers,
            flag_params["transformer_num_layers"],
            flag_params["control_num_layers"],
            flag_params["latent_dim"],
            flag_params["use_revin"],
            flag_params["use_instancenorm"],
            flag_params["regularize_rank"],
            flag_params["add_global_operator"],
            flag_params["add_control"],
        )
    elif flag_params["model"] == "koopkernelseq":
        model_name = "seed{}_jumps{}_freq{}_bz{}_lr{}_decay{}_dim{}_inp{}_pred{}_num{}_kknc{}".format(  # noqa: E501, UP032
            flag_params["seed"],
            flag_params["jumps"],
            flag_params["data_freq"],
            flag_params["batch_size"],
            flag_params["learning_rate"],
            flag_params["decay_rate"],
            flag_params["input_dim"],
            flag_params["input_length"],
            flag_params["train_output_length"],
            flag_params["num_steps"],
            flag_params["koopman_kernel_num_centers"],
        )
    else:
        raise Exception("Wrong model_str.")

    return model_name
