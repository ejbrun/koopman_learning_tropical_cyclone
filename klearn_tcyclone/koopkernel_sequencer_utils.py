"""Utils for Koopman Kernel Seq2Seq architecture."""

import json

# from datetime import datetime
import logging
import os

# import time
from time import time

import numpy as np
import torch
from kkseq.koopkernel_sequencer import (
    KoopKernelLoss,
    NystroemKoopKernelSequencer,
    RBFKernel,
)
from kkseq.koopkernel_sequencer_utils import (
    batch_tensor_context,
    eval_one_epoch,
    predict_koopkernel_sequencer,
    train_one_epoch,
)
from matplotlib.axes._axes import Axes
from sklearn.model_selection import train_test_split

# from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from xarray import Dataset

from klearn_tcyclone.climada.tc_tracks import TCTracks
from klearn_tcyclone.data_utils import (
    LinearScaler,
    standardized_context_dataset_from_TCTracks,
)
from klearn_tcyclone.KNF.modules.eval_metrics import RMSE_TCTracks
from klearn_tcyclone.knf_data_utils import TCTrackDataset
from klearn_tcyclone.models_utils import get_model_name
from klearn_tcyclone.plot_utils import plot_TCTrackDataset_item_2D

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def standardized_batched_context_from_TCTracks(
    tc_tracks: TCTracks,
    batch_size: int,
    feature_list: list[str],
    scaler: StandardScaler | MinMaxScaler | LinearScaler,
    context_length: int = 2,
    time_lag: int = 1,
    fit: bool = True,
    periodic_shift: bool = True,
    basin: str | None = None,
    verbose: int = 0,
    input_length: int | None = None,
    output_length: int | None = None,
    backend: str = "auto",
    **backend_kw,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate standardized and batched tensor contexts for inputs and outputs.

    Args:
        tc_tracks (TCTracks): _description_
        batch_size (int): _description_
        feature_list (list[str]): _description_
        scaler (StandardScaler | MinMaxScaler | LinearScaler): _description_
        context_length (int, optional): _description_. Defaults to 2.
        time_lag (int, optional): _description_. Defaults to 1.
        fit (bool, optional): _description_. Defaults to True.
        periodic_shift (bool, optional): _description_. Defaults to True.
        basin (str | None, optional): _description_. Defaults to None.
        verbose (int, optional): _description_. Defaults to 0.
        input_length (int | None, optional): _description_. Defaults to None.
        output_length (int | None, optional): _description_. Defaults to None.
        backend (str, optional): _description_. Defaults to "auto".

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Tuple of batched input and output tensor
            contexts, with shapes
            output_length = 1:
            (batch_size, n_data // batch_size, input_length, num_feats) and
            (batch_size, n_data // batch_size, input_length, num_feats).
            output_length > 1:
            (batch_size, n_data // batch_size, input_length, num_feats) and
            (batch_size, n_data // batch_size, input_length, output_length, num_feats).
    """
    tensor_context = standardized_context_dataset_from_TCTracks(
        tc_tracks,
        feature_list,
        scaler,
        context_length,
        time_lag,
        fit,
        periodic_shift,
        basin,
        verbose,
        input_length,
        output_length,
        backend,
        **backend_kw,
    )
    # shape: (n_data, input_length + output_length, num_feats)
    tensor_context_inps, tensor_context_tgts = batch_tensor_context(
        tensor_context,
        batch_size=batch_size,
        input_length=input_length,
        output_length=output_length,
    )
    # shapes: (batch_size, n_data // batch_size, input_length, num_feats) or
    # (batch_size, n_data // batch_size, input_length, output_length, num_feats)
    return tensor_context_inps, tensor_context_tgts


def train_KKSeq2Seq(
    model: NystroemKoopKernelSequencer,
    eval_metric: RMSE_TCTracks,
    tc_tracks: TCTracks | list[Dataset],
    num_epochs: int,
    batch_size: int,
    feature_list,
    scaler: StandardScaler | MinMaxScaler | LinearScaler,
    basin: str,
    log_file_handler: logging.FileHandler,
    results_dir: str,
    model_name: str,
    flag_params: dict,
    results_file_name: str,
    save_model: str = "best",
    split_valid_set: bool = True,
    early_stopping: bool = False,
    backend: str = "auto",
    **backend_kw,
) -> tuple[NystroemKoopKernelSequencer, list[float]]:
    """Train Koopman kernal sequence model.

    Args:
        model (NystroemKoopKernelSequencer): _description_
        eval_metric (RMSE_TCTracks): _description_
        tc_tracks (TCTracks | list[Dataset]): _description_
        num_epochs (int): _description_
        batch_size (int): _description_
        feature_list (_type_): _description_
        scaler (_type_): _description_
        basin (str): _description_
        input_length (int): _description_
        output_length (int): _description_
        decay_rate (float): _description_
        learning_rate (float): _description_
        log_file_handler (_type_): _description_
        results_dir (str): _description_
        model_name (str): _description_
        flag_params (dict): _description_
        results_file_name (str): _description_
        save_model (str, optional): If model should be saved. For "best" only the best
            model is save, for "all" the model after each epoch is saved. For anything
            else, no model is saved. Defaults to "best".
        early_stopping (bool): If to apply early stopping. Defaults to False.
        backend (str, optional): _description_. Defaults to "auto".

    Raises:
        ValueError: _description_

    Returns:
        NystroemKoopKernelSequencer: _description_
    """
    logger.addHandler(log_file_handler)

    tc_tracks_train, tc_tracks_test = train_test_split(
        tc_tracks.data, test_size=0.1, random_state=flag_params["seed"]
    )
    if split_valid_set:
        tc_tracks_train, tc_tracks_valid = train_test_split(
            tc_tracks_train, test_size=0.1, random_state=flag_params["seed"] + 1
        )
    else:
        tc_tracks_valid = tc_tracks_test

    tensor_context_train_standardized = standardized_context_dataset_from_TCTracks(
        tc_tracks_train,
        feature_list=feature_list,
        scaler=scaler,
        context_length=flag_params["context_length"],
        time_lag=1,
        fit=True,
        periodic_shift=True,
        basin=basin,
        input_length=flag_params["input_length"],
        output_length=flag_params["train_output_length"],
    )

    model._initialize_nystrom_data(tensor_context_train_standardized)
    del tensor_context_train_standardized

    tensor_context_inps_train, tensor_context_tgts_train = (
        standardized_batched_context_from_TCTracks(
            tc_tracks_train,
            batch_size,
            feature_list,
            scaler,
            context_length=flag_params["context_length"],
            time_lag=1,
            fit=True,
            periodic_shift=True,
            basin=basin,
            input_length=flag_params["input_length"],
            output_length=flag_params["train_output_length"],
            backend=backend,
            **backend_kw,
        )
    )
    # shapes: (batch_size, n_data // batch_size, input_length, num_feats) or
    # (batch_size, n_data // batch_size, input_length, output_length, num_feats)
    tensor_context_inps_valid, tensor_context_tgts_valid = (
        standardized_batched_context_from_TCTracks(
            tc_tracks_valid,
            batch_size,
            feature_list,
            scaler,
            context_length=flag_params["context_length"],
            time_lag=1,
            fit=False,
            periodic_shift=True,
            basin=basin,
            input_length=flag_params["input_length"],
            output_length=flag_params["train_output_length"],
            backend=backend,
            **backend_kw,
        )
    )
    tensor_context_inps_test, tensor_context_tgts_test = (
        standardized_batched_context_from_TCTracks(
            tc_tracks_test,
            batch_size,
            feature_list,
            scaler,
            context_length=flag_params["context_length"],
            time_lag=1,
            fit=False,
            periodic_shift=True,
            basin=basin,
            input_length=flag_params["input_length"],
            output_length=flag_params["train_output_length"],
            backend=backend,
            **backend_kw,
        )
    )
    del tc_tracks_train
    del tc_tracks_valid
    del tc_tracks_test

    if flag_params["train_output_length"] == 1:
        assert torch.all(
            tensor_context_inps_train[:, :, 1:] == tensor_context_tgts_train[:, :, :-1]
        )
    else:
        for idx in range(flag_params["train_output_length"]):
            assert torch.all(
                tensor_context_inps_train[:, :, idx + 1 :]
                == tensor_context_tgts_train[:, :, : -idx - 1, idx]
            )

    optimizer = torch.optim.Adam(model.parameters(), lr=flag_params["learning_rate"])
    loss_koopkernel = KoopKernelLoss(model.nystrom_data_Y, model._kernel)

    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # tb_writer = SummaryWriter("runs/fashion_trainer_{}".format(timestamp))

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=1, gamma=flag_params["decay_rate"]
    )  # stepwise learning rate decay

    all_train_rmses, all_eval_rmses = [], []
    best_eval_rmse = 1e6

    training_time_start = time()
    for epoch_index, epoch in enumerate(range(num_epochs)):
        start_time = time()

        train_rmse = train_one_epoch(
            model,
            optimizer,
            loss_koopkernel,
            tensor_context_inps_train,
            tensor_context_tgts_train,
        )
        eval_rmse, _, _ = eval_one_epoch(
            model,
            loss_koopkernel,
            tensor_context_inps_valid,
            tensor_context_tgts_valid,
        )

        print("eval comparison", eval_rmse, best_eval_rmse)
        if save_model == "all":
            torch.save(
                [model, epoch, optimizer.param_groups[0]["lr"]],
                results_file_name + f"_epoch{epoch}" + ".pth",
            )

        if eval_rmse < best_eval_rmse:
            best_eval_rmse = eval_rmse
            best_model = model
            if save_model == "best":
                torch.save(
                    [best_model, epoch, optimizer.param_groups[0]["lr"]],
                    results_file_name + "_best.pth",
                )

        all_train_rmses.append(train_rmse)
        all_eval_rmses.append(eval_rmse)

        if np.isnan(train_rmse) or np.isnan(eval_rmse):
            raise ValueError("The model generate NaN values")

        # Save test scores.
        _, test_preds, test_tgts = eval_one_epoch(
            best_model,
            loss_koopkernel,
            tensor_context_inps_test,
            tensor_context_tgts_test,
        )
        torch.save(
            {
                "test_preds": test_preds,
                "test_tgts": test_tgts,
                "eval_score": eval_metric(
                    test_preds, test_tgts
                ),  # FIXME eval_metric() here is a tuple of four elements, why? Should be a single number.
            },
            os.path.join(results_dir, f"ep{epoch}_test" + model_name + ".pt"),
        )

        # train the model at least 60 epochs and do early stopping
        if early_stopping:
            if epoch > flag_params["min_epochs"] and np.mean(
                all_eval_rmses[-10:]
            ) > np.mean(all_eval_rmses[-20:-10]):
                break

        epoch_time = time() - start_time
        scheduler.step()
        logger.info(
            "Epoch {} | T: {:0.2f} | Train RMSE: {:0.3f} | Valid RMSE: {:0.3f}".format(  # noqa: UP032
                epoch + 1, epoch_time / 60, train_rmse, eval_rmse
            )
        )

    training_runtime = time() - training_time_start

    logger.info("Evaluate test metric.")
    _, test_preds, test_tgts = eval_one_epoch(
        best_model,
        loss_koopkernel,
        tensor_context_inps_test,
        tensor_context_tgts_test,
    )
    torch.save(
        {
            "test_preds": test_preds,
            "test_tgts": test_tgts,
            "eval_score": eval_metric(
                test_preds, test_tgts
            ),  # FIXME eval_metric() here is a tuple of four elements, why? Should be a single number.
            "train_rmses": all_train_rmses,
            "eval_rmses": all_eval_rmses,
            "training_runtime": training_runtime,
        },
        os.path.join(results_dir, "test_" + model_name + ".pt"),
    )
    # with open(os.path.join(results_dir, "test_" + model_name + ".json"), "w") as jsonfile:
    #     json.dump(
    #         {
    #             "eval_score": list(map(float, eval_metric(test_preds, test_tgts))), #FIXME eval_metric() here is a tuple of four elements, why? Should be a single number.
    #             "train_rmses": list(map(float, all_train_rmses)),
    #             "eval_rmses": list(map(float, all_eval_rmses)),
    #         },
    #         jsonfile,
    #         indent=4,
    #     )
    logger.info(f"eval_metric: {eval_metric(test_preds, test_tgts)}")

    return model, all_train_rmses


def plot_TCTrackDataset_item_2D_with_prediction_koopman_seq2seq(
    tc_tracks_dataset: TCTrackDataset,
    idx: int,
    prediction_steps,
    model,
    ax: Axes,
    color: str,
    dimensions: tuple[int, int] = (0, 1),
):
    plot_TCTrackDataset_item_2D(tc_tracks_dataset, idx, ax, color, dimensions)
    time_series, _ = tc_tracks_dataset[idx]
    predictions = predict_koopkernel_sequencer(time_series, prediction_steps, model)
    predictions = predictions.cpu().data.numpy()
    ax.plot(
        predictions[:, dimensions[0]],
        predictions[:, dimensions[1]],
        marker="s",
        color=color,
    )


def train_koopkernel_seq2seq_model(
    tc_tracks: TCTracks | list[Dataset],
    feature_list: list,
    flag_params: dict,
    basin: str,
    log_file_handler: logging.FileHandler,
    results_dir: str,
    save_model: str = "best",
    early_stopping: bool = False,
) -> tuple[NystroemKoopKernelSequencer, list[float]]:
    """Train Koopman kernel sequence model.

    Args:
        tc_tracks (TCTracks | list[Dataset]): _description_
        feature_list (list): _description_
        flag_params (dict): _description_
        basin (str): _description_
        log_file_handler (_type_): _description_
        results_dir (str): _description_
        save_model (str, optional): If model should be saved. For "best" only the best
            model is save, for "all" the model after each epoch is saved. For anything
            else, no model is saved. Defaults to "best".
        early_stopping (bool): If to apply early stopping. Defaults to False.

    Returns:
        NystroemKoopKernelSequencer: _description_
    """
    scaler = LinearScaler()

    num_feats = len(feature_list)

    eval_metric = RMSE_TCTracks

    model_name = get_model_name(flag_params)
    results_file_name = os.path.join(results_dir, model_name)

    rbf = RBFKernel(length_scale=flag_params["koopman_kernel_length_scale"])
    koopkernelmodel = NystroemKoopKernelSequencer(
        kernel=rbf,
        input_dim=num_feats,
        input_length=flag_params["input_length"],
        output_length=flag_params["train_output_length"],
        output_dim=1,
        num_steps=1,
        num_nys_centers=flag_params["koopman_kernel_num_centers"],
        rng_seed=42,
        context_mode=flag_params["context_mode"],
        mask_koopman_operator=flag_params["mask_koopman_operator"],
        mask_version=flag_params["mask_version"],
        use_nystroem_context_window=flag_params["use_nystroem_context_window"],
    )

    model, all_train_rmses = train_KKSeq2Seq(
        model=koopkernelmodel,
        eval_metric=eval_metric,
        tc_tracks=tc_tracks,
        num_epochs=flag_params["num_epochs"],
        batch_size=flag_params["batch_size"],
        feature_list=feature_list,
        scaler=scaler,
        basin=basin,
        log_file_handler=log_file_handler,
        results_dir=results_dir,
        model_name=model_name,
        flag_params=flag_params,
        results_file_name=results_file_name,
        save_model=save_model,
        early_stopping=early_stopping,
    )

    return model, all_train_rmses
