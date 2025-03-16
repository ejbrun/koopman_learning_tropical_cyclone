"""Utils for Koopman Kernel Seq2Seq architecture."""

import os
# import time

from time import time

# from datetime import datetime
import logging

import numpy as np
import torch
import json
from torch.optim import Optimizer
from kooplearn.data import TensorContextDataset
from matplotlib.axes._axes import Axes
from sklearn.model_selection import train_test_split
from klearn_tcyclone.models_utils import get_model_name

# from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.nn.utils import clip_grad_norm_
from xarray import Dataset

from klearn_tcyclone.climada.tc_tracks import TCTracks
from klearn_tcyclone.data_utils import (
    LinearScaler,
    standardized_context_dataset_from_TCTracks,
)
from klearn_tcyclone.KNF.modules.eval_metrics import RMSE_TCTracks
from klearn_tcyclone.knf_data_utils import TCTrackDataset
from klearn_tcyclone.koopkernel_seq2seq import (
    KoopKernelLoss,
    NystroemKoopKernelSequencer,
    RBFKernel,
)
from klearn_tcyclone.plot_utils import plot_TCTrackDataset_item_2D

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def batch_tensor_context(
    tensor_context: TensorContextDataset,
    batch_size: int,
    input_length: int,
    output_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get batched tensor context.

    Args:
        tensor_context (TensorContextDataset): Tensor context of shape
            (n_data, input_length + output_length, num_feats).
        batch_size (int): Batch size.
        input_length (int): Input length.
        output_length (int): Output length.

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
    if tensor_context.context_length != input_length + output_length:
        raise Exception(
            f"""
            tensor_context.context_lenght (={tensor_context.context_length}) must be
            equal to input_length (={input_length}) + output_length (={output_length}).
            """
        )
    if output_length == 1:
        tensor_context_inps = torch.tensor(
            tensor_context.lookback(input_length), dtype=torch.float32
        ).to(device)
        # shape: (n_data, input_length, num_feats)
        tensor_context_tgts = torch.tensor(
            tensor_context.lookback(input_length, slide_by=1),
            dtype=torch.float32,
        ).to(device)
        # shape: (n_data, input_length, num_feats)
    else:
        tensor_context_inps = torch.tensor(
            tensor_context.lookback(input_length), dtype=torch.float32
        ).to(device)
        # shape: (n_data, input_length, num_feats)
        tensor_context_tgts = torch.tensor(
            np.array([
                tensor_context.lookback(input_length, slide_by=idx + 1)
                for idx in range(output_length)
            ]),
            dtype=torch.float32,
        ).to(device)
        # shape: (output_length, n_data, input_length, num_feats)
        tensor_context_tgts = torch.einsum("abcd->bcad", tensor_context_tgts)
        # shape: (n_data, input_length, output_length, num_feats)

    # FIXME add random seed to randperm.
    rand_perm = torch.randperm(tensor_context_inps.shape[0])
    integer_divisor = tensor_context_inps.shape[0] // batch_size

    tensor_context_inps = tensor_context_inps[rand_perm]
    tensor_context_tgts = tensor_context_tgts[rand_perm]

    tensor_context_inps = tensor_context_inps[: integer_divisor * batch_size]
    tensor_context_tgts = tensor_context_tgts[: integer_divisor * batch_size]

    tensor_context_inps = tensor_context_inps.reshape(
        shape=[
            batch_size,
            integer_divisor,
            *tensor_context_inps.shape[1:],
        ]
    )
    # shape: (batch_size, n_data // batch_size, input_length, num_feats)
    tensor_context_tgts = tensor_context_tgts.reshape(
        shape=[
            batch_size,
            integer_divisor,
            *tensor_context_tgts.shape[1:],
        ]
    )
    # shape: (batch_size, n_data // batch_size, input_length, output_length, num_feats)

    return tensor_context_inps, tensor_context_tgts


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


def train_one_epoch(
    model: NystroemKoopKernelSequencer,
    optimizer: Optimizer,
    loss_fun: KoopKernelLoss,
    tensor_context_inps: torch.tensor,
    tensor_context_tgts: torch.tensor,
) -> float:
    """Train one epoch.

    Args:
        model (NystroemKoopKernelSequencer): _description_
        optimizer (Optimizer): _description_
        loss_fun (KoopKernelLoss): _description_
        tensor_context_inps (torch.tensor): Input tensor context with shape
            (batch_size, n_data // batch_size, input_length, num_feats).
        tensor_context_tgts (torch.tensor): Output tensor context with shape
            output_length = 1:
            (batch_size, n_data // batch_size, input_length, num_feats), 
            output_length > 1:
            (batch_size, n_data // batch_size, input_length, output_length, num_feats).
        
    Returns:
        float: Square root of MSL.
    """
    train_loss = []
    for i in range(tensor_context_inps.shape[1]):
        # Every data instance is an input + label pair
        inputs, labels = tensor_context_inps[:, i], tensor_context_tgts[:, i]
        if model.context_mode == "last_context":
            #FIXME This might not work for output_length > 1.
            labels = labels[:, -1, :]

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss
        loss = loss_fun(outputs, labels)
        train_loss.append(loss.item())

        # Zero your gradients for every batch, and compute loss gradient.
        optimizer.zero_grad()
        loss.backward()

        # clip_grad_norm_(model.parameters(), max_norm=5.0, norm_type=2)

        # Adjust learning weights
        optimizer.step()

    return np.sqrt(np.mean(train_loss))


def eval_one_epoch(
    model: NystroemKoopKernelSequencer,
    loss_fun: KoopKernelLoss,
    tensor_context_inps: torch.tensor,
    tensor_context_tgts: torch.tensor,
):
    eval_loss = []
    all_preds = []
    all_trues = []
    for i in range(tensor_context_inps.shape[1]):
        # Every data instance is an input + label pair
        inputs, labels = tensor_context_inps[:, i], tensor_context_tgts[:, i]
        if model.context_mode == "last_context":
            labels = labels[:, -1, :]

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss
        loss = loss_fun(outputs, labels)
        eval_loss.append(loss.item())
        all_preds.append(outputs.cpu().data.numpy())
        all_trues.append(labels.cpu().data.numpy())

    return (
        np.sqrt(np.mean(eval_loss)),
        np.concatenate(all_preds, axis=0),
        np.concatenate(all_trues, axis=0),
    )


def train_one_epoch_old(
    model: NystroemKoopKernelSequencer,
    optimizer,
    loss_fun: KoopKernelLoss,
    epoch_index,
    tb_writer,
    tensor_context_inps,
    tensor_context_tgts,
):
    """From https://pytorch.org/tutorials/beginner/introyt/trainingyt.html."""

    running_loss = 0.0
    last_loss = 0.0

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting

    for i in range(tensor_context_inps.shape[1]):
        # Every data instance is an input + label pair
        inputs, labels = tensor_context_inps[:, i], tensor_context_tgts[:, i]
        if model.context_mode == "last_context":
            labels = labels[:, -1, :]

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fun(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 10 == 9:
            last_loss = running_loss / 10  # loss per batch
            print("  batch {} loss: {}".format(i + 1, last_loss))
            tb_x = epoch_index * tensor_context_inps.shape[1] + i + 1
            tb_writer.add_scalar("Loss/train", last_loss, tb_x)
            running_loss = 0.0

    return last_loss


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


def predict_koopman_seq2seq_model(
    time_series: torch.Tensor,
    prediction_steps: int,
    model: NystroemKoopKernelSequencer,
):
    assert len(time_series.shape) == 2
    # tgts_dummy only needed to set the number of predicted time steps by model.
    """
    Note: The model performs a number of single_forward steps, each predicting
    `num_steps` future time steps, such that the total number of predicted time steps is
    larger than `tgts.shape[1]`, i.e. `prediction_steps` or the `output_length` of the
    dataset. In the end only the first `output_length` predicted data points are given
    as output, the remaining (further) steps are discarded.
    """

    num_feats = time_series.shape[1]

    if model.context_mode == "no_context":
        time_series_unsqueeze = time_series.unsqueeze(0).to(device)
    elif model.context_mode == "full_context":
        time_series_unsqueeze = (
            time_series[-model.input_length :].unsqueeze(0).to(device)
        )
    if model.context_mode == "last_context":
        time_series_unsqueeze = (
            time_series[-model.input_length :].unsqueeze(0).to(device)
        )

    n_eval_steps = int(prediction_steps // model.output_length)
    if n_eval_steps * model.output_length < prediction_steps:
        n_eval_steps += 1

    predictions = torch.zeros(
        size=(
            n_eval_steps * model.output_length,
            num_feats,
        ),
        device=device,
        dtype=torch.float32,
    )
    prediction = time_series_unsqueeze
    # shape: (1, input_length, num_feats)

    for idx in range(n_eval_steps):
        new_prediction = model(prediction)
        if model.context_mode == "last_context":
            if model.output_length == 1:
                new_prediction = new_prediction.unsqueeze(1)
            prediction = torch.cat(
                [prediction[:, model.output_length :], new_prediction], dim=1
            )
            # shape: (1, input_length, num_feats)
        else:
            if model.output_length == 1:
                new_prediction = new_prediction.unsqueeze(2)
            prediction = torch.cat(
                [prediction[:, model.output_length :], new_prediction[:, -1]], dim=1
            )
            # shape: (1, input_length, num_feats)
        predictions[idx * model.output_length : (idx + 1) * model.output_length] = (
            prediction[0, -model.output_length :]
        )
    # shape: (n_eval_steps, num_feats)

    print(predictions.shape)

    return predictions[:prediction_steps]


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
    predictions = predict_koopman_seq2seq_model(time_series, prediction_steps, model)
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
        early_stopping = early_stopping,
    )

    return model, all_train_rmses
