"""Utils for Koopman model."""

import copy
from xarray import Dataset
import os
import time

import numpy as np
import torch
from torch import nn
from torch.utils import data

from klearn_tcyclone.climada.tc_tracks import TCTracks
from klearn_tcyclone.data_utils import (
    LinearScaler,
)
from klearn_tcyclone.KNF.modules.eval_metrics import RMSE_TCTracks
from klearn_tcyclone.KNF.modules.models import Koopman
from klearn_tcyclone.KNF.modules.train_utils import (
    eval_epoch_koopman,
    train_epoch_koopman,
)
from klearn_tcyclone.knf_data_utils import TCTrackDataset
from klearn_tcyclone.models_utils import get_model_name

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict_koopman_model(
    time_series: torch.Tensor,
    prediction_steps: int,
    model: Koopman,
    num_steps: int = None,
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
    tgts_dummy = torch.zeros((1, prediction_steps, time_series.shape[1])).to(device)
    time_series_unsqueeze = time_series.unsqueeze(0).to(device)
    model_copy = copy.copy(model)
    if isinstance(num_steps, int):
        model_copy.num_steps = num_steps
    res = model_copy(time_series_unsqueeze, tgts_dummy)
    # The zeroth axis has length 1 (batch_size = 1), so just squeeze this dimension again.
    denorm_outs = res[0][0]
    return denorm_outs


def train_KNF_model(
    tc_tracks_train: TCTracks | list[Dataset],
    tc_tracks_test: TCTracks | list[Dataset],
    feature_list: list,
    flag_params: dict,
    logger,
    results_dir: str,
    save_model: str = "best",
) -> Koopman:
    """Train KNF model.

    Args:
        tc_tracks_train (TCTracks | list[Dataset]): _description_
        tc_tracks_test (TCTracks | list[Dataset]): _description_
        feature_list (list): _description_
        flag_params (dict): _description_
        logger (_type_): _description_
        results_dir (str): _description_
        save_model (str, optional): If model should be saved. For "best" only the best
            model is save, for "all" the model after each epoch is saved. For anything
            else, no model is saved. Defaults to "best".

    Raises:
        Exception: _description_
        ValueError: _description_

    Returns:
        Koopman: _description_
    """
    encoder_hidden_dim = flag_params["hidden_dim"]
    decoder_hidden_dim = flag_params["hidden_dim"]
    encoder_num_layers = flag_params["num_layers"]
    decoder_num_layers = flag_params["num_layers"]
    output_dim = flag_params["input_dim"]
    num_feats = len(feature_list)
    learning_rate = flag_params["learning_rate"]

    scaler = LinearScaler()
    eval_metric = RMSE_TCTracks

    train_set = TCTrackDataset(
        input_length=flag_params["input_length"],
        output_length=flag_params["train_output_length"],
        tc_tracks=tc_tracks_train,
        feature_list=feature_list,
        mode="train",
        jumps=flag_params["jumps"],
        scaler=scaler,
        fit=True,
    )
    valid_set = TCTrackDataset(
        input_length=flag_params["input_length"],
        output_length=flag_params["train_output_length"],
        tc_tracks=tc_tracks_train,
        feature_list=feature_list,
        mode="valid",
        jumps=flag_params["jumps"],
        scaler=scaler,
        fit=False,
    )
    test_set = TCTrackDataset(
        input_length=flag_params["input_length"],
        output_length=flag_params["test_output_length"],
        tc_tracks=tc_tracks_test,
        feature_list=feature_list,
        mode="test",
        # jumps=flag_params["jumps"], # jumps not used in test mode
        scaler=scaler,
        fit=False,
    )
    train_loader = data.DataLoader(
        train_set, batch_size=flag_params["batch_size"], shuffle=True, num_workers=1
    )
    valid_loader = data.DataLoader(
        valid_set, batch_size=flag_params["batch_size"], shuffle=True, num_workers=1
    )
    test_loader = data.DataLoader(
        test_set, batch_size=flag_params["batch_size"], shuffle=False, num_workers=1
    )

    if len(train_loader) == 0:
        raise Exception(
            "There are likely too few data points in the test set. Try to increase year_range."
        )

    model_name = get_model_name(flag_params)
    # model_name = "seed{}_jumps{}_freq{}_poly{}_sin{}_exp{}_bz{}_lr{}_decay{}_dim{}_inp{}_pred{}_num{}_enchid{}_dechid{}_trm{}_conhid{}_enclys{}_declys{}_trmlys{}_conlys{}_latdim{}_RevIN{}_insnorm{}_regrank{}_globalK{}_contK{}".format(  # noqa: E501, UP032
    #     flag_params["seed"],
    #     flag_params["jumps"],
    #     flag_params["data_freq"],
    #     flag_params["num_poly"],
    #     flag_params["num_sins"],
    #     flag_params["num_exp"],
    #     flag_params["batch_size"],
    #     flag_params["learning_rate"],
    #     flag_params["decay_rate"],
    #     flag_params["input_dim"],
    #     flag_params["input_length"],
    #     flag_params["train_output_length"],
    #     flag_params["num_steps"],
    #     encoder_hidden_dim,
    #     decoder_hidden_dim,
    #     flag_params["transformer_dim"],
    #     flag_params["control_hidden_dim"],
    #     encoder_num_layers,
    #     decoder_num_layers,
    #     flag_params["transformer_num_layers"],
    #     flag_params["control_num_layers"],
    #     flag_params["latent_dim"],
    #     flag_params["use_revin"],
    #     flag_params["use_instancenorm"],
    #     flag_params["regularize_rank"],
    #     flag_params["add_global_operator"],
    #     flag_params["add_control"],
    # )

    results_file_name = os.path.join(results_dir, model_name)

    if os.path.exists(results_file_name + ".pth"):
        model, last_epoch, learning_rate = torch.load(results_file_name + ".pth")
        logger.info("Resume Training")
        logger.info(f"last_epoch: {last_epoch}, learning_rate: {learning_rate}")
    else:
        last_epoch = 0
        # os.makedirs(results_dir, exist_ok=True)
        model = Koopman(
            # number of steps of historical observations encoded at every step
            input_dim=flag_params["input_dim"],
            # input length of ts
            input_length=flag_params["input_length"],
            # number of output features
            output_dim=output_dim,
            # number of prediction steps every forward pass
            num_steps=flag_params["num_steps"],
            # hidden dimension of encoder
            encoder_hidden_dim=encoder_hidden_dim,
            # hidden dimension of decoder
            decoder_hidden_dim=decoder_hidden_dim,
            # number of layers in the encoder
            encoder_num_layers=encoder_num_layers,
            # number of layers in the decoder
            decoder_num_layers=decoder_num_layers,
            # number of feature
            num_feats=num_feats,
            # dimension of finite koopman space
            latent_dim=flag_params["latent_dim"],
            # whether to learn a global operator shared across all time series
            add_global_operator=flag_params["add_global_operator"],
            # whether to use a feedback module
            add_control=flag_params["add_control"],
            # hidden dim in the control module
            control_hidden_dim=flag_params["control_hidden_dim"],
            # number of layers in the control module
            use_revin=flag_params["use_revin"],
            # whether to use reversible normalization
            control_num_layers=flag_params["control_num_layers"],
            # whether to use instance normalization on hidden states
            use_instancenorm=flag_params["use_instancenorm"],
            # Regularize rank.
            regularize_rank=flag_params["regularize_rank"],
            # number of pairs of sine and cosine measurement functions
            num_sins=flag_params["num_sins"],
            # the highest order of polynomial functions
            num_poly=flag_params["num_poly"],
            # number of exponential functions
            num_exp=flag_params["num_exp"],
            # Number of the head the transformer encoder
            num_heads=flag_params["num_heads"],
            # hidden dimension of tranformer encoder
            transformer_dim=flag_params["transformer_dim"],
            # number of layers in the transformer encoder
            transformer_num_layers=flag_params["transformer_num_layers"],
            # dropout rate of MLP modules
            dropout_rate=flag_params["dropout_rate"],
            # global_local_combination
            global_local_combination=flag_params["global_local_combination"],
        ).to(device)

        logger.info("New model")
    logger.info(
        f"number of params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}",
        # sum(p.numel() for p in model.parameters() if p.requires_grad),
    )

    logger.info("define optimizer")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=1, gamma=flag_params["decay_rate"]
    )  # stepwise learning rate decay
    loss_fun = nn.MSELoss()

    all_train_rmses, all_eval_rmses = [], []
    best_eval_rmse = 1e6

    logger.info("start training")
    for epoch in range(last_epoch, flag_params["num_epochs"]):
        start_time = time.time()

        train_rmse = train_epoch_koopman(
            train_loader,
            model,
            loss_fun,
            optimizer,
            regularize_rank=flag_params["regularize_rank"],
        )
        eval_rmse, _, _ = eval_epoch_koopman(
            valid_loader,
            model,
            loss_fun,
            regularize_rank=flag_params["regularize_rank"],
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
        _, test_preds, test_tgts = eval_epoch_koopman(test_loader, best_model, loss_fun)
        torch.save(
            {
                "test_preds": test_preds,
                "test_tgts": test_tgts,
                "eval_score": eval_metric(test_preds, test_tgts),
            },
            os.path.join(results_dir, f"ep{epoch}_test" + model_name + ".pt"),
        )

        # train the model at least 60 epochs and do early stopping
        if epoch > flag_params["min_epochs"] and np.mean(
            all_eval_rmses[-10:]
        ) > np.mean(all_eval_rmses[-20:-10]):
            break

        epoch_time = time.time() - start_time
        scheduler.step()
        logger.info(
            "Epoch {} | T: {:0.2f} | Train RMSE: {:0.3f} | Valid RMSE: {:0.3f}".format(  # noqa: UP032
                epoch + 1, epoch_time / 60, train_rmse, eval_rmse
            )
        )

    logger.info("Evaluate test metric.")
    _, test_preds, test_tgts = eval_epoch_koopman(test_loader, best_model, loss_fun)
    torch.save(
        {
            "test_preds": test_preds,
            "test_tgts": test_tgts,
            "eval_score": eval_metric(test_preds, test_tgts),
            "train_rmses": all_train_rmses,
            "eval_rmses": all_eval_rmses,
        },
        os.path.join(results_dir, "test_" + model_name + ".pt"),
    )
    logger.info(f"eval_metric: {eval_metric(test_preds, test_tgts)}")

    return model, all_train_rmses
