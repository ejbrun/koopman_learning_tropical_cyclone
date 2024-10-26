"""Training of models."""

import logging
import os
import random
import time

import numpy as np
import torch
from absl import app
from sklearn.model_selection import train_test_split
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
from klearn_tcyclone.training_utils.args import FLAGS
from klearn_tcyclone.training_utils.training_utils import set_flags


def main(argv):
    random.seed(FLAGS.seed)  # python random generator
    np.random.seed(FLAGS.seed)  # numpy random generator

    torch.manual_seed(FLAGS.seed)
    torch.cuda.manual_seed_all(FLAGS.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # parameters from flag
    flag_params = set_flags(FLAGS=FLAGS)

    # Logging and define save paths
    current_file_dir_path = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(
        current_file_dir_path,
        "training_results",
        "{}_yrange{}".format(
            flag_params["dataset"],
            "".join(map(str, flag_params["year_range"])),
        ),
        flag_params["model"] + "_glc{}".format(flag_params["global_local_combination"]),
    )
    logs_dir = os.path.join(
        current_file_dir_path,
        "logs",
        "{}_yrange{}".format(
            flag_params["dataset"],
            "".join(map(str, flag_params["year_range"])),
        ),
    )
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(logs_dir, flag_params["model"] + ".log"),
        encoding="utf-8",
        filemode="a",
        format="{asctime} - {name} - {filename}:{lineno} - {levelname} - {message}",
        style="{",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    logger = logging.getLogger(flag_params["model"] + "_logger")
    logger.info(flag_params)

    # Set remaining parameters
    # feature_list = ["lat", "lon"]
    feature_list = ["lat", "lon", "max_sustained_wind"]
    # feature_list = ["lat", "lon", "max_sustained_wind", "central_pressure"]

    # these are not contained as flags
    encoder_hidden_dim = flag_params["hidden_dim"]
    decoder_hidden_dim = flag_params["hidden_dim"]
    encoder_num_layers = flag_params["num_layers"]
    decoder_num_layers = flag_params["num_layers"]
    output_dim = flag_params["input_dim"]
    num_feats = len(feature_list)
    learning_rate = flag_params["learning_rate"]
    # ---------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device {device}")

    scaler = LinearScaler()
    eval_metric = RMSE_TCTracks

    # Datasets
    tc_tracks = TCTracks.from_ibtracs_netcdf(
        provider="usa",
        year_range=flag_params["year_range"],
        basin="NA",
        correct_pres=False,
    )

    tc_tracks_train, tc_tracks_test = train_test_split(tc_tracks.data, test_size=0.1)
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

    # model_folder_path = (
    #     "Koopman_"
    #     + str(flag_params["dataset"])
    #     + "_model{}_glc{}".format(
    #         flag_params["model"], flag_params["global_local_combination"]
    #     )
    # )
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
        torch.save(
            [model, epoch, optimizer.param_groups[0]["lr"]],
            results_file_name + f"_epoch{epoch}" + ".pth",
        )

        if eval_rmse < best_eval_rmse:
            best_eval_rmse = eval_rmse
            best_model = model
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
        },
        os.path.join(results_dir, "test_" + model_name + ".pt"),
    )
    logger.info(f"eval_metric: {eval_metric(test_preds, test_tgts)}")


if __name__ == "__main__":
    app.run(main)
