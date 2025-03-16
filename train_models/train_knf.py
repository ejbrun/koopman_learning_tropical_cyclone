"""Comparing KNF and koopkernel sequence models."""

import logging
import os
import random
import time
from datetime import datetime
from itertools import product

import numpy as np
import torch
from absl import app
from klearn_tcyclone.models_utils import get_model_name
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils import data

from klearn_tcyclone.climada.tc_tracks import TCTracks
from klearn_tcyclone.climada.utils import get_TCTrack_dict
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
from klearn_tcyclone.koopkernel_seq2seq_utils import (
    train_koopkernel_seq2seq_model,
)
from klearn_tcyclone.training_utils.args import FLAGS
from klearn_tcyclone.training_utils.training_utils import (
    extend_by_default_flag_values,
    set_flags,
)
from klearn_tcyclone.knf_model_utils import train_KNF_model

def main():

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    early_stopping = False
    save_model = "best"


    time_lag = 1
    feature_list = [
        "lon",
        "lat",
        "max_sustained_wind",
        # "radius_max_wind",
        # "radius_oci",
        "central_pressure",
        "environmental_pressure",
    ]


    # Set training settings
    # training_settings_full = {
    #     "koopman_kernel_length_scale": [0.06, 0.08, 0.1, 0.12, 0.14],
    #     "koopman_kernel_num_centers": [2000],
    #     "context_mode": ["full_context", "last_context"],
    #     # "context_mode": ["no_context", "full_context", "last_context"],
    #     "mask_koopman_operator": [True, False],
    #     "mask_version": [1],
    #     # "mask_version": [0, 1],
    #     "use_nystroem_context_window": [False, True],
    #     "output_length": [1],
    # }
    # selection 1
    # training_settings = {
    #     "koopman_kernel_length_scale": [0.06, 0.08, 0.1, 0.12, 0.14],
    #     "koopman_kernel_num_centers": [2000],
    #     "context_mode": ["full_context"],
    #     # "context_mode": ["no_context", "full_context", "last_context"],
    #     "mask_koopman_operator": [True, False],
    #     "mask_version": [1],
    #     # "mask_version": [0, 1],
    #     "use_nystroem_context_window": [False, True],
    #     "output_length": [1],
    # }

    # koopman_kernel_length_scale_arr = [0.16, 0.18, 0.2, 0.22, 0.24]
    # koopman_kernel_length_scale_arr = [0.06, 0.08, 0.1, 0.12, 0.14]
    # koopman_kernel_length_scale_arr = [1e-2, 1e-1, 1e0, 1e1]
    # koopman_kernel_num_centers_arr = [100, 200]
    # koopman_kernel_num_centers_arr = [1000]
    # koopman_kernel_num_centers_arr = [1000, 2000]
    tc_tracks_time_step = 3.0


    flag_params = {
        # "year_range": [1980, 2021],
        # "year_range": [2018, 2021],
        "year_range": [2000, 2021],
    }
    flag_params = extend_by_default_flag_values(flag_params)
    # FIXME add context_mode, mask_koopman_operator, mask_version,
    #   use_nystroem_context_window to default flat_params parameters

    flag_params["batch_size"] = 32
    # flag_params["num_epochs"] = 10
    flag_params["num_epochs"] = 100

    flag_params["num_steps"] = 1
    #FIXME Remove num_steps, not accessed for the KooplearnSequencer.
    flag_params["time_step_h"] = tc_tracks_time_step
    flag_params["basin"] = "NA"





    random.seed(flag_params["seed"])  # python random generator
    np.random.seed(flag_params["seed"])  # numpy random generator


    # Datasets
    tc_tracks_dict = get_TCTrack_dict(
        basins=[flag_params["basin"]],
        time_step_h=flag_params["time_step_h"],
        year_range=flag_params["year_range"],
    )
    tc_tracks = tc_tracks_dict[flag_params["basin"]]


    # Logging and define save paths
    current_file_dir_path = os.path.dirname(os.path.abspath(__file__))
    # current_file_dir_path = os.getcwd()


    model_str = "KNF"
    print()
    print("===================================")
    print(f"Train {model_str}.")
    print("===================================")
    print()

    date_time = datetime.fromtimestamp(time.time())
    str_date_time = date_time.strftime("%Y-%m-%d-%H-%M-%S")
    flag_params["model"] = model_str

    results_dir = os.path.join(
        current_file_dir_path,
        "training_results",
        "comparison_knf_koopkernel_seq",
        "{}_yrange{}_basin{}_tstep{}".format(
            flag_params["dataset"],
            "".join(map(str, flag_params["year_range"])),
            flag_params["basin"],
            flag_params["time_step_h"],
        ),
        flag_params["model"],
        str_date_time,
    )
    logs_dir = os.path.join(
        current_file_dir_path,
        "logs",
        "comparison_knf_koopkernel_seq",
        "{}_yrange{}_basin{}_tstep{}".format(
            flag_params["dataset"],
            "".join(map(str, flag_params["year_range"])),
            flag_params["basin"],
            flag_params["time_step_h"],
        ),
        flag_params["model"],
    )
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    log_file_name = os.path.join(logs_dir, flag_params["model"] + ".log")

    logFormatter = logging.Formatter(
        "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
        # style="{",
        # datefmt="%Y-%m-%d %H:%M:%S",
    )
    fileHandler = logging.FileHandler(log_file_name)
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)
    # consoleHandler = logging.StreamHandler()
    # consoleHandler.setFormatter(logFormatter)
    # logger.addHandler(consoleHandler)

    # logging.basicConfig(
    #     filename=os.path.join(logs_dir, flag_params["model"] + ".log"),
    #     encoding="utf-8",
    #     filemode="a",
    #     format="{asctime} - {name} - {filename}:{lineno} - {levelname} - {message}",
    #     style="{",
    #     datefmt="%Y-%m-%d %H:%M:%S",
    #     force=True,
    # )
    # logger = logging.getLogger(flag_params["model"] + "_logger")

    logger.info(flag_params)


    output_length = 1

    flag_params["train_output_length"] = output_length
    flag_params["test_output_length"] = flag_params["train_output_length"]
    flag_params["input_length"] = 12
    flag_params["input_dim"] = 4
    flag_params["context_length"] = (
        flag_params["input_length"] + flag_params["train_output_length"]
    )
    assert (
        flag_params["context_length"]
        == flag_params["input_length"] + flag_params["train_output_length"]
    )
    if flag_params["input_length"] % flag_params["input_dim"] != 0:
        raise Exception("input_length must be divisible by input_dim")


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device {device}")


    tc_tracks_train, tc_tracks_test = train_test_split(tc_tracks.data, test_size=0.1)

    _, _ = train_KNF_model(
        tc_tracks_train=tc_tracks_train,
        tc_tracks_test=tc_tracks_test,
        feature_list=feature_list,
        flag_params=flag_params,
        log_file_handler=fileHandler,
        results_dir=results_dir,
    )

    # encoder_hidden_dim = flag_params["hidden_dim"]
    # decoder_hidden_dim = flag_params["hidden_dim"]
    # encoder_num_layers = flag_params["num_layers"]
    # decoder_num_layers = flag_params["num_layers"]
    # output_dim = flag_params["input_dim"]
    # num_feats = len(feature_list)
    # learning_rate = flag_params["learning_rate"]

    # scaler = LinearScaler()
    # eval_metric = RMSE_TCTracks

    # train_set = TCTrackDataset(
    #     input_length=flag_params["input_length"],
    #     output_length=flag_params["train_output_length"],
    #     tc_tracks=tc_tracks_train,
    #     feature_list=feature_list,
    #     mode="train",
    #     jumps=flag_params["jumps"],
    #     scaler=scaler,
    #     fit=True,
    # )
    # valid_set = TCTrackDataset(
    #     input_length=flag_params["input_length"],
    #     output_length=flag_params["train_output_length"],
    #     tc_tracks=tc_tracks_train,
    #     feature_list=feature_list,
    #     mode="valid",
    #     jumps=flag_params["jumps"],
    #     scaler=scaler,
    #     fit=False,
    # )
    # test_set = TCTrackDataset(
    #     input_length=flag_params["input_length"],
    #     output_length=flag_params["test_output_length"],
    #     tc_tracks=tc_tracks_test,
    #     feature_list=feature_list,
    #     mode="test",
    #     # jumps=flag_params["jumps"], # jumps not used in test mode
    #     scaler=scaler,
    #     fit=False,
    # )
    # train_loader = data.DataLoader(
    #     train_set, batch_size=flag_params["batch_size"], shuffle=True, num_workers=1
    # )
    # valid_loader = data.DataLoader(
    #     valid_set, batch_size=flag_params["batch_size"], shuffle=True, num_workers=1
    # )
    # test_loader = data.DataLoader(
    #     test_set, batch_size=flag_params["batch_size"], shuffle=False, num_workers=1
    # )



    # model_name = get_model_name(flag_params)
    # results_file_name = os.path.join(results_dir, model_name)

    # last_epoch = 0
    # model = Koopman(
    #     # number of steps of historical observations encoded at every step
    #     input_dim=flag_params["input_dim"],
    #     # input length of ts
    #     input_length=flag_params["input_length"],
    #     # number of output features
    #     output_dim=output_dim,
    #     # number of prediction steps every forward pass
    #     num_steps=flag_params["num_steps"],
    #     # hidden dimension of encoder
    #     encoder_hidden_dim=encoder_hidden_dim,
    #     # hidden dimension of decoder
    #     decoder_hidden_dim=decoder_hidden_dim,
    #     # number of layers in the encoder
    #     encoder_num_layers=encoder_num_layers,
    #     # number of layers in the decoder
    #     decoder_num_layers=decoder_num_layers,
    #     # number of feature
    #     num_feats=num_feats,
    #     # dimension of finite koopman space
    #     latent_dim=flag_params["latent_dim"],
    #     # whether to learn a global operator shared across all time series
    #     add_global_operator=flag_params["add_global_operator"],
    #     # whether to use a feedback module
    #     add_control=flag_params["add_control"],
    #     # hidden dim in the control module
    #     control_hidden_dim=flag_params["control_hidden_dim"],
    #     # number of layers in the control module
    #     use_revin=flag_params["use_revin"],
    #     # whether to use reversible normalization
    #     control_num_layers=flag_params["control_num_layers"],
    #     # whether to use instance normalization on hidden states
    #     use_instancenorm=flag_params["use_instancenorm"],
    #     # Regularize rank.
    #     regularize_rank=flag_params["regularize_rank"],
    #     # number of pairs of sine and cosine measurement functions
    #     num_sins=flag_params["num_sins"],
    #     # the highest order of polynomial functions
    #     num_poly=flag_params["num_poly"],
    #     # number of exponential functions
    #     num_exp=flag_params["num_exp"],
    #     # Number of the head the transformer encoder
    #     num_heads=flag_params["num_heads"],
    #     # hidden dimension of tranformer encoder
    #     transformer_dim=flag_params["transformer_dim"],
    #     # number of layers in the transformer encoder
    #     transformer_num_layers=flag_params["transformer_num_layers"],
    #     # dropout rate of MLP modules
    #     dropout_rate=flag_params["dropout_rate"],
    #     # global_local_combination
    #     global_local_combination=flag_params["global_local_combination"],
    # ).to(device)

    # logger.info(
    #     f"number of params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}",
    #     # sum(p.numel() for p in model.parameters() if p.requires_grad),
    # )

    # logger.info("define optimizer")
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # scheduler = torch.optim.lr_scheduler.StepLR(
    #     optimizer, step_size=1, gamma=flag_params["decay_rate"]
    # )  # stepwise learning rate decay
    # loss_fun = nn.MSELoss()

    # all_train_rmses, all_eval_rmses = [], []
    # best_eval_rmse = 1e6

    # logger.info("start training")
    # training_time_start = time.time()
    # for epoch in range(last_epoch, flag_params["num_epochs"]):
    #     start_time = time.time()

    #     train_rmse = train_epoch_koopman(
    #         train_loader,
    #         model,
    #         loss_fun,
    #         optimizer,
    #         regularize_rank=flag_params["regularize_rank"],
    #     )
    #     eval_rmse, _, _ = eval_epoch_koopman(
    #         valid_loader,
    #         model,
    #         loss_fun,
    #         regularize_rank=flag_params["regularize_rank"],
    #     )

    #     print("eval comparison", eval_rmse, best_eval_rmse)
    #     if save_model == "all":
    #         torch.save(
    #             [model, epoch, optimizer.param_groups[0]["lr"]],
    #             results_file_name + f"_epoch{epoch}" + ".pth",
    #         )

    #     if eval_rmse < best_eval_rmse:
    #         best_eval_rmse = eval_rmse
    #         best_model = model
    #         if save_model == "best":
    #             torch.save(
    #                 [best_model, epoch, optimizer.param_groups[0]["lr"]],
    #                 results_file_name + "_best.pth",
    #             )

    #     all_train_rmses.append(train_rmse)
    #     all_eval_rmses.append(eval_rmse)

    #     if np.isnan(train_rmse) or np.isnan(eval_rmse):
    #         raise ValueError("The model generate NaN values")

    #     # Save test scores.
    #     _, test_preds, test_tgts = eval_epoch_koopman(test_loader, best_model, loss_fun)
    #     torch.save(
    #         {
    #             "test_preds": test_preds,
    #             "test_tgts": test_tgts,
    #             "eval_score": eval_metric(test_preds, test_tgts),
    #         },
    #         os.path.join(results_dir, f"ep{epoch}_test" + model_name + ".pt"),
    #     )

    #     # train the model at least 60 epochs and do early stopping
    #     if early_stopping:
    #         if epoch > flag_params["min_epochs"] and np.mean(
    #             all_eval_rmses[-10:]
    #         ) > np.mean(all_eval_rmses[-20:-10]):
    #             break

    #     epoch_time = time.time() - start_time
    #     scheduler.step()
    #     logger.info(
    #         "Epoch {} | T: {:0.2f} | Train RMSE: {:0.3f} | Valid RMSE: {:0.3f}".format(  # noqa: UP032
    #             epoch + 1, epoch_time / 60, train_rmse, eval_rmse
    #         )
    #     )

    # training_runtime = time.time() - training_time_start

    # logger.info("Evaluate test metric.")
    # _, test_preds, test_tgts = eval_epoch_koopman(test_loader, best_model, loss_fun)
    # torch.save(
    #     {
    #         "test_preds": test_preds,
    #         "test_tgts": test_tgts,
    #         "eval_score": eval_metric(test_preds, test_tgts),
    #         "train_rmses": all_train_rmses,
    #         "eval_rmses": all_eval_rmses,
    #         "training_runtime": training_runtime,
    #     },
    #     os.path.join(results_dir, "test_" + model_name + ".pt"),
    # )
    # logger.info(f"eval_metric: {eval_metric(test_preds, test_tgts)}")



if __name__ == "__main__":
    main()
    # app.run(main)

