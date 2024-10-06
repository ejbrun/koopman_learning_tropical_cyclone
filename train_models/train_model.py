"""Training of models."""

import os
import random
import time

import numpy as np
import torch
from absl import app
from args import FLAGS
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
from klearn_tcyclone.training_utils import set_flags


def main(argv):
    print(FLAGS.seed)
    print(FLAGS.model)
    print(FLAGS.dataset)
    print(FLAGS.global_local_combination)
    print(FLAGS.learning_rate)
    print(FLAGS.decay_rate)
    print(FLAGS.batch_size)
    print(FLAGS.num_epochs)
    print(FLAGS.min_epochs)

    random.seed(FLAGS.seed)  # python random generator
    np.random.seed(FLAGS.seed)  # numpy random generator

    torch.manual_seed(FLAGS.seed)
    torch.cuda.manual_seed_all(FLAGS.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



    # parameters from flag
    flag_params = set_flags(FLAGS=FLAGS)

    (
        seed,
        model_str,
        dataset,
        year_range,
        global_local_combination,
        learning_rate,
        decay_rate,
        batch_size,
        num_epochs,
        min_epochs,
    ) = flag_params.values()

    print(flag_params)

    feature_list = ["lat", "lon"]
    # feature_list = ["lat", "lon", "central_pressure"]

    # fixed parameters
    freq = None
    num_feats = len(feature_list)
    train_output_length = 15
    input_dim = 3
    input_length = 21
    
    latent_dim = 32
    num_layers = 4
    control_num_layers = 3
    control_hidden_dim = 64
    transformer_dim = 64
    transformer_num_layers = 3
    test_output_length = train_output_length
    # num_steps can be chosen differently to train_output_length.
    num_steps = 15
    jumps = 2
    
    hidden_dim = 128
    control_hidden_dim = 64
    # hidden_dim = num_feats
    # control_hidden_dim = hidden_dim

    encoder_hidden_dim = hidden_dim
    decoder_hidden_dim = hidden_dim
    encoder_num_layers = num_layers
    decoder_num_layers = num_layers
    output_dim = input_dim
    use_revin = True
    use_instancenorm = False
    # use_instancenorm = True
    regularize_rank = False
    add_global_operator = True
    add_control = True
    num_sins = -1
    num_poly = -1
    num_exp = -1
    num_heads = 1
    dropout_rate = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device", device)

    scaler = LinearScaler()
    eval_metric = RMSE_TCTracks

    # Datasets
    tc_tracks = TCTracks.from_ibtracs_netcdf(
        provider="usa", year_range=year_range, basin="NA", correct_pres=False
    )

    tc_tracks_train, tc_tracks_test = train_test_split(tc_tracks.data, test_size=0.1)
    train_set = TCTrackDataset(
        input_length=input_length,
        output_length=train_output_length,
        tc_tracks=tc_tracks_train,
        feature_list=feature_list,
        mode="train",
        jumps=jumps,
        scaler=scaler,
        fit=True,
    )
    valid_set = TCTrackDataset(
        input_length=input_length,
        output_length=train_output_length,
        tc_tracks=tc_tracks_train,
        feature_list=feature_list,
        mode="valid",
        jumps=jumps,
        scaler=scaler,
        fit=False,
    )
    test_set = TCTrackDataset(
        input_length=input_length,
        output_length=test_output_length,
        tc_tracks=tc_tracks_test,
        feature_list=feature_list,
        mode="test",
        # jumps=jumps, # jumps not used in test mode
        scaler=scaler,
        fit=False,
    )
    train_loader = data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=1
    )
    valid_loader = data.DataLoader(
        valid_set, batch_size=batch_size, shuffle=True, num_workers=1
    )
    test_loader = data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=1
    )

    if len(train_loader) == 0:
        raise Exception(
            "There are likely too few data points in the test set. Try to increase year_range."
        )

    # TODO Full model name is too long for torch.save().
    model_name = (
        "Koopman_"
        + str(dataset)
        + f"_model{model_str}_glc{global_local_combination}"
        + "_seed{}_jumps{}_freq{}_poly{}_sin{}_exp{}_bz{}_lr{}_decay{}_dim{}_inp{}_pred{}_num{}".format(  # noqa: E501, UP032
            seed,
            jumps,
            freq,
            num_poly,
            num_sins,
            num_exp,
            batch_size,
            learning_rate,
            decay_rate,
            input_dim,
            input_length,
            train_output_length,
            num_steps,
        )
        # + "_seed{}_jumps{}_freq{}_poly{}_sin{}_exp{}_bz{}_lr{}_decay{}_dim{}_inp{}_pred{}_num{}_enchid{}_dechid{}_trm{}_conhid{}_enclys{}_declys{}_trmlys{}_conlys{}_latdim{}_RevIN{}_insnorm{}_regrank{}_globalK{}_contK{}".format(  # noqa: E501, UP032
        #     seed,
        #     jumps,
        #     freq,
        #     num_poly,
        #     num_sins,
        #     num_exp,
        #     batch_size,
        #     learning_rate,
        #     decay_rate,
        #     input_dim,
        #     input_length,
        #     train_output_length,
        #     num_steps,
        #     encoder_hidden_dim,
        #     decoder_hidden_dim,
        #     transformer_dim,
        #     control_hidden_dim,
        #     encoder_num_layers,
        #     decoder_num_layers,
        #     transformer_num_layers,
        #     control_num_layers,
        #     latent_dim,
        #     use_revin,
        #     use_instancenorm,
        #     regularize_rank,
        #     add_global_operator,
        #     add_control,
        # )
    )

    print(model_name)

    file_dir_path = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(
        file_dir_path,
        "training_results",
        dataset,
    )

    file_name_model = os.path.join(results_dir, model_name + ".pth")
    if os.path.exists(file_name_model):
        model, last_epoch, learning_rate = torch.load(file_name_model)
        print("Resume Training")
        print("last_epoch:", last_epoch, "learning_rate:", learning_rate)
    else:
        last_epoch = 0
        if not os.path.exists(results_dir):
            # os.mkdir(results_dir)
            os.makedirs(results_dir, exist_ok=True)
        model = Koopman(
            # number of steps of historical observations encoded at every step
            input_dim=input_dim,
            # input length of ts
            input_length=input_length,
            # number of output features
            output_dim=output_dim,
            # number of prediction steps every forward pass
            num_steps=num_steps,
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
            latent_dim=latent_dim,
            # whether to learn a global operator shared across all time series
            add_global_operator=add_global_operator,
            # whether to use a feedback module
            add_control=add_control,
            # hidden dim in the control module
            control_hidden_dim=control_hidden_dim,
            # number of layers in the control module
            use_revin=use_revin,  # whether to use reversible normalization
            control_num_layers=control_num_layers,
            # whether to use instance normalization on hidden states
            use_instancenorm=use_instancenorm,
            # Regularize rank.
            regularize_rank=regularize_rank,
            # number of pairs of sine and cosine measurement functions
            num_sins=num_sins,
            # the highest order of polynomial functions
            num_poly=num_poly,
            # number of exponential functions
            num_exp=num_exp,
            # Number of the head the transformer encoder
            num_heads=num_heads,
            # hidden dimension of tranformer encoder
            transformer_dim=transformer_dim,
            # number of layers in the transformer encoder
            transformer_num_layers=transformer_num_layers,
            # dropout rate of MLP modules
            dropout_rate=dropout_rate,
            # global_local_combination
            global_local_combination=global_local_combination
        ).to(device)

        print("New model")
    print(
        "number of params:",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )

    print("define optimizer")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=1, gamma=decay_rate
    )  # stepwise learning rate decay
    loss_fun = nn.MSELoss()

    all_train_rmses, all_eval_rmses = [], []
    best_eval_rmse = 1e6

    print("start training")
    for epoch in range(last_epoch, num_epochs):
        start_time = time.time()

        train_rmse = train_epoch_koopman(
            train_loader, model, loss_fun, optimizer, regularize_rank=regularize_rank
        )
        eval_rmse, _, _ = eval_epoch_koopman(
            valid_loader, model, loss_fun, regularize_rank=regularize_rank
        )

        if eval_rmse < best_eval_rmse:
            best_eval_rmse = eval_rmse
            best_model = model
            # file_name_model2 = "test_path_model.pth"
            torch.save(
                [best_model, epoch, optimizer.param_groups[0]["lr"]],
                file_name_model,
            )

        all_train_rmses.append(train_rmse)
        all_eval_rmses.append(eval_rmse)

        if np.isnan(train_rmse) or np.isnan(eval_rmse):
            raise ValueError("The model generate NaN values")

        # Save test scores.
        _, test_preds, test_tgts = eval_epoch_koopman(test_loader, best_model, loss_fun)
        file_name_test_model = os.path.join(results_dir, f"ep{epoch}_test" + model_name + ".pt")
        torch.save(
            {
                "test_preds": test_preds,
                "test_tgts": test_tgts,
                "eval_score": eval_metric(test_preds, test_tgts),
            },
            file_name_test_model,
        )

        # train the model at least 60 epochs and do early stopping
        if epoch > min_epochs and np.mean(all_eval_rmses[-10:]) > np.mean(
            all_eval_rmses[-20:-10]
        ):
            break

        epoch_time = time.time() - start_time
        scheduler.step()
        print(
            "Epoch {} | T: {:0.2f} | Train RMSE: {:0.3f} | Valid RMSE: {:0.3f}".format(  # noqa: UP032
                epoch + 1, epoch_time / 60, train_rmse, eval_rmse
            )
        )

    print("Evaluate test metric.")
    _, test_preds, test_tgts = eval_epoch_koopman(test_loader, best_model, loss_fun)

    file_name_test_model = os.path.join(results_dir, "test_" + model_name + ".pt")
    torch.save(
        {
            "test_preds": test_preds,
            "test_tgts": test_tgts,
            "eval_score": eval_metric(test_preds, test_tgts),
        },
        file_name_test_model,
    )

    
    print(model_name)
    print(eval_metric(test_preds, test_tgts))


if __name__ == "__main__":
    app.run(main)
