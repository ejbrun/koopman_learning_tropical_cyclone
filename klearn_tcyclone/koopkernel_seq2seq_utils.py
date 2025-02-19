"""Utils for Koopman Kernel Seq2Seq architecture."""

from datetime import datetime

import torch
from matplotlib.axes._axes import Axes
from torch.utils.tensorboard import SummaryWriter

from klearn_tcyclone.knf_data_utils import TCTrackDataset
from klearn_tcyclone.koopkernel_seq2seq import (
    KoopKernelLoss,
    KoopmanKernelSeq2Seq,
    batch_tensor_context,
)
from klearn_tcyclone.plot_utils import plot_TCTrackDataset_item_2D

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(
    model: KoopmanKernelSeq2Seq,
    optimizer,
    loss_fun,
    scheduler,
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

    print(range(tensor_context_inps.shape[1]))
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

        # print(loss.item())

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

    scheduler.step()

    return last_loss


def train_KKSeq2Seq(
    model: KoopmanKernelSeq2Seq,
    tensor_context,
    num_epochs: int,
    batch_size: int,
    input_length: int,
    output_length: int,
    decay_rate: float,
    learning_rate: float,
):
    tensor_context_inps, tensor_context_tgts = batch_tensor_context(
        tensor_context,
        batch_size=batch_size,
        input_length=input_length,
        output_length=output_length,
    )
    if output_length == 1:
        assert torch.all(tensor_context_inps[:, :, 1:] == tensor_context_tgts[:, :, :-1])
    else:
        for idx in range(output_length):
            assert torch.all(tensor_context_inps[:,:,idx+1:] == tensor_context_tgts[:,:,:-idx-1,idx])
    

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_koopkernel = KoopKernelLoss(model.nystrom_data_Y, model._kernel)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tb_writer = SummaryWriter("runs/fashion_trainer_{}".format(timestamp))

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=1, gamma=decay_rate
    )  # stepwise learning rate decay

    epoch_index = 1
    train_loss = []
    for epoch_index, epoch in enumerate(range(num_epochs)):
        train_loss.append(
            train_one_epoch(
                model,
                optimizer,
                loss_koopkernel,
                scheduler,
                epoch_index,
                tb_writer,
                tensor_context_inps,
                tensor_context_tgts,
            )
        )

    return model, train_loss


# def train_KKSeq2Seq(num_epochs, context_mode) -> tuple[KoopmanKernelSeq2Seq, list]:

#     if flag_params["train_output_length"] != 1:
#         raise Exception("This function still requires output_length = 1.")

#     rbf = RBFKernel(length_scale=1E-1)

#     #TODO some parameters don't play a role in model definition -> remove them
#     koopkernelmodel = KoopmanKernelSeq2Seq(
#         kernel=rbf,
#         input_dim = num_feats,
#         input_length = flag_params["input_length"],
#         output_length = flag_params["train_output_length"],
#         output_dim = 1,
#         num_steps = 1,
#         num_nys_centers = flag_params["koopman_kernel_num_centers"],
#         rng_seed = 42,
#         context_mode=context_mode,
#     )

#     koopkernelmodel._initialize_nystrom_data(tensor_context_train_standardized)

#     optimizer = torch.optim.Adam(koopkernelmodel.parameters(), lr=learning_rate)
#     loss_koopkernel = KoopKernelLoss(koopkernelmodel.nystrom_data_Y, koopkernelmodel._kernel)

#     timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#     tb_writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))

#     scheduler = torch.optim.lr_scheduler.StepLR(
#         optimizer, step_size=1, gamma=flag_params["decay_rate"]
#     )  # stepwise learning rate decay


#     epoch_index = 1
#     train_loss = []
#     for epoch_index, epoch in enumerate(range(num_epochs)):
#         train_loss.append(train_one_epoch(koopkernelmodel, optimizer, loss_koopkernel, scheduler, epoch_index, tb_writer, tensor_context_inps, tensor_context_tgts))


#     return koopkernelmodel, train_loss


# def predict_koopman_seq2seq_model(
#     time_series: torch.Tensor,
#     prediction_steps: int,
#     model: KoopmanKernelSeq2Seq,
# ):
#     assert len(time_series.shape) == 2
#     # tgts_dummy only needed to set the number of predicted time steps by model.
#     """
#     Note: The model performs a number of single_forward steps, each predicting
#     `num_steps` future time steps, such that the total number of predicted time steps is
#     larger than `tgts.shape[1]`, i.e. `prediction_steps` or the `output_length` of the
#     dataset. In the end only the first `output_length` predicted data points are given
#     as output, the remaining (further) steps are discarded.
#     """
#     if model.context_mode == "no_context":
#         time_series_unsqueeze = time_series.unsqueeze(0).to(device)
#     elif model.context_mode == "full_context":
#         time_series_unsqueeze = time_series[-model.input_length:].unsqueeze(0).to(device)
#     if model.context_mode == "last_context":
#         time_series_unsqueeze = time_series[-model.input_length:].unsqueeze(0).to(device)
#     # outs_list = []
#     prediction = time_series_unsqueeze
#     # outs_list.append(prediction)
#     for _ in range(prediction_steps):
#         new_prediction = model(prediction)
#         if model.context_mode == "last_context":
#             new_prediction = new_prediction.unsqueeze(1).to(device)
#             prediction = torch.cat([prediction[:,1:], new_prediction], dim=1)
#         else:
#             prediction = torch.cat([prediction[:,1:], new_prediction[:,-1:]], dim=1)
#         # outs_list.append(prediction)

#     res = prediction[0,-prediction_steps:]
#     return res


def predict_koopman_seq2seq_model(
    time_series: torch.Tensor,
    prediction_steps: int,
    model: KoopmanKernelSeq2Seq,
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
