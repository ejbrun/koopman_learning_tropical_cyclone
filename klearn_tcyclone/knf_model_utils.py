"""Utils for Koopman model."""

import copy

from klearn_tcyclone.KNF.modules.models import Koopman
import torch

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
