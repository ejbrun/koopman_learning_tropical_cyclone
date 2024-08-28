"""Utils for testing. This should be moved to a conftest.py file in future."""

from klearn_tcyclone.data_utils import context_dataset_from_TCTracks
from climada.hazard import TCTracks
from kooplearn.models import Kernel
from sklearn.gaussian_process.kernels import RBF
from klearn_tcyclone.data_utils import characteristic_length_scale_from_TCTracks


def provide_model():
    reduced_rank = True
    rank = 25
    tikhonov_reg = 1e-6
    length_scale = 50.0
    kernel = RBF(length_scale=length_scale)
    model = Kernel(
        kernel=kernel,
        reduced_rank=reduced_rank,
        svd_solver="randomized",
        tikhonov_reg=tikhonov_reg,
        rank=rank,
        rng_seed=42,
    )
    return model


def provide_TensorContextData():
    tc_tracks: TCTracks = TCTracks.from_ibtracs_netcdf(
        provider="usa", year_range=(2000, 2000), basin="EP", correct_pres=False
    )
    context_length = 42
    feature_list = ["lat", "lon"]
    # feature_list = ["lat", "lon", "max_sustained_wind", "central_pressure"]
    tensor_context = context_dataset_from_TCTracks(
        tc_tracks.data, feature_list=feature_list, context_length=context_length
    )
    return tensor_context
