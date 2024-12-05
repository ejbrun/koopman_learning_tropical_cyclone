"""Generate data for Koopman spectral analysis."""

import os
import pickle
import numpy as np

from klearn_tcyclone.climada.tc_tracks_tools import BASINS
from klearn_tcyclone.climada.utils import get_TCTrack_dict
from klearn_tcyclone.kooplearn.spectral_analysis import (
    time_lag_scaling,
)


time_step_h = 0.5
# time_step_h = 1.0
time_step_scaling = int(3 / time_step_h)
# BASINS = ["EP", "NA", "NI", "SI", "SP", "WP", "SA"]
basins = ["EP", "NA", "SI", "SP", "WP"]
# basins = BASINS[:-1]
# year_range = (2000, 2021)
year_range = (1980, 2021)
tc_tracks_dict = get_TCTrack_dict(basins, time_step_h, year_range)

reduced_rank = True

for num_centers in [800, 1600]:
    for tikhonov_reg in [1E-4, 1E-8]:
        model_config = {
            "length_scale": 10.0,
            "reduced_rank": reduced_rank,
            "rank": 50,
            "num_centers": num_centers,
            "tikhonov_reg": tikhonov_reg,
            "svd_solver": "arnoldi",
            "rng_seed": 42,
        }
        context_lengths = [2, 4, 8, 16]
        top_k = 20
        feature_list = [
            "lon",
            "lat",
            "max_sustained_wind",
            "radius_max_wind",
            "radius_oci",
            "central_pressure",
            "environmental_pressure",
        ]
        time_lags = [1, *list(np.array(range(1, 6, 1)) * time_step_scaling)]

        folder_name = "_".join(
            [
                "year_range",
                *map(str, year_range),
                "clengths",
                *map(str, context_lengths),
                "tlags",
                *map(str, time_lags),
            ]
        )
        save_path = os.path.join(
            "../../data/", "koopman_spectral_analysis/", "time_lag_scaling/", folder_name
        )
        os.makedirs(save_path, exist_ok=True)

        for context_length in context_lengths:
            print("Spectral analysis for context length:", context_length)
            if model_config["reduced_rank"]:
                reduced_rank_str = "redrank"
            else:
                reduced_rank_str = ""
            file_name = "_".join(
                [
                    f"cl{context_length}",
                    f"tsteph{time_step_h}"
                    f"nc{model_config['num_centers']}",
                    f"tkreg{model_config['tikhonov_reg']}",
                    reduced_rank_str,
                ]
            )

            evals = {}
            errors = {}
            time_scales = {}
            for basin in basins:
                print("Basin:", basin)
                ev, e, ts = time_lag_scaling(
                    tc_tracks_dict[basin],
                    basin=basin,
                    time_lags=time_lags,
                    context_length=context_length,
                    top_k=top_k,
                    model_config=model_config,
                    feature_list=feature_list,
                )
                evals[basin] = ev
                errors[basin] = e
                time_scales[basin] = ts

            print("Save data.")
            with open(os.path.join(save_path, "evals_" + file_name + ".pickle"), "wb") as file:
                pickle.dump(evals, file)
            with open(os.path.join(save_path, "errors_" + file_name + ".pickle"), "wb") as file:
                pickle.dump(errors, file)
            with open(
                os.path.join(save_path, "time_scales_" + file_name + ".pickle"), "wb"
            ) as file:
                pickle.dump(time_scales, file)
