
from klearn_tcyclone.utils import check_time_steps_TCTracks
from climada.hazard import TCTracks
import xarray as xr
import datetime as dt
import numpy as np


def test_check_time_steps_TCTracks():
    tc_tracks = TCTracks.from_ibtracs_netcdf(provider='usa', year_range=(1993, 1994), basin='EP', correct_pres=False)
    check = check_time_steps_TCTracks(tc_tracks, time_step_h=3)
    assert check == False
    tc_tracks.equal_timestep(3)
    check = check_time_steps_TCTracks(tc_tracks, time_step_h=3)
    assert check == True


def test_check_time_steps_TCTracks_2():

    hours = range(0,33, 3)
    times = np.array([dt.datetime(2000, 1, 1) + dt.timedelta(hours=x) for x in hours], dtype=np.datetime64)
    lat = np.random.normal(0,1,len(times))
    lon = np.random.normal(0,1,len(times))
    data = np.random.normal(0,1,len(times))

    test_data = xr.DataArray(
        dims=["time"],
        coords=dict(
            time=times,
            lat = ("time", lat),
            lon = ("time", lon),
        ),
        data=data,
    )

    test_tc_track = TCTracks(data=[test_data, test_data])
    check = check_time_steps_TCTracks(test_tc_track, time_step_h=3)
    assert check == True
    check = check_time_steps_TCTracks(test_tc_track, time_step_h=2)
    assert check == False
