"""Utils for Climada package."""

from klearn_tcyclone.climada.tc_tracks import TCTracks
from klearn_tcyclone.utils import check_time_steps_TCTracks


def get_TCTrack_dict(
    basins: list[str], time_step_h: float, year_range: tuple[int, int] = (2000, 2021)
):
    tc_tracks_dict = {}
    for basin in basins:
        tc_tracks = TCTracks.from_ibtracs_netcdf(
            provider="official", year_range=(2000, 2021), basin=basin
        )
        print("Number of tracks:", tc_tracks.size)
        tc_tracks.equal_timestep(time_step_h=time_step_h)
        assert check_time_steps_TCTracks(tc_tracks, time_step_h)
        tc_tracks_dict[basin] = tc_tracks
    return tc_tracks_dict
