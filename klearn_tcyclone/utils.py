import numpy as np
from climada.hazard import TCTracks

def check_time_steps_TCTracks(tc_tracks: TCTracks, time_step_h: int) -> bool:
    tc_track_data = tc_tracks.data
    is_close = []
    for tc_track in tc_track_data:
        times = tc_track.time.data
        time_diffs_h = np.array(np.diff(times), dtype=np.float64)/1E9/3600
        is_close.append(np.allclose(time_diffs_h, time_step_h, atol=1E-21))
    return np.all(is_close)