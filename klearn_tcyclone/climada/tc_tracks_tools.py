"""Tools for tc_tracks.py."""

BASINS_ALL = ["EP", "NA", "NI", "SI", "SP", "WP", "SA"]
BASINS_SELECTION = ["EP", "NA", "SI", "SP", "WP"]

BASIN_SHIFT_SWITCHER = {
    BASINS_ALL[0]: 180,
    BASINS_ALL[1]: 0,
    BASINS_ALL[2]: 0,
    BASINS_ALL[3]: 0,
    BASINS_ALL[4]: 180,
    BASINS_ALL[5]: 180,
    BASINS_ALL[6]: 0,
}