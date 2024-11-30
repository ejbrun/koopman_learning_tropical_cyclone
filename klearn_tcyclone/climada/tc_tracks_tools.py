"""Tools for tc_tracks.py."""

BASINS = ["EP", "NA", "NI", "SI", "SP", "WP", "SA"]

BASIN_SHIFT_SWITCHER = {
    BASINS[0]: 180,
    BASINS[1]: 0,
    BASINS[2]: 0,
    BASINS[3]: 0,
    BASINS[4]: 180,
    BASINS[5]: 180,
    BASINS[6]: 0,
}