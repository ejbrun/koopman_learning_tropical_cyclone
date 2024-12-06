"""Customize TCTracks methods."""

import cartopy.crs as ccrs
import climada.util.coordinates as u_coord
import climada.util.plot as u_plot
import matplotlib.pyplot as plt
import numpy as np
from climada.hazard import TCTracks as qTCTracks
from climada.hazard.tc_tracks import (
    CAT_COLORS,
    CAT_NAMES,
    LOGGER,
    SAFFIR_SIM_CAT,
)
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D


class TCTracks(qTCTracks):
    def plot(
        self,
        axis=None,
        figsize=(9, 13),
        legend=True,
        adapt_fontsize=True,
        linestyle=None,
        extent = None,
        loc_legend=None,
        ncols_legend = None,
        **kwargs,
    ):
        """Track over earth. Historical events are blue, probabilistic black.

        Parameters
        ----------
        axis : matplotlib.axes._subplots.AxesSubplot, optional
            axis to use
        figsize: (float, float), optional
            figure size for plt.subplots
            The default is (9, 13)
        legend : bool, optional
            whether to display a legend of Tropical Cyclone categories.
            Default: True.
        kwargs : optional
            arguments for LineCollection matplotlib, e.g. alpha=0.5
        adapt_fontsize : bool, optional
            If set to true, the size of the fonts will be adapted to the size of the figure.
            Otherwise the default matplotlib font size is used. Default is True.
        Returns
        -------
        axis : matplotlib.axes._subplots.AxesSubplot
        """
        if "lw" not in kwargs:
            kwargs["lw"] = 2
        if "transform" not in kwargs:
            kwargs["transform"] = ccrs.PlateCarree()
        if loc_legend is None:
            loc_legend = 0
        if ncols_legend is None:
            ncols_legend = 1

        if not self.size:
            LOGGER.info("No tracks to plot")
            return None

        if extent is None:
            extent = self.get_extent(deg_buffer=1)
        mid_lon = 0.5 * (extent[1] + extent[0])

        if not axis:
            proj = ccrs.PlateCarree(central_longitude=mid_lon)
            fig, axis, _ = u_plot.make_map(
                proj=proj, figsize=figsize, adapt_fontsize=adapt_fontsize
            )
        else:
            proj = axis.projection
        axis.set_extent(extent, crs=kwargs["transform"])
        u_plot.add_shapes(axis)

        cmap = ListedColormap(colors=CAT_COLORS)
        norm = BoundaryNorm([0] + SAFFIR_SIM_CAT, len(SAFFIR_SIM_CAT))
        for track in self.data:
            lonlat = np.stack([track.lon.values, track.lat.values], axis=-1)
            lonlat[:, 0] = u_coord.lon_normalize(lonlat[:, 0], center=mid_lon)
            segments = np.stack([lonlat[:-1], lonlat[1:]], axis=1)

            # Truncate segments which cross the antimeridian.
            # Note: Since we apply `lon_normalize` above and shift the central longitude of the
            # plot to `mid_lon`, this is not necessary (and will do nothing) in cases where all
            # tracks are located in a region around the antimeridian, like the Pacific ocean.
            # The only case where this is relevant: Crowded global data sets where `mid_lon`
            # falls back to 0, i.e. using the [-180, 180] range.
            mask = (segments[:, 0, 0] > 100) & (segments[:, 1, 0] < -100)
            segments[mask, 1, 0] = 180
            mask = (segments[:, 0, 0] < -100) & (segments[:, 1, 0] > 100)
            segments[mask, 1, 0] = -180

            if linestyle is None:
                linestyle = "solid"
            track_lc = LineCollection(
                segments,
                linestyle=linestyle if track.orig_event_flag else ":",
                cmap=cmap,
                norm=norm,
                **kwargs,
            )
            track_lc.set_array(track.max_sustained_wind.values)
            axis.add_collection(track_lc)

        if legend:
            leg_lines = [
                Line2D([0], [0], color=CAT_COLORS[i_col], lw=2)
                for i_col in range(len(SAFFIR_SIM_CAT))
            ]
            leg_names = [CAT_NAMES[i_col] for i_col in sorted(CAT_NAMES.keys())]
            if any(not tr.orig_event_flag for tr in self.data):
                leg_lines.append(Line2D([0], [0], color="grey", lw=2, ls="solid"))
                leg_lines.append(Line2D([0], [0], color="grey", lw=2, ls=":"))
                leg_names.append("Historical")
                leg_names.append("Synthetic")
            axis.legend(
                leg_lines,
                leg_names,
                loc=loc_legend,
                ncols=ncols_legend,
            )
        plt.tight_layout()
        return axis
