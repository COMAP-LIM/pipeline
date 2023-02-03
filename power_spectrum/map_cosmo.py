import numpy as np

import re
import os
import sys
from typing import Optional
from dataclasses import dataclass, field

from astropy import units as u
from astropy import constants
from astropy import wcs
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM

current = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current)

sys.path.append(os.path.join(parent_directory, "tod2comap"))
from COmap import COmap


@dataclass
class MapCosmo:
    """Cosmological map class which assigns cosmological units to map."""

    def __init__(
        self,
        mappath: str,
        feed: Optional[int] = None,
        jk: Optional[int] = None,
        split: Optional[str] = None,
    ):
        """Init method to set up needed class attributes.

        Args:
            mappath (str): Path to COmap file from which to set up cosmological map.
            feed (Optional[int], optional): Index of feed to use. Defaults to None results in feed-coadded map.
            jk (Optional[int], optional): DEPRICATED. Jackknife index. Defaults to None.
            split (Optional[str], optional): Key of split map to use.
            Defaults to None results in feed or feed-coadded map being used.

        Raises:
            ValueError: If no feed is specified and split key is provided.
            ValueError: If split key does not contain 'map'.
        """

        self.feed = feed
        self.interpret_mapname(mappath)

        mapdata = COmap(mappath)

        mapdata.read_map()

        self.x = mapdata["ra_centers"]
        self.y = mapdata["dec_centers"]

        x_edges = mapdata["ra_edges"]
        y_edges = mapdata["dec_edges"]

        if split is not None:
            if feed is None:
                raise ValueError(
                    "Can only make cosmological map if both split and feed are specified."
                )

            if "map" not in split:
                raise ValueError(
                    "Make sure to provide the split 'map' key, not the nhit or sigma_wn key."
                )
            self.map = np.array(mapdata[split][feed])
            sigma_key = re.sub(
                r"map",
                "sigma_wn",
                split,
            )
            self.rms = np.array(mapdata[sigma_key][feed])

        elif feed is not None:
            self.map = np.array(mapdata["map"][feed])
            self.rms = np.array(mapdata["sigma_wn"][feed])

        else:
            self.map = np.array(mapdata["map_coadd"][:])
            self.rms = np.array(mapdata["sigma_wn_coadd"][:])

        cosmo = FlatLambdaCDM(Om0=0.286, Ob0=0.047, H0=0.7 * 100 * u.km / u.s / u.Mpc)

        NSIDEBAND, NCHANNEL, NDEC, NRA = self.map.shape

        Z_MID = 2.9  # Middle of the redshift range of map
        NU_REST = 115.27 * u.GHz  # Rest frequency of CO J(1->0)

        N_FREQ = NSIDEBAND * NCHANNEL
        nu = np.linspace(26, 34, N_FREQ) * u.GHz  # NOTE: GENERALIZE LATER

        dnu = nu[1] - nu[0]

        dredshift = (1 + Z_MID) ** 2 * dnu / NU_REST

        angle2Mpc = cosmo.kpc_comoving_per_arcmin(Z_MID).to(u.Mpc / u.arcmin)

        NRA, NDEC = self.x.size, self.y.size

        # Centering RA and Dec relative to field centers before converting to cosmological distnace
        x = (
            (
                (self.x - x_edges[NDEC // 2])
                * np.abs(
                    np.cos(np.radians(y_edges[NDEC // 2]))
                )  # Converting to physical degrees
            )
            * u.deg
            * angle2Mpc
        ).to(u.Mpc)
        y = ((self.y - y_edges[NDEC // 2]) * u.deg * angle2Mpc).to(u.Mpc)

        # Cosmological distance corresponding to redshift width in middle of box
        dz = cosmo.comoving_distance(Z_MID + dredshift / 2) - cosmo.comoving_distance(
            Z_MID - dredshift / 2
        )

        # Generating equispaced cosmological grid from redshifts, relative to first frequency
        z = np.arange(0, N_FREQ) * dz

        # Spacial resolutions
        dx = np.abs(x[1] - x[0])
        dy = np.abs(y[1] - y[0])
        dz = np.abs(z[1] - z[0])

        # Grid sizes
        nx = len(x)
        ny = len(y)
        nz = len(z)

        # Cosmologial volumes
        voxel_volume = dx * dy * dz

        # Making them class attributes
        self.x = x.value
        self.y = y.value
        self.z = z.value

        self.dx = dx.value
        self.dy = dy.value
        self.dz = dz.value

        self.nx = nx
        self.ny = ny
        self.nz = nz

        self.voxel_volume = voxel_volume.value

        self.map = self.map.transpose(3, 2, 0, 1) * u.K
        self.rms = self.rms.transpose(3, 2, 0, 1) * u.K

        self.map = self.map.reshape((NDEC, NRA, N_FREQ)).to(u.uK)
        self.rms = self.rms.reshape((NDEC, NRA, N_FREQ)).to(u.uK)

        self.mask = np.isfinite(
            self.rms
        )  # rms should be either np.inf or np.nan where no hits or masked

        self.w = np.zeros(self.rms.shape) / u.uK**2
        self.w[self.mask] = 1 / self.rms[self.mask] ** 2

        self.map[~self.mask] = 0.0
        self.rms[~self.mask] = 0.0

        self.map = self.map.value
        self.rms = self.rms.value
        self.w = self.w.value

    def interpret_mapname(self, mappath):
        self.mappath = mappath
        mapname = mappath.rpartition("/")[-1]
        mapname = "".join(mapname.rpartition(".")[:-2])

        parts = mapname.rpartition("_")
        try:
            self.field = parts[0]
            self.map_string = "".join(parts[2:])
            if not self.field == "":
                self.save_string = "_" + self.field + "_" + self.map_string
            else:
                self.save_string = "_" + self.map_string
        except:
            print("Unable to find field or map_string")
            self.field = ""
            self.map_string = ""
            self.save_string = ""

        if self.feed is not None:
            self.save_string = self.save_string + "_%02i" % self.feed
