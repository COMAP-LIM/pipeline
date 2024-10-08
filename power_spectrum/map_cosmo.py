import numpy as np

import re
import os
import sys
from typing import Optional
from dataclasses import dataclass, field

from astropy import units as u
import astropy.cosmology

import time

import argparse

current = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current)

sys.path.append(os.path.join(parent_directory, "tod2comap"))
from COmap import COmap


@dataclass
class MapCosmo:
    """Cosmological map class which assigns cosmological units to map."""

    def __init__(
        self,
        params: argparse.Namespace,
        mappath: str,
        cosmology: astropy.cosmology.flrw.FlatLambdaCDM,
        feed: Optional[int] = None,
        split: Optional[str] = None,

    ):
        """Init method to set up needed class attributes.

        Args:
            params (FlatLambdaCDM): Paramter file object argparse Namspace object containing run parameters.
            field (str): Field name string, should be one of ["co2", "co7", "co6"]
            feed (Optional[int], optional): Index of feed to use. Defaults to None results in feed-coadded map.
            split (Optional[str], optional): Key of split map to use.
            Defaults to None results in feed or feed-coadded map being used.

        Raises:
            ValueError: If no feed is specified and split key is provided.
            ValueError: If split key does not contain 'map'.
        """
        
        self.params = params
        if feed is not None:
            self.feed = feed + 1
        
        mapdata = COmap(mappath)

        key_list = [
            "ra_centers",
            "dec_centers",
            "ra_edges",
            "dec_edges",
        ]

        mapdata.read_and_append(key_list)

        self.x = mapdata["ra_centers"]
        self.y = mapdata["dec_centers"]

        x_edges = mapdata["ra_edges"]
        y_edges = mapdata["dec_edges"]

        if split is not None:
            if feed is None:
                raise ValueError(
                    "Can only make cosmological map if both split and feed are specified."
                )

            if params.psx_null_diffmap:
                split1, split2 = split

                if "map" not in split1 or "map" not in split2:
                    raise ValueError(
                    "Make sure to provide the split 'map' key, not the nhit or sigma_wn key."
                )
                
                if params.psx_mode == "saddlebag":
                    split1 = re.sub(r"map", "map_saddlebag", split1)
                    split2 = re.sub(r"map", "map_saddlebag", split2)
                
                sigma_key1 = re.sub(
                    r"map",
                    "sigma_wn",
                    split1,
                )
                sigma_key2 = re.sub(
                    r"map",
                    "sigma_wn",
                    split2,
                )

                mapdata.read_and_append([sigma_key1, sigma_key2])

                if params.psx_generate_white_noise_sim:
                    seed = params.psx_white_noise_sim_seed
                    
                    if params.psx_null_diffmap:
                        split1, _ = split
                        split_number = split1.split("map_")[-1][5:]
                        null_var = split1.split("map_")[-1][:4]
                        split_number = int(split_number[4:]) + params.primary_variables.index(null_var)
                    else:
                        split_number = int(split.split("map_")[-1][4:])

                    # Make unique seed for any split and feed combination
                    seed += int(self.feed + (split_number + 1)%2 * 1e6 * np.pi) 

                    np.random.seed(seed)
                    self.white_noise_seed = seed

                    self.map = (np.random.randn(*mapdata[sigma_key1][feed].shape) * mapdata[sigma_key1][feed][()] 
                               -  np.random.randn(*mapdata[sigma_key2][feed].shape) * mapdata[sigma_key2][feed][()]) / 2
                else:

                    mapdata.read_and_append([split1, split2])

                    self.map = (mapdata[split1][feed] - mapdata[split2][feed]) / 2
                # self.rms = np.sqrt(0.5 * (mapdata[sigma_key1][feed] ** 2 + mapdata[sigma_key2][feed] ** 2))
                self.rms = np.sqrt(mapdata[sigma_key1][feed] ** 2 + mapdata[sigma_key2][feed] ** 2) / 2

            else:
                if "map" not in split:
                    raise ValueError(
                        "Make sure to provide the split 'map' key, not the nhit or sigma_wn key."
                    )
                
                if params.psx_mode == "saddlebag":
                    split = re.sub(r"map", "map_saddlebag", split)

                mapdata.read_and_append([split])
                self.map = mapdata[split][feed]
                sigma_key = re.sub(
                    r"map",
                    "sigma_wn",
                    split,
                )

                mapdata.read_and_append([sigma_key])
                self.rms = mapdata[sigma_key][feed]

        elif feed is not None and params.psx_mode == "feed":
            mapdata.read_and_append(["map", "sigma_wn"])
            
            self.map = mapdata["map"][feed]
            self.rms = mapdata["sigma_wn"][feed]
        
        elif feed is not None and params.psx_mode == "saddlebag":
            mapdata.read_and_append(["map_saddlebag", "sigma_wn_saddlebag"])
            
            self.map = mapdata["map_saddlebag"][feed]
            self.rms = mapdata["sigma_wn_saddlebag"][feed]
            
        else:
            mapdata.read_and_append(["map_coadd", "sigma_wn_coadd"])

            self.map = mapdata["map_coadd"][:]
            self.rms = mapdata["sigma_wn_coadd"][:]
        
        # If wanted the maps can be substituted with pure white noise
        if params.psx_generate_white_noise_sim and not params.psx_null_diffmap:
            seed = params.psx_white_noise_sim_seed
            
            if params.psx_null_diffmap:
                split1, _ = split
                split_number = split1.split("map_")[-1][5:]
                split_number = int(split_number[4:])
            else:
                split_number = int(split.split("map_")[-1][4:])

            # Make unique seed for any split and feed combination
            seed += int(self.feed + (split_number + 1)%2 * 1e6 * np.pi) 

            np.random.seed(seed)
            self.white_noise_seed = seed

            self.map = np.random.randn(*self.rms.shape) * self.rms 
        
        NSIDEBAND, NCHANNEL, NDEC, NRA = self.map.shape

        Z_MID = params.phy_center_redshift  # Middle of the redshift range of map
        NU_REST = params.sim_nu_rest * u.GHz  # Rest frequency of CO J(1->0)

        N_FREQ = NSIDEBAND * NCHANNEL
        nu = np.linspace(params.sim_nu_f, params.sim_nu_i, N_FREQ) * u.GHz  

        dnu = nu[1] - nu[0]

        dredshift = (1 + Z_MID) ** 2 * dnu / NU_REST

        angle2Mpc = cosmology.kpc_comoving_per_arcmin(Z_MID).to(u.Mpc / u.arcmin)
        
        self.angle2Mpc = angle2Mpc

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

        x_edges = (x_edges * np.abs(np.cos(np.radians(y_edges[NDEC // 2]))) * u.deg * angle2Mpc).to(u.Mpc)
        y_edges = (y_edges * u.deg * angle2Mpc).to(u.Mpc)

        # Cosmological distance corresponding to redshift width in middle of box
        dz = cosmology.comoving_distance(Z_MID + dredshift / 2) - cosmology.comoving_distance(
            Z_MID - dredshift / 2
        )

        # Generating equispaced cosmological grid from redshifts, relative to first frequency
        z = np.arange(0, N_FREQ) * dz

        # Spacial resolutions
        dx = np.abs(x_edges[1] - x_edges[0]) 
        dy = np.abs(y_edges[1] - y_edges[0]) 
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

        self.nyquist_x = 2 * np.pi * np.abs(np.fft.fftfreq(self.nx, self.dx)).max()
        self.nyquist_y = 2 * np.pi * np.abs(np.fft.fftfreq(self.ny, self.dy)).max()
        self.nyquist_z = 2 * np.pi * np.abs(np.fft.fftfreq(self.nz, self.dz)).max()

        self.min_k_x = np.sort(2 * np.pi * np.abs(np.fft.fftfreq(self.nx, self.dx)))[1]
        self.min_k_y = np.sort(2 * np.pi * np.abs(np.fft.fftfreq(self.ny, self.dy)))[1]
        self.min_k_z = np.sort(2 * np.pi * np.abs(np.fft.fftfreq(self.nz, self.dz)))[1]


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
