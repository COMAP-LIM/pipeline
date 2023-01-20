from __future__ import annotations
import numpy as np
import h5py
import scipy.interpolate as interp
import numpy.typing as ntyping

from dataclasses import dataclass, field
from pixell import enmap, utils
import os

import limlam_mocker as llm
from llm_mocker.extensions import llm_doppler as llmd

@dataclass
class SimGenerator:
    """class to make simulation cubes using the limlam_mocker package"""

    def __init__(self, params):

        # keep only the arguments set to begin with 'sim' in a separate dict
        # to pass them easily to the simulation functions
        param_dict = vars(params)
        simparams = llm.empty_table()
        for key, val in param_dict.items():
            if key[0:4] == 'sim_':
                setattr(simparams, key[4:], val)

        self.simparams = simparams

    def generate_co_sims(self):
        """
        Generator function to get the simulated LIM cube, and an associated catalogue
        of DM halos with simulated CO properties.
        """

        # pull the simulation parameters object out to make it easier to work with
        simpars = self.simparams

        # set up the output files to save to
        llm.make_output_filenames(self.simparams, output_dir=simpars.output_dir)

        # set up map objects to be output
        mapinst = llm.params_to_mapinst(simpars)

        # load halos from catalogue and trim to only the relevant ones
        halos, cosmo = llm.load_peakpatch_catalogue(simpars.halo_catalogue_file,
                                                    output=simpars.halo_output_file)
        halos = llm.cull_peakpatch_catalogue(halos, simpars.min_mass, mapinst)

        # calculate the luminosity of each halo
        halos.Lco = llm.Mhalo_to_Lco(halos, simpars.model_name, simpars.model_coeffs)

        # if broadening spectrally, find the velocity that the CO is assumed to be moving at
        # in each halo
        if simpars.freqbroaden:
            halos = llmd.get_halo_velocities(halos, cosmo, simpars)

        # make the maps, applying broadening in whichever axes were requested
        if simpars.freqbroaden and simpars.spacebroaden:
            mapinst.maps = llmd.Lco_to_map_doppler_synthbeam(halos, mapinst,
                                                             bincount=simpars.velocity_bins,
                                                             binattr=simpars.freq_attr,
                                                             fwhmattr=simpars.freq_attr,
                                                             freqrefine=simpars.freq_subpix,
                                                             xrefine=simpars.x_subpix,
                                                             beamfwhm=simpars.beam_fwhm)

        elif simpars.freqbroaden:
            mapinst.maps = llmd.Lco_to_map_doppler(halos, mapinst,
                                                     bincount=simpars.velocity_bins,
                                                     binattr=simpars.freq_attr,
                                                     fwhmattr=simpars.freq_attr,
                                                     freqrefine=simpars.freq_subpix)

        else:
            mapinst.maps = llm.Lco_to_map(halos, mapinst)

        # if requested, save the map as a data cube





@dataclass
class SimCube:
    """Class containing simulation cube data"""

    path: str = field(default_factory=str)
    _data: dict[str, ntyping.ArrayLike] = field(default_factory=dict)

    def __init__(self, path):
        self.path = path
        self._data = {}

    def read(self):
        with h5py.File(self.path, "r") as infile:
            for key, value in infile.items():
                self._data[key] = value[()]

        self.keys = self._data.keys()

    def __getitem__(self, key: str) -> ntyping.ArrayLike:
        """Method for indexing map data as dictionary

        Args:
            key (str): Dataset key, corresponds to HDF5 map data keys

        Returns:
            dataset (ntyping.ArrayLike): Dataset from HDF5 map file
        """

        return self._data[key]

    def get_bin_centers(self, ra_edges: np.ndarray, dec_edges: np.ndarray) -> tuple:
        dra = ra_edges[1] - ra_edges[0]
        ddec = dec_edges[1] - dec_edges[0]

        ra_centers = ra_edges[:-1] + dra / 2
        dec_centers = dec_edges[:-1] + ddec / 2

        return ra_centers, dec_centers

    def interpolate_cube(self, fieldname: str):
        import os
        from pixell import enmap

        standard_geometry_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
            f"tod2comap/standard_geometries/{fieldname}_standard_geometry.fits",
        )

        standard_geometry = enmap.read_map(standard_geometry_path).copy()

        standard_geometry = enmap.upgrade(standard_geometry, 4)

        # Number of pixels in {DEC, RA}
        NSIDE_DEC, NSIDE_RA = standard_geometry.shape
        FIELD_CENTER = np.degrees(
            enmap.pix2sky(
                (NSIDE_DEC, NSIDE_RA),
                standard_geometry.wcs,
                ((NSIDE_DEC - 1) / 2, (NSIDE_RA - 1) / 2),
            )
        )[::-1]

        ra = self._data["x"] + FIELD_CENTER[0]
        dec = self._data["y"] + FIELD_CENTER[1]

        ra, dec = self.get_bin_centers(ra, dec)

        simdata = self._data["simulation"]

        # channels = np.arange(simdata.shape[0], dtype=np.int32)
        # sidebands = np.arange(simdata.shape[1], dtype=np.int32)

        # self.signal = interp.RegularGridInterpolator(
        #     (sidebands, channels, ra, dec), simdata
        # )

        self.signal = [[] for _ in range(simdata.shape[0])]

        import time

        t0 = time.perf_counter()

        for sb in range(simdata.shape[0]):
            for channel in range(simdata.shape[1]):
                interpolation = interp.RectBivariateSpline(
                    ra, dec, simdata[sb, channel, :, :]
                )
                self.signal[sb].append(interpolation)
        print("Time iterpolation", time.perf_counter() - t0, "s")
        # self.signal = [
        #     interp.RectBivariateSpline(ra, dec, simdata[sb, channel, :, :])
        #     for sb in range(simdata.shape[0])
        #     for channel in range(simdata.shape[1])
        # ]

        # print("hei", self.signal)

    def prepare_geometry(self, fieldname: str, boost: float = 1) -> None:
        """Method that defines the WCS geometry of the signal cube given the
        COMAP standard geometries. The signal cube is then transformed into an
        enmap with the computed WCS geometry and an optional signal boost applied.

        Args:
            fieldname (str): Name of field from which to load standard geometry.
            boost (float, optional): Boost factor with which signal can be boosted. Defaults to 1.

        Returns:
            enmap.enmap: Simulation cube enmap object which contains geometry and simulated (boosted) signal.
        """

        # Path to standard geometry directory
        standard_geometry_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
            f"tod2comap/standard_geometries/{fieldname}_standard_geometry.fits",
        )

        # Load and upgrade standard geomtry
        standard_geometry = enmap.read_map(standard_geometry_path).copy()
        standard_geometry = enmap.upgrade(
            standard_geometry, (self._data["x"].size - 1) // standard_geometry.shape[1]
        )

        # Defining an enmap with geometry equal to upgraded standard geometry
        self.simdata = enmap.enmap(
            self._data["simulation"], standard_geometry.wcs, copy=False
        )

        # NOTE: this will change in future due to the changed clock frequency in early 2022
        # Flipping frequencies
        # self.simdata[(0, 2), ...] = self.simdata[(0, 2), ::-1, ...]

        # muK to K
        self.simdata *= 1e-6
        # Boost signal
        self.simdata *= boost

    def sim2tod(self, ra: np.ndarray, dec: np.ndarray) -> np.ndarray:
        """Method that, given the telescope pointing in RA and Dec, projects
        the signal from the singal cube into time-stream space for all feeds
        frequencies at the same time.

        Args:
            ra (np.ndarray): Right Acention from telescope pointing.
            dec (np.ndarray): Declination from telescope pointing.

        Returns:
            np.ndarray: Simulation signal in time-stream space along
            telescope pointing for all feeds and frequencies.
        """

        # Concatenating dec and ra, and changing units to radians
        coords = np.array((dec, ra))
        coords = np.deg2rad(coords)

        # Find rounded pixel index corresponding to telescope pointing given the simulation map geometry
        y_idx, x_idx = utils.nint(self.simdata.sky2pix(coords))

        # Return pointing along pointing
        return np.moveaxis(self.simdata[:, :, y_idx, x_idx], 2, 0)
