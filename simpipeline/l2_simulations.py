from __future__ import annotations
import numpy as np
import numpy.typing as npt
import h5py
import scipy.interpolate as interp
import numpy.typing as ntyping

from dataclasses import dataclass, field
from pixell import enmap, utils
import os


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

    def get_bin_centers(self, ra_edges: npt.NDArray, dec_edges: npt.NDArray) -> tuple:
        dra = ra_edges[1] - ra_edges[0]
        ddec = dec_edges[1] - dec_edges[0]

        ra_centers = ra_edges[:-1] + dra / 2
        dec_centers = dec_edges[:-1] + ddec / 2

        return ra_centers, dec_centers

    def interpolate_cube(self, fieldname: str):
        import os
        from pixell import enmap

        ra = self._data["x"]  # + self.FIELD_CENTER[0]
        dec = self._data["y"]  # + self.FIELD_CENTER[1]

        ra, dec = self.get_bin_centers(ra, dec)

        simdata = self._data["simulation"]

        self.signal = [[] for _ in range(simdata.shape[0])]

        for sb in range(simdata.shape[0]):
            for channel in range(simdata.shape[1]):

                interpolation = interp.RectBivariateSpline(
                    dec, ra, simdata[sb, channel, :, :]
                )

                self.signal[sb].append(interpolation)

    def rotate_pointing_to_equator(
        self, ra: npt.NDArray, dec: npt.NDArray
    ) -> tuple[npt.NDArray, npt.NDArray]:

        # A cartesian vector from ra, dec pointing
        vector = utils.ang2rect((np.deg2rad(ra), np.deg2rad(dec)))

        # Rotation matrix to rotate from field pointing to equator region around {ra, dec} = {0, 0}
        center_ra, center_dec = self.FIELD_CENTER
        # Rotation matrix in right ascention
        rotation_ra = utils.rotmatrix(-np.deg2rad(center_ra), raxis="z")
        # Rotation matrix in declination
        rotation_dec = utils.rotmatrix(np.deg2rad(center_dec), raxis="y")

        # Complete rotation matrix
        rotation_matrix = rotation_dec @ rotation_ra

        # Rotate pointing
        new_pointing = np.rad2deg(utils.rect2ang(rotation_matrix @ vector))
        new_ra, new_dec = new_pointing

        return new_ra, new_dec

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

        # Number of pixels in {DEC, RA}
        NSIDE_DEC, NSIDE_RA = standard_geometry.shape
        self.FIELD_CENTER = np.degrees(
            enmap.pix2sky(
                (NSIDE_DEC, NSIDE_RA),
                standard_geometry.wcs,
                ((NSIDE_DEC - 1) / 2, (NSIDE_RA - 1) / 2),
            )
        )[::-1]

        # NOTE: this will change in future due to the changed clock frequency in early 2022
        # Flipping frequencies
        # self.simdata[(0, 2), ...] = self.simdata[(0, 2), ::-1, ...]

        # muK to K
        self.simdata *= 1e-6
        # Boost signal
        self.simdata *= boost

    def sim2tod(self, ra: npt.NDArray, dec: npt.NDArray) -> npt.NDArray:
        """Method that, given the telescope pointing in RA and Dec, projects
        the signal from the singal cube into time-stream space for all feeds
        frequencies at the same time.

        Args:
            ra (npt.NDArray): Right Acention from telescope pointing.
            dec (npt.NDArray): Declination from telescope pointing.

        Returns:
            npt.NDArray: Simulation signal in time-stream space along
            telescope pointing for all feeds and frequencies.
        """

        # Concatenating dec and ra, and changing units to radians
        coords = np.array((dec, ra))
        coords = np.deg2rad(coords)

        # Find rounded pixel index corresponding to telescope pointing given the simulation map geometry
        y_idx, x_idx = utils.nint(self.simdata.sky2pix(coords))

        # Return pointing along pointing
        return np.moveaxis(self.simdata[:, :, y_idx, x_idx], 2, 0)
