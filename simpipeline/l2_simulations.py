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

    # Path from which to load simulation cube
    path: str = field(default_factory=str)

    # Private data storage dictionary
    _data: dict[str, ntyping.ArrayLike] = field(default_factory=dict)

    def read(self):
        """Method that reads in simulation cube from file"""
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

    def get_bin_centers(
        self, ra_edges: npt.NDArray, dec_edges: npt.NDArray
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """Method that takes in bin grid bin edges and returns pixel bin centers.

        Args:
            ra_edges (npt.NDArray): Right ascention grid bin edges.
            dec_edges (npt.NDArray): Declination grid bin edges

        Returns:
            tuple[npt.NDArray, npt.NDArray]: Tuple of npt.NDArray with right ascention and declination pixel centers.
        """
        # Pixel resolutions
        dra = ra_edges[1] - ra_edges[0]
        ddec = dec_edges[1] - dec_edges[0]

        # Compute pixel centers
        ra_centers = ra_edges[:-1] + dra / 2
        dec_centers = dec_edges[:-1] + ddec / 2

        return ra_centers, dec_centers

    def interpolate_cube(self):
        """Methode that computes a cubic spline of the rectangular simulation grid."""

        # Right ascention and declination bin edges to bin centers
        ra = self._data["x"]
        dec = self._data["y"]
        ra, dec = self.get_bin_centers(ra, dec)

        simdata = self._data["simulation"]

        # Empty array to fill up with callable interpolation functions for each frequency
        self.signal = [[] for _ in range(simdata.shape[0])]

        # Generating all interpolation slices
        for sb in range(simdata.shape[0]):
            for channel in range(simdata.shape[1]):

                interpolation = interp.RectBivariateSpline(
                    dec, ra, simdata[sb, channel, :, :]
                )

                self.signal[sb].append(interpolation)

    def rotate_pointing_to_equator(
        self, ra: npt.NDArray, dec: npt.NDArray
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """Method that rotates pixel grid or pointing from field centered grid to
        equatorial origin centered grid.

        Args:
            ra (npt.NDArray): Right ascention pointing or grid center
            dec (npt.NDArray): Declination pointing or grid center

        Returns:
            tuple[npt.NDArray, npt.NDArray]: Rotated pointing or grid centers.
        """

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

    def rotate_equator_to_field_center(
        self, ra: npt.NDArray, dec: npt.NDArray
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """Method that rotates pointing or pixel grid centers from equatorial origin to field centered grid.

        Args:
            ra (npt.NDArray): Right ascention pointing or grid center
            dec (npt.NDArray): Declination pointing or grid center

        Returns:
            tuple[npt.NDArray, npt.NDArray]: Rotated pointing or grid centers.
        """

        # A cartesian vector from ra, dec pointing
        vector = utils.ang2rect((np.deg2rad(ra), np.deg2rad(dec)))

        # Rotation matrix to rotate from field pointing to equator region around {ra, dec} = {0, 0}
        center_ra, center_dec = self.FIELD_CENTER
        # Rotation matrix in right ascention
        rotation_ra = utils.rotmatrix(np.deg2rad(center_ra), raxis="z")
        # Rotation matrix in declination
        rotation_dec = utils.rotmatrix(-np.deg2rad(center_dec), raxis="y")

        # Complete rotation matrix
        rotation_matrix = rotation_ra @ rotation_dec

        # Rotate pointing
        new_pointing = np.rad2deg(utils.rect2ang(rotation_matrix @ vector))
        new_ra, new_dec = new_pointing

        return new_ra, new_dec

    def bin_cube2field_geometry(
        self,
        geometry: enmap.ndmap,
    ) -> enmap.ndmap:
        """Method that takes simulation cube, rotates its pixel centers to field centerd
        grid, and bins up simualtion cube to provided map geometry.

        Args:
            geometry (enmap.ndmap): Target map geometry.

        Returns:
            enmap.ndmap: Rotated and binned simulation cube data.
        """

        # Get sizes of original simulation cube at equator
        NSB, NCHANNEL, NDEC_sim, NRA_sim = self.simdata.shape

        # Total number of frequencies
        NFREQ = NSB * NCHANNEL

        # Flipping RA axis to be consistant with geometry
        self.simdata = self.simdata.reshape(NFREQ, NDEC_sim, NRA_sim)[..., ::-1]

        # Number of angular grid points in target geometry
        NDEC, NRA = geometry.shape

        # Define source geometry situated at equatorial origin
        ra_edges = self._data["x"]
        dec_edges = self._data["y"]
        box = (
            np.array([[dec_edges[0], ra_edges[-1]], [dec_edges[-1], ra_edges[0]]])
            * utils.degree
        )
        shape, equator_wcs = enmap.geometry(
            pos=box, res=np.deg2rad(ra_edges[1] - ra_edges[0]), proj="car"
        )
        equator_geometry = enmap.zeros(shape, equator_wcs)

        # Get equatorial origin geometry grid
        dec_grid, ra_grid = np.rad2deg(equator_geometry.posmap())

        # Euler rotate grid centers to field center
        ra_grid_rotated, dec_grid_rotated = self.rotate_equator_to_field_center(
            ra_grid.flatten(),
            dec_grid.flatten(),
        )

        # Make joint array and change unites to radians for enmap to process coordinates
        coords = np.array((dec_grid_rotated, ra_grid_rotated))
        coords = np.deg2rad(coords)

        # Find rounded pixel index corresponding to telescope pointing given the simulation map geometry
        y_idx, x_idx = utils.nint(geometry.sky2pix(coords))

        # Clipping away all pointings that are outside target geometry patch
        y_idx = np.clip(y_idx.flatten(), 0, NDEC - 1)
        x_idx = np.clip(x_idx.flatten(), 0, NRA - 1)
        y_mask = np.where(~np.logical_and(y_idx >= 0, y_idx < NDEC))
        x_mask = np.where(~np.logical_and(x_idx >= 0, x_idx < NRA))
        self.simdata[:, y_mask, x_mask] = 0

        # Define pixel index from RA and Dec index
        px_idx = (NRA * y_idx + x_idx).flatten()

        # Extend pointing index to frequency dimension as well
        px_idx = (
            NDEC * NRA * np.arange(NFREQ, dtype=np.int32)[:, None] + px_idx[None, :]
        )
        px_idx = px_idx.flatten()

        # Number of hits per target geometry pixel
        hits = np.bincount(px_idx, minlength=NDEC * NRA * NFREQ)

        # Binned up simulation data per target geometry pixel
        new_simdata = np.bincount(
            px_idx, minlength=NDEC * NRA * NFREQ, weights=self.simdata.flatten()
        )

        # New simulation data is average of binned up values
        new_simdata = (new_simdata / hits).reshape(NFREQ, NDEC, NRA)

        # Overwrite and return new simulation data as enmap.ndmap object
        self.simdata = enmap.enmap(
            new_simdata.reshape(NSB, NCHANNEL, NDEC, NRA), geometry.wcs, copy=False
        )
        return self.simdata

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
