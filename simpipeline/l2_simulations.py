from __future__ import annotations
import numpy as np
import h5py
import scipy.interpolate as interp
import numpy.typing as ntyping

from dataclasses import dataclass, field


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
            os.path.dirname(os.path.realpath(__file__)),
            f"standard_geometries/{fieldname}_standard_geometry.fits",
        )

        # standard_geometry_path = os.path.join(
        #     "/mn/stornext/d22/cmbco/comap/nils/comap_python/tod2comap",
        #     f"standard_geometries/{fieldname}_standard_geometry.fits",
        # )

        standard_geometry = enmap.read_map(standard_geometry_path).copy()

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
        simdata = simdata.reshape(
            simdata.shape[0] * simdata.shape[1], simdata.shape[2], simdata.shape[3]
        )
        simdata = simdata.transpose(1, 2, 0)
        print(simdata.shape)

        channels = np.arange(simdata.shape[-1], dtype=np.int32)

        self.signal = interp.RegularGridInterpolator((ra, dec, channels), simdata)
