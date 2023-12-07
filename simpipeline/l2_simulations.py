from __future__ import annotations
import numpy as np
import numpy.typing as npt
from typing import Optional
import h5py
import scipy.interpolate as interp
import numpy.typing as ntyping
import sys

from dataclasses import dataclass, field
from pixell import enmap, utils
import os

current = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current)

sys.path.append(parent_directory)

from simpipeline.limsim_tools import *
from simpipeline.load_halos import *
from simpipeline.make_sim_maps import *
from simpipeline.generate_luminosities import *

class SimParameters():
    """
    simple Class creating an empty table
    used for halo catalogue and map instances
    """
    def __init__(self):
        pass

    def copy(self):
        """@brief Creates a copy of the table."""
        return copy.deepcopy(self)

    def print(self):
        attrlist = []
        for i in dir(self):
            if i[0]=='_': continue
            elif i == 'copy': continue
            else: attrlist.append(i)
        print(attrlist)

@dataclass
class SimGenerator:
    """class to make simulation cubes using the limlam_mocker package"""

    def __init__(self, params):

        # keep only the arguments set to begin with 'sim' in a separate dict
        # to pass them easily to the simulation functions
        param_dict = vars(params)
        simparams = SimParameters()
        for key, val in param_dict.items():
            if key[0:4] == 'sim_':
                setattr(simparams, key[4:], val)

        # function to broaden the spectral axes
        simparams.filterfunc = gaussian_filter1d

        self.simparams = simparams

    def run(self):

        simpars = self.simparams

        # generate the halo catalog from the passed peak-patch catalog
        halos = HaloCatalog(simpars, simpars.halo_catalogue_file)

        # generate luminosities for each halo
        Mhalo_to_Ls(halos, simpars)

        # set up map parameters
        map = SimMap(simpars)

        # make the map
        map.mockmapmaker(halos, simpars)

        # output files for map and catalog
        simpars.map_output_file = simpars.output_dir +'/'+ simpars.map_output_file_name
        simpars.cat_output_file = simpars.output_dir +'/'+ simpars.cat_output_file_name

        map.write(simpars)
        halos.write_cat(simpars)

        return




@dataclass
class SimCube:
    """Class containing simulation cube data"""

    # Path from which to load simulation cube
    path: str = field(default_factory=str)

    # Private data storage dictionary
    _data: dict[str, ntyping.ArrayLike] = field(default_factory=dict)
    
    def read(self, boost: float = 1, dec_indices: Optional[tuple[int, int]] = (None, None), ra_indecies: Optional[tuple[int, int]] = (None, None)):
        """Method that reads in simulation cube from file
        
        Args:
            boost (float, optional): Boost factor with which signal can be boosted. Defaults to 1.
            dec_indecies (tuple[int]): Tuple containing min and max of declination index to slice out of simulation cube. 
            ra_indecies (tuple[int]): Tuple containing min and max of right ascension index to slice out of simulation cube. 
        """
        min_dec_idx, max_dec_idx = dec_indices
        min_ra_idx, max_ra_idx = ra_indecies

        if ".h5" in self.path: 
            self.npz_format = False
            with h5py.File(self.path, "r") as infile:
                
                # Note that grid in RA direction is flipped since cubes are not saved in astronomical decreasing RA standard 
                self._data["simulation"] = enmap.enmap(infile["simulation"][..., min_dec_idx:max_dec_idx, min_ra_idx:max_ra_idx], wcs = self.equator_geometry.wcs, dtype = np.float32)

                # Pad a few pixels to be sure when slicing the signal cube
                if (min_dec_idx is not None or max_dec_idx is not None) or (min_ra_idx is not None or max_ra_idx is not None):
                    self.equator_geometry = enmap.pad(self.equator_geometry, pix = 5)
                    self._data["simulation"] = enmap.pad(self._data["simulation"], pix = 5)

        
        elif ".npz" in self.path:
            self.npz_format = True
            with np.load(self.path) as infile:
                self._data["simulation"] = infile["map_cube"][min_dec_idx:max_dec_idx, min_ra_idx:max_ra_idx, ...]
                self._data["simulation"] = self._data["simulation"].transpose(2, 0, 1)
                nfreq, ndec, nra = self._data["simulation"].shape
                # Flipping frequency axis of box to match that of signal injection mechanism
                self._data["simulation"] = self._data["simulation"][::-1, ...] 
                self._data["simulation"] = self._data["simulation"].reshape(4, nfreq // 4, ndec, nra)
                self._data["simulation"] = np.ascontiguousarray(self._data["simulation"])
        else:
            raise NameError("Provided path to simulation cube is not of a valid format. Please use HDF5 or npz format.")
        
        self.keys = self._data.keys()

        if self._data["simulation"].dtype != np.float32:
            self._data["simulation"] = self._data["simulation"].astype(np.float32) 

        # muK to K
        self._data["simulation"] *= 1e-6
        # Boost signal
        self._data["simulation"] *= boost

        self.simdata = self._data["simulation"]

    def read_cube_geometry(self):
        """Method that reads in simulation cube bin edges/centers from file"""

        if ".h5" in self.path: 
            self.npz_format = False
            with h5py.File(self.path, "r") as infile:
                for key, value in infile.items():
                    if "simulation" not in key:
                        self._data[key] = value[()]

            self._data["x_centers"] = self._data["x_bin_centers"]
            self._data["y_centers"] = self._data["y_bin_centers"]
            self._data["x_edges"] = self._data["x_bin_edges"]
            self._data["y_edges"] = self._data["y_bin_edges"]
                
                
        elif ".npz" in self.path:
            self.npz_format = True
            with np.load(self.path) as infile:
                self._data["x_centers"] = infile["map_pixel_ra"] 
                self._data["y_centers"] = infile["map_pixel_dec"]

                self._data["x_edges"] = np.zeros(self._data["x_centers"].size + 1)
                self._data["y_edges"] = np.zeros(self._data["y_centers"].size + 1)

                dx = self._data["x_centers"][1] - self._data["x_centers"][0]

                dy = self._data["y_centers"][1] - self._data["y_centers"][0]

                self._data["x_edges"][:-1] = self._data["x_centers"] - dx / 2
                self._data["x_edges"][-1] = self._data["x_centers"][-1] + dx / 2

                self._data["y_edges"][:-1] = self._data["y_centers"] - dy / 2
                self._data["y_edges"][-1] = self._data["y_centers"][-1] + dy / 2
                
                self._data["frequencies"] = infile["map_frequencies"][::-1] 
                nfreq, = self._data["frequencies"].shape
                self._data["frequencies"] = self._data["frequencies"].reshape(4, nfreq // 4) 
        else:
            raise NameError("Provided path to simulation cube is not of a valid format. Please use HDF5 or npz format.")

        ra_edges = self._data["x_edges"]
        dec_edges = self._data["y_edges"]
    
        # box = (
        #     np.array([[dec_edges[0], ra_edges[-1]], [dec_edges[-1], ra_edges[0]]])
        #     * utils.degree
        # )

        box = (
            np.array([[dec_edges[0], ra_edges[0]], [dec_edges[-1], ra_edges[-1]]])
            * utils.degree
        )

        shape, equator_wcs = enmap.geometry(
            pos=box, res=np.deg2rad(ra_edges[1] - ra_edges[0]), proj="car", force = True
        )
        self.equator_geometry = enmap.zeros(shape, equator_wcs)

    def __getitem__(self, key: str) -> ntyping.ArrayLike:
        """Method for indexing map data as dictionary

        Args:
            key (str): Dataset key, corresponds to HDF5 map data keys

        Returns:
            dataset (ntyping.ArrayLike): Dataset from HDF5 map file
        """

        return self._data[key]

    def get_bin_centers(self) -> tuple[npt.NDArray, npt.NDArray]:
        """Method that takes equator geometry and returns pixel bin centers.
        Returns:
            tuple[npt.NDArray, npt.NDArray]: Tuple of npt.NDArray with right ascention and declination pixel centers.
        """
    
        # Get equatorial origin geometry grid

        dec_grid, ra_grid = np.rad2deg(self.equator_geometry.posmap())

        ra_centers = ra_grid[0, :]
        dec_centers = dec_grid[:, 0]
        return ra_centers, dec_centers

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
        new_pointing = utils.rect2ang(rotation_matrix @ vector)
        new_ra, new_dec = new_pointing

        return np.rad2deg(new_ra), np.rad2deg(new_dec)

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
        new_pointing = utils.rect2ang(rotation_matrix @ vector)
        new_ra, new_dec = new_pointing

        return np.rad2deg(new_ra), np.rad2deg(new_dec)

    def bin_cube2field_geometry(
        self,
        geometry: enmap.ndmap,
    ) -> enmap.ndmap:
        """Method that takes simulation cube, rotates its pixel centers to field centered
        grid, and bins up simualtion cube to provided map geometry.

        Args:
            geometry (enmap.ndmap): Target map geometry.

        Returns:
            enmap.ndmap: Rotated and binned simulation cube data.
        """
        
        # Get sizes of original simulation cube at equator
        NSB, NCHANNEL, NDEC_sim, NRA_sim = self._data["simulation"].shape

        # Total number of frequencies
        NFREQ = NSB * NCHANNEL

        # Flipping RA axis to be consistant with geometry
        # self._data["simulation"] = self._data["simulation"].reshape(NFREQ, NDEC_sim, NRA_sim)[..., ::-1]
        self._data["simulation"] = self._data["simulation"].reshape(NFREQ, NDEC_sim, NRA_sim)

        # Number of angular grid points in target geometry
        NDEC, NRA = geometry.shape

        equator_geometry = self.equator_geometry.copy()

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
        self._data["simulation"][:, y_mask, x_mask] = 0

        # Define pixel index from RA and Dec index
        px_idx = (NRA * y_idx + x_idx).flatten()

        # Extend pointing index to frequency dimension as well
        px_idx = (
            NDEC * NRA * np.arange(NFREQ, dtype=np.int64)[:, None] + px_idx[None, :]
        )
        px_idx = px_idx.flatten()

        # Number of hits per target geometry pixel
        hits = np.bincount(px_idx, minlength=NDEC * NRA * NFREQ)

        # Binned up simulation data per target geometry pixel
        new_simdata = np.bincount(
            px_idx, minlength=NDEC * NRA * NFREQ, weights=self._data["simulation"].flatten()
        )

        # New simulation data is average of binned up values
        new_simdata = (new_simdata / hits).reshape(NFREQ, NDEC, NRA)

        # Overwrite and return new simulation data as enmap.ndmap object
        self._data["simulation"] = enmap.enmap(
            new_simdata.reshape(NSB, NCHANNEL, NDEC, NRA), geometry.wcs, copy=False
        )
        return self._data["simulation"]

    def prepare_geometry(self, fieldname: str) -> None:
        """Method that defines the WCS geometry of the signal cube given the
        COMAP standard geometries. The signal cube is then transformed into an
        enmap with the computed WCS geometry and an optional signal boost applied.

        Args:
            fieldname (str): Name of field from which to load standard geometry.

        """

        # Path to standard geometry directory
        standard_geometry_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
            f"tod2comap/standard_geometries/{fieldname}_standard_geometry.fits",
        )

        # Load and upgrade standard geomtry
        standard_geometry = enmap.read_map(standard_geometry_path).copy()
        standard_geometry = enmap.upgrade(
            standard_geometry, (self._data["x_edges"].size - 1) // standard_geometry.shape[1]
        )

        # Defining an enmap with geometry equal to upgraded standard geometry
        # self.simdata = enmap.enmap(
        #     self._data["simulation"], standard_geometry.wcs, copy=False
        # )

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


