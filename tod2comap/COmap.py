from __future__ import annotations
from typing import Optional
import h5py
import numpy as np
import numpy.typing as npt
from pixell import enmap
from dataclasses import dataclass, field
import re
import os
import argparse

from astropy import units as u
from astropy import constants
from astropy.cosmology import FlatLambdaCDM


@dataclass
class COmap:
    """COMAP map data class"""

    path: str = field(default_factory=str)
    _data: dict[str, npt.ArrayLike] = field(default_factory=dict)

    def read_map(self) -> None:
        """Function for reading map data from file and fill data dictionary of Map class"""

        # Empty data dict
        self._data = {}

        # Open and read file
        with h5py.File(self.path, "r") as infile:
            for key, value in infile.items():
                if isinstance(value, h5py.Group):
                    # If value is a group we want to copy the data in that group
                    if key == "multisplits":
                        try:
                            # For all parent splits
                            for split_key, split_group in value.items():
                                # For all datasets in parent split
                                for data_key, data_value in split_group.items():
                                    # Path to dataset
                                    complete_key = f"{key}/{split_key}/{data_key}"
                                    self._data[complete_key] = data_value[()]
                        except AttributeError:
                            continue
                    elif "wcs" in key:
                        # Reading in world coordinate system parameters
                        self._data[f"{key}"] = {}
                        for wcs_key, wcs_param in infile[key].items():
                            self._data[f"{key}"][wcs_key] = wcs_param[()]
                    elif "params" in key:
                        # Reading in parameter file parameters
                        self._data[f"{key}"] = {}
                        for params_key, params_param in infile[key].items():
                            self._data[f"{key}"][params_key] = params_param[()]
                    else:
                        # TODO: fill in if new groups are implemented in map file later
                        pass
                else:
                    # Copy dataset data to data dictionary
                    self._data[key] = value[()]

            if "is_pca_subtr" in infile.keys():
                self._data["is_pca_subtr"] = infile["is_pca_subtr"][()]
            else:
                self._data["is_pca_subtr"] = False

        self.keys = self._data.keys()

    def init_emtpy_map(
        self,
        fieldname: str,
        decimation_freqs: tuple,
        resolution_factor: float = 1,
        make_nhit: bool = False,
        maps_to_bin: list = ["numerator_map"],
    ) -> None:
        """Methode used to set up empty maps and field grid for given field.
        Both an empty numerator and denominator map are generated to acumulate
        data during binning of TODs. Optionally an empty hit map is generated.

        Args:
            fieldname (str): Name of field patch.
            decimation_freqs (float): Number of frequency channels after decimation in l2gen.
            resolution_factor (int): Integer factor to upgrade or downgrade standard
            geometry (2' pixels) with.
            make_nhit (bool, optional): Boolean specifying whether to make an empty hit map.
            maps_to_bin (optional, list): List of numerator map datasets to initialize.
            The same number of denominator maps will also be made. By default no splits are made
            and hence only one numerator and denominator map are made.
        """        

        standard_geometry_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            f"standard_geometries/{fieldname}_standard_geometry.fits",
        )

        self.standard_geometry = enmap.read_map(standard_geometry_path).copy()

        if resolution_factor > 1:
            self.standard_geometry = enmap.upgrade(
                self.standard_geometry, resolution_factor
            )
        elif resolution_factor < 1:
            resolution_factor = int(1 / resolution_factor)
            self.standard_geometry = enmap.downgrade(
                self.standard_geometry, resolution_factor
            )

        # Defining pixel centers
        # Ensuring that RA in (0, 360) degrees and dec in (-180, 180) degrees
        RA_center = np.degrees(self.standard_geometry.posmap()[1][0, :]) % 360
        DEC_center = (
            np.degrees(self.standard_geometry.posmap()[0][:, 0]) - 180
        ) % 360 - 180

        dRA, dDEC = self.standard_geometry.wcs.wcs.cdelt

        # Defining pixel edges
        RA_edge = RA_center - dRA / 2
        RA_edge = np.append(RA_edge, RA_center[-1] + dRA / 2)
        DEC_edge = DEC_center - dDEC / 2
        DEC_edge = np.append(DEC_edge, DEC_center[-1] + np.abs(dDEC) / 2)

        # Number of pixels in {RA, DEC}
        NSIDE_DEC, NSIDE_RA = self.standard_geometry.shape

        FIELD_CENTER = np.degrees(
            enmap.pix2sky(
                self.standard_geometry.shape,
                self.standard_geometry.wcs,
                ((NSIDE_DEC - 1) / 2, (NSIDE_RA - 1) / 2),
            )
        )[::-1]

        # Number of feeds
        NFEED = 20

        # Number of sidebands
        NSB = 4

        # Initializing empty split maps:
        for numerator_key in maps_to_bin:
            denominator_key = re.sub(r"numerator", "denominator", numerator_key)

            if make_nhit:
                # Empty hit map
                hit_key = re.sub(r"numerator_map", "nhit", numerator_key)

                self._data[hit_key] = np.zeros(
                    (NFEED, NSIDE_RA, NSIDE_DEC, NSB * decimation_freqs), dtype=np.int32
                )

            # Empty denomitator map containing sum TOD / sigma^2
            self._data[numerator_key] = np.zeros(
                (NFEED, NSIDE_RA, NSIDE_DEC, NSB * decimation_freqs), dtype=np.float32
            )

            # Empty denomitator map containing sum 1 / sigma^2
            self._data[denominator_key] = np.zeros_like(self._data["numerator_map"])

        # Save grid in private data dict
        self._data["ra_centers"] = RA_center
        self._data["dec_centers"] = DEC_center
        self._data["ra_edges"] = RA_edge
        self._data["dec_edges"] = DEC_edge

        # Save number of pixels
        self._data["n_ra"] = NSIDE_RA
        self._data["n_dec"] = NSIDE_DEC

        # Save number of sidebands and channels
        self._data["n_sidebands"] = NSB
        self._data["n_channels"] = decimation_freqs

        # Save patch center
        self._data["patch_center"] = np.array(FIELD_CENTER)

        # Minimum grid centers
        self.ra_min = RA_center[0]
        self.dec_min = DEC_center[0]

        # Define keys of internal map data dictionary
        self.keys = self._data.keys()

        # Map starts out not being PCA subtracted
        self._data["is_pca_subtr"] = False

        # Save World-Coordinate-System parameters (WCS) for
        # construction fits headers later.
        self._data["wcs"] = {
            "CDELT1": self.standard_geometry.wcs.wcs.cdelt[0],
            "CDELT2": self.standard_geometry.wcs.wcs.cdelt[1],
            "CRPIX1": self.standard_geometry.wcs.wcs.crpix[0],
            "CRPIX2": self.standard_geometry.wcs.wcs.crpix[1],
            "CRVAL1": self.standard_geometry.wcs.wcs.crval[0],
            "CRVAL2": self.standard_geometry.wcs.wcs.crval[1],
        }

        # By default the map is not a simulation map and not a pure simulation cube map
        self._data["is_sim"] = False
        self._data["is_simcube"] = False

    def write_map(
        self, outpath: Optional[str] = None, 
        primary_splits: Optional[list] = None, 
        params: Optional[argparse.Namespace] = None
    ) -> None:
        """Method for writing map data to file.

        Args:
            outpath (str): path to save output map file.
        """

        if not outpath:
            outpath = self.path

        if self._data["is_pca_subtr"]:
            # If PCA subtraction was performed append number
            # of components and "subtr" to path and name
            ncomps = self._data["n_pca"]
            norm_mode = self._data["pca_norm"]
            outname = self.path.split("/")[-1]
            namelen = len(outname)
            if self._data["pca_approx_noise"]:
                outname = re.sub(r".h5", rf"_n{ncomps}_subtr_approx_{norm_mode}.h5", outname)
            else:
                outname = re.sub(r".h5", rf"_n{ncomps}_subtr_{norm_mode}.h5", outname)
            outpath = self.path[:-namelen]
            outpath += outname
        elif self._data["is_simcube"]:
            outname = self.path.split("/")[-1]
            namelen = len(outname)
            outname = re.sub(r".h5", rf"_simcube.h5", outname)
            outpath = self.path[:-namelen]
            outpath += outname
            


        if len(self._data) == 0:
            raise ValueError("Cannot save map if data object is empty.")
        else:
            print("Saving map to: ", outpath)
            with h5py.File(outpath, "w") as outfile:
                outfile.create_group("wcs")

                if primary_splits is not None:
                    # Create all needed groups for multisplits
                    outfile.create_group("multisplits")

                    for primary_split in primary_splits:
                        outfile.create_group(f"multisplits/{primary_split}")

                for key in self.keys:
                    if "wcs" in key:
                        # Saving World Coordinate System parameters to group
                        for wcs_key, wcs_param in self._data["wcs"].items():
                            outfile.create_dataset(f"wcs/{wcs_key}", data=wcs_param)
                    elif "multisplits" in key:
                        outfile.create_dataset(f"{key}", data=self._data[key])

                    elif "/" in key:
                        # If splis are to be performed save data to correct group
                        primary_split = key.split("_")[-1]

                        primary_split = primary_split[:4]
                        outfile.create_dataset(
                            f"multisplits/{primary_split}{key}", data=self._data[key]
                        )
                    elif "params" in key: 
                        print(len(self._data["params"].keys()))
                        for param_key in self._data["params"].keys():
                            # print(param_key)
                            outfile[f"params/{param_key}"] = self._data["params"][param_key] 
                    else:
                        outfile.create_dataset(key, data=self._data[key])
                
                if params is not None and "params" not in self.keys:
                    for key in vars(params):  # Writing entire parameter file to separate hdf5 group.
                        if getattr(params, key) == None:  # hdf5 didn't like the None type.
                            outfile[f"params/{key}"] = "None"
                        else:
                            outfile[f"params/{key}"] = getattr(params, key)

    def __getitem__(self, key: str) -> npt.ArrayLike:
        """Method for indexing map data as dictionary

        Args:
            key (str): Dataset key, corresponds to HDF5 map data keys

        Returns:
            dataset (npt.ArrayLike): Dataset from HDF5 map file
        """

        return self._data[key]

    def __setitem__(self, key: str, value: npt.ArrayLike) -> None:
        """Method for saving value corresponding to key

        Args:
            key (str): Key to new dataset
            value (npt.ArrayLike): New dataset
        """
        # Set new item
        self._data[key] = value
        # Get new keys
        self.keys = self._data.keys()

    def __delitem__(self, key: str) -> None:
        """Method for deleting value corresponding to key

        Args:
            key (str): Key to dataset
        """
        # Set new item
        del self._data[key]



    