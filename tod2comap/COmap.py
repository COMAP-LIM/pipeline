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
from astropy import wcs
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM

field_id_dict = {
    "co2": "field1",
    "co7": "field2",
    "co3": "field3",
}


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
        horizontal: bool = False,
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
        if horizontal:
            standard_geometry_path = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                f"standard_geometries/{fieldname}_horizontal_high_res_geometry.fits",  # NOTE: change this later to something more general
            )
        else:
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
            "CTYPE1": self.standard_geometry.wcs.wcs.ctype[0],
            "CDELT1": self.standard_geometry.wcs.wcs.cdelt[0],
            "CRPIX1": self.standard_geometry.wcs.wcs.crpix[0],
            "CRVAL1": self.standard_geometry.wcs.wcs.crval[0],
            "NAXIS1": NSIDE_RA,
            "CUNIT1": self.standard_geometry.wcs.wcs.cunit[
                0
            ].to_string(),  # Astropy unit to string
            "CTYPE2": self.standard_geometry.wcs.wcs.ctype[1],
            "CDELT2": self.standard_geometry.wcs.wcs.cdelt[1],
            "CRPIX2": self.standard_geometry.wcs.wcs.crpix[1],
            "CRVAL2": self.standard_geometry.wcs.wcs.crval[1],
            "NAXIS2": NSIDE_DEC,
            "CUNIT2": self.standard_geometry.wcs.wcs.cunit[
                1
            ].to_string(),  # Astropy unit to string
        }

        # By default the map is not a simulation map and not a pure simulation cube map
        self._data["is_sim"] = False
        self._data["is_simcube"] = False

    def get_hdu_list(self, map_key, wcs):
        # Primary header for saving important general metadata
        primary_header = fits.Header()
        primary = fits.PrimaryHDU(header=primary_header)

        field_name = os.path.basename(self.path).split("_")[0]  # i.e co2, co7 or co6
        field_id = field_id_dict[field_name]  # i.e. field1 field2 or field3

        primary_header["FIELDNM"] = field_name
        primary_header["FIELDID"] = field_id
        primary_header["ISPCASBR"] = self._data["is_pca_subtr"]
        primary_header["ISCUBE"] = self._data["is_simcube"]
        primary_header["ISSIM"] = self._data["is_sim"]

        primary_header["PATCHDEC"] = self._data["patch_center"][0]
        primary_header["PATCHRA"] = self._data["patch_center"][1]

        if self.params is not None and "params" not in self.keys:
            for idx, key in enumerate(
                vars(self.params)
            ):  # Writing entire parameter file to separate hdf5 group.
                if getattr(self.params, key) == None:  # hdf5 didn't like the None type.
                    param = "None"
                else:
                    param = str(getattr(self.params, key))

                # Add parameter to metadata and use the parameter name (key) as comment since parameter names are too long to be name of the header entry.
                primary_header[f"P{idx:07}"] = (param, key)

        NFEED = 20  # Number of feeds
        NSIDEBAND = self._data["n_sidebands"]
        NCHANNEL = self._data["n_channels"]

        NRA = self._data["n_ra"]
        NDEC = self._data["n_dec"]

        nhit_key = re.sub(r"map", "nhit", map_key)
        sigma_wn_key = re.sub(r"map", "sigma_wn", map_key)

        if "coadd" in map_key:
            map_data = (
                self._data[map_key].copy().reshape(NSIDEBAND * NCHANNEL, NDEC, NRA)
            )

            nhit_data = (
                self._data[nhit_key].copy().reshape(NSIDEBAND * NCHANNEL, NDEC, NRA)
            )

            sigma_wn_data = (
                self._data[sigma_wn_key].copy().reshape(NSIDEBAND * NCHANNEL, NDEC, NRA)
            )
        else:
            map_data = (
                self._data[map_key]
                .copy()
                .reshape(NFEED, NSIDEBAND * NCHANNEL, NDEC, NRA)
            )

            nhit_data = (
                self._data[nhit_key]
                .copy()
                .reshape(NFEED, NSIDEBAND * NCHANNEL, NDEC, NRA)
            )

            sigma_wn_data = (
                self._data[sigma_wn_key]
                .copy()
                .reshape(NFEED, NSIDEBAND * NCHANNEL, NDEC, NRA)
            )

        # Header from wcs_dict
        map_header = wcs.to_header()
        map_header["BTYPE"] = "Temperature"
        map_header["BUNIT"] = "K"

        map_hdu = fits.ImageHDU(map_data, header=map_header, name="MAP")

        nhit_header = wcs.to_header()
        nhit_header["BTYPE"] = "Number of hits"
        nhit_hdu = fits.ImageHDU(nhit_data, header=nhit_header, name="NHIT")

        sigma_wn_header = wcs.to_header()
        sigma_wn_header["BTYPE"] = "Temperature"
        sigma_wn_header["BUNIT"] = "K"
        sigma_wn_hdu = fits.ImageHDU(
            sigma_wn_data, header=sigma_wn_header, name="SIGMA_WN"
        )

        significance_header = wcs.to_header()
        significance_header["BTYPE"] = "Significance"
        significance_hdu = fits.ImageHDU(
            map_data / sigma_wn_data, header=significance_header, name="SIGNIFICANCE"
        )

        primary_header["HDU1"] = "MAP"
        # HDU lists with primary (metadata) and one of map, hit map and white noise map
        map_hdu_list = [
            primary,
            map_hdu,
        ]

        primary_header["HDU1"] = "NHIT"
        nhit_hdu_list = [
            primary,
            nhit_hdu,
        ]

        primary_header["HDU1"] = "SIGMA_WN"
        sigma_wn_hdu_list = [
            primary,
            sigma_wn_hdu,
        ]

        primary_header["HDU1"] = "SIGNIFICANCE"
        significance_hdu_list = [
            primary,
            significance_hdu,
        ]

        return map_hdu_list, nhit_hdu_list, sigma_wn_hdu_list, significance_hdu_list

    def write_map(
        self,
        outpath: Optional[str] = None,
        primary_splits: Optional[list] = None,
        params: Optional[argparse.Namespace] = None,
        save_hdf5=True,
        save_fits=False,
    ) -> None:
        """Method for writing map data to file.

        Args:
            outpath (str): path to save output map file.
        """

        self.params = params

        if not save_hdf5 and not save_fits:
            raise ValueError(
                "Make sure to chose either to save map as HDF5 or FITS file(s)."
            )

        if outpath is None:
            outpath = self.path

        if self._data["is_pca_subtr"]:
            # If PCA subtraction was performed append number
            # of components and "subtr" to path and name
            ncomps = self._data["n_pca"]
            norm_mode = self._data["pca_norm"]
            outname = outpath.split("/")[-1]
            namelen = len(outname)
            if self._data["pca_approx_noise"]:
                outname = re.sub(
                    r".h5", rf"_n{ncomps}_subtr_approx_{norm_mode}.h5", outname
                )
            else:
                outname = re.sub(r".h5", rf"_n{ncomps}_subtr_{norm_mode}.h5", outname)
            outpath = outpath[:-namelen]
            outpath += outname
        elif self._data["is_simcube"]:
            outname = self.path.split("/")[-1]
            namelen = len(outname)
            outname = re.sub(r".h5", rf"_simcube.h5", outname)
            outpath = self.path[:-namelen]
            outpath += outname

        if save_fits:
            wcs_coadd, wcs_full = self.get_full_wcs()

            outdir = os.path.dirname(outpath)
            fits_dir = os.path.basename(outpath).split(".h5")[0]

            outdir = os.path.join(outdir, fits_dir)

            # Make output directry if it does not exist
            if not os.path.exists(outdir):
                os.mkdir(outdir)

            ######## Save full feed map ########
            (
                full_map_hdu_list,
                full_nhit_hdu_list,
                full_sigma_wn_hdu_list,
                full_significance_hdu_list,
            ) = self.get_hdu_list("map", wcs_full)
            full_map_hdul = fits.HDUList(full_map_hdu_list)
            full_nhit_hdul = fits.HDUList(full_nhit_hdu_list)
            full_sigma_wn_hdul = fits.HDUList(full_sigma_wn_hdu_list)
            full_significance_hdul = fits.HDUList(full_significance_hdu_list)

            full_map_outname = os.path.join(outdir, f"{fits_dir}_temperature.fits")
            full_nhit_outname = os.path.join(outdir, f"{fits_dir}_nhit.fits")
            full_sigma_wn_outname = os.path.join(outdir, f"{fits_dir}_sigma_wn.fits")
            full_significance_outname = os.path.join(
                outdir, f"{fits_dir}_significance.fits"
            )

            print("Saving feed map to: ", full_map_outname)
            print("Saving feed nhit to: ", full_nhit_outname)
            print("Saving feed sigma_wn to: ", full_sigma_wn_outname)
            print("Saving feed significance to: ", full_significance_outname)

            full_map_hdul.writeto(full_map_outname, overwrite=True)
            full_nhit_hdul.writeto(full_nhit_outname, overwrite=True)
            full_sigma_wn_hdul.writeto(full_sigma_wn_outname, overwrite=True)
            full_significance_hdul.writeto(full_significance_outname, overwrite=True)

            ######## Save feed-coadded map ########
            (
                coadd_map_hdu_list,
                coadd_nhit_hdu_list,
                coadd_sigma_wn_hdu_list,
                coadd_significance_hdu_list,
            ) = self.get_hdu_list("map_coadd", wcs_coadd)
            coadd_map_hdul = fits.HDUList(coadd_map_hdu_list)
            coadd_nhit_hdul = fits.HDUList(coadd_nhit_hdu_list)
            coadd_sigma_wn_hdul = fits.HDUList(coadd_sigma_wn_hdu_list)
            coadd_significance_hdul = fits.HDUList(coadd_significance_hdu_list)

            coadd_map_outname = os.path.join(
                outdir, f"{fits_dir}_coadd_temperature.fits"
            )
            coadd_nhit_outname = os.path.join(outdir, f"{fits_dir}_coadd_nhit.fits")
            coadd_sigma_wn_outname = os.path.join(
                outdir, f"{fits_dir}_coadd_sigma_wn.fits"
            )
            coadd_significance_outname = os.path.join(
                outdir, f"{fits_dir}_coadd_significance.fits"
            )

            print("Saving feed-coadded map to: ", coadd_map_outname)
            print("Saving feed-coadded nhit to: ", coadd_nhit_outname)
            print("Saving feed-coadded sigma_wn to: ", coadd_sigma_wn_outname)
            print("Saving feed-coadded significance to: ", coadd_significance_outname)

            coadd_map_hdul.writeto(coadd_map_outname, overwrite=True)
            coadd_nhit_hdul.writeto(coadd_nhit_outname, overwrite=True)
            coadd_sigma_wn_hdul.writeto(coadd_sigma_wn_outname, overwrite=True)
            coadd_significance_hdul.writeto(coadd_significance_outname, overwrite=True)

            splits = False
            for key in self.keys:
                if "pca" in key:
                    continue

                if "map" in key:
                    if "multisplits" in key:
                        # outfile.create_dataset(f"{key}", data=self._data[key])
                        splits = True

                    elif "/" in key and "wcs" not in key:
                        splits = True

                    if splits:
                        # Create all needed groups for multisplits
                        split_basedir = os.path.join(outdir, "multisplits")
                        # Make output directry if it does not exist
                        if not os.path.exists(split_basedir):
                            os.mkdir(split_basedir)
                        which_split = key.split("_")[
                            1
                        ]  # e.g. want "elev0ambt1" of key name "map_elev0ambt1"
                        primary_split = which_split[:4]

                        split_dir = os.path.join(split_basedir, f"{primary_split}")
                        if not os.path.exists(split_dir):
                            os.mkdir(split_dir)

                        split_dir = os.path.join(split_basedir, f"{primary_split}")
                        if not os.path.exists(split_dir):
                            os.mkdir(split_dir)

                        (
                            split_map_hdu_list,
                            split_nhit_hdu_list,
                            split_sigma_wn_hdu_list,
                            split_significance_hdu_list,
                        ) = self.get_hdu_list(key, wcs_full)
                        split_map_hdul = fits.HDUList(split_map_hdu_list)
                        split_nhit_hdul = fits.HDUList(split_nhit_hdu_list)
                        split_sigma_wn_hdul = fits.HDUList(split_sigma_wn_hdu_list)
                        split_significance_hdul = fits.HDUList(
                            split_significance_hdu_list
                        )

                        split_map_outname = os.path.join(
                            split_dir, f"{fits_dir}_{which_split}_temperature.fits"
                        )
                        split_nhit_outname = os.path.join(
                            split_dir, f"{fits_dir}_{which_split}_nhit.fits"
                        )
                        split_sigma_wn_outname = os.path.join(
                            split_dir, f"{fits_dir}_{which_split}_sigma_wn.fits"
                        )
                        split_significance_outname = os.path.join(
                            split_dir, f"{fits_dir}_{which_split}_significance.fits"
                        )

                        # print("Saving split map to: ", split_map_outname)
                        # print("Saving split nhit to: ", split_nhit_outname)
                        # print(
                        #     "Saving split sigma_wn to: ",
                        #     split_sigma_wn_outname,
                        # )
                        # print(
                        #     "Saving split significance to: ",
                        #     split_significance_outname,
                        # )
                        split_map_hdul.writeto(split_map_outname, overwrite=True)
                        split_nhit_hdul.writeto(split_nhit_outname, overwrite=True)
                        split_sigma_wn_hdul.writeto(
                            split_sigma_wn_outname, overwrite=True
                        )
                        split_significance_hdul.writeto(
                            split_significance_outname, overwrite=True
                        )

        if save_hdf5:
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
                                f"multisplits/{primary_split}{key}",
                                data=self._data[key],
                            )
                        elif "params" in key:
                            for param_key in self._data["params"].keys():
                                # print(param_key)
                                outfile[f"params/{param_key}"] = self._data["params"][
                                    param_key
                                ]
                        else:
                            outfile.create_dataset(key, data=self._data[key])

                    if params is not None and "params" not in self.keys:
                        for key in vars(
                            params
                        ):  # Writing entire parameter file to separate hdf5 group.
                            if (
                                getattr(params, key) == None
                            ):  # hdf5 didn't like the None type.
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

    def get_full_wcs(self) -> tuple[astropy.wcs.wcs.WCS]:
        """Method that defines WCS with angular information from
        standard geometry and adds the frequency axis.

        Returns:
            tuple[astropy.wcs.wcs.WCS]: The WCS of the feed coadded maps and the feed maps.
        """
        angular_wcs_params = self._data["wcs"]

        NFEED = 20  # Number of feeds
        NSIDEBAND = self._data["n_sidebands"]
        NCHANNEL = self._data["n_channels"]

        wcs_dict_coadd = {
            "CTYPE1": angular_wcs_params["CTYPE1"],
            "CUNIT1": angular_wcs_params["CUNIT1"],
            "CRVAL1": angular_wcs_params["CRVAL1"],
            "CDELT1": angular_wcs_params["CDELT1"],
            "CRPIX1": angular_wcs_params["CRPIX1"],
            "NAXIS1": angular_wcs_params["NAXIS1"],
            "CTYPE2": angular_wcs_params["CTYPE2"],
            "CUNIT2": angular_wcs_params["CUNIT2"],
            "CRVAL2": angular_wcs_params["CRVAL2"],
            "CDELT2": angular_wcs_params["CDELT2"],
            "CRPIX2": angular_wcs_params["CRPIX2"],
            "NAXIS2": angular_wcs_params["NAXIS2"],
            "CTYPE3": "FREQ",
            "CUNIT3": "Hz",
            "CRVAL3": 26.015625
            * 1e9,  # NOTE: CHANGE THIS LATER ONCE JONAS DEFINES STANDARD FREQ GRID IN L2GEN
            "CDELT3": (31.25)
            * 1e6,  # NOTE: CHANGE THIS LATER ONCE JONAS DEFINES STANDARD FREQ GRID IN L2GEN
            "CRPIX3": 1,
            "NAXIS3": NSIDEBAND * NCHANNEL,
            "RESTFRQ": 115.27,  # For CO (J 1->0)
        }

        # Must use decode to convert from bytes type to string, after extracting string from HDF5. Can only save bytes string to HDF5.
        for key, item in wcs_dict_coadd.items():
            if isinstance(item, bytes):
                wcs_dict_coadd[key] = item.decode()

        wcs_dict_full = wcs_dict_coadd.copy()

        # Add dimensionless feed axis
        wcs_dict_full["CTYPE4"] = "FEED"
        wcs_dict_full["CUNIT4"] = " "  # Feeds have no unit
        wcs_dict_full["CRVAL4"] = 0
        wcs_dict_full["CRPIX4"] = 0
        wcs_dict_full["CDELT4"] = 1
        wcs_dict_full["NAXIS4"] = "NFEED"

        return wcs.WCS(wcs_dict_coadd), wcs.WCS(wcs_dict_full)
