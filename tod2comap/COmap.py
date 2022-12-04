from __future__ import annotations
from typing import Optional
import h5py
import numpy as np
import numpy.typing as ntyping
from dataclasses import dataclass, field
import re


@dataclass
class COmap:
    """COMAP map data class"""

    path: str = field(default_factory=str)
    _data: dict[str, ntyping.ArrayLike] = field(default_factory=dict)

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
        self, field_info: tuple, decimation_freqs: tuple, make_nhit: bool = False
    ) -> None:
        """Methode used to set up empty maps and field grid for given field.
        Both an empty numerator and denominator map are generated to acumulate
        data during binning of TODs. Optionally an empty hit map is generated.


        Args:
            field_info (tuple): Tuple containing;
                                * NSIDE: Number of pixels in RA and Dec.
                                * PIX_RES: pixel resolution (in degrees) in RA and Dec as
                                  an ArrayLike with size 2.
                                * FIELD_CENTER: RA and Dec field center coordinates (in degrees)
                                  as an ArrayLike with size 2.
            decimation_freqs (float): Number of frequency channels after decimation in l2gen.
            make_nhit (bool, optional): Boolean specifying whether to make an empty hit map.
        """
        GRID_SIZE, PIX_RES, FIELD_CENTER = field_info

        # Number of pixels in {RA, DEC}
        NSIDE_RA, NSIDE_DEC = GRID_SIZE

        # Number of feeds
        NFEED = 20

        # Number of sidebands
        NSB = 4

        if make_nhit:
            # Empty hit map
            self._data["nhit"] = np.zeros(
                (NFEED, NSIDE_RA, NSIDE_DEC, NSB * decimation_freqs), dtype=np.int32
            )

        # Empty denomitator map containing sum TOD / sigma^2
        self._data["numerator_map"] = np.zeros(
            (NFEED, NSIDE_RA, NSIDE_DEC, NSB * decimation_freqs), dtype=np.float32
        )

        # Empty denomitator map containing sum 1 / sigma^2
        self._data["denominator_map"] = np.zeros_like(self._data["numerator_map"])

        # RA/Dec grid
        RA_center = np.zeros(NSIDE_RA)
        DEC_center = np.zeros(NSIDE_DEC)
        RA_edge = np.zeros(NSIDE_RA + 1)
        DEC_edge = np.zeros(NSIDE_DEC + 1)

        dRA = PIX_RES[0]
        dDEC = PIX_RES[1]

        # Min values in RA/Dec. directions (i.e. bin edges)
        if NSIDE_RA % 2 == 0:
            RA_min = FIELD_CENTER[0] - dRA * NSIDE_RA / 2.0
        else:
            RA_min = FIELD_CENTER[0] - dRA * NSIDE_RA / 2.0 - dRA / 2.0

        if NSIDE_DEC % 2 == 0:
            DEC_min = FIELD_CENTER[1] - dDEC * NSIDE_DEC / 2.0
        else:
            DEC_min = FIELD_CENTER[1] - dDEC * NSIDE_DEC / 2.0 - dDEC / 2.0

        # Defining pixel centers
        RA_center[0] = RA_min + dRA / 2
        DEC_center[0] = DEC_min + dDEC / 2
        for i in range(1, NSIDE_RA):
            RA_center[i] = RA_center[i - 1] + dRA

        for i in range(1, NSIDE_DEC):
            DEC_center[i] = DEC_center[i - 1] + dDEC

        # Defining pixel edges
        RA_edge[0] = RA_min
        DEC_edge[0] = DEC_min
        for i in range(1, NSIDE_RA + 1):
            RA_edge[i] = RA_edge[i - 1] + dRA

        for i in range(1, NSIDE_DEC + 1):
            DEC_edge[i] = DEC_edge[i - 1] + dDEC

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

    def write_map(self, outpath: Optional[str] = None) -> None:
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
            outname = self.path.split("/")[-1]
            namelen = len(outname)
            outname = re.sub(r".h5", rf"_n{ncomps}_subtr.h5", outname)
            outpath = self.path[:-namelen]
            outpath += outname

        if len(self._data) == 0:
            raise ValueError("Cannot save map if data object is empty.")
        else:
            print("Saving map to: ", outpath)
            with h5py.File(outpath, "w") as outfile:
                for key in self.keys:
                    outfile.create_dataset(key, data=self._data[key])

    def __getitem__(self, key: str) -> ntyping.ArrayLike:
        """Method for indexing map data as dictionary

        Args:
            key (str): Dataset key, corresponds to HDF5 map data keys

        Returns:
            dataset (ntyping.ArrayLike): Dataset from HDF5 map file
        """

        return self._data[key]

    def __setitem__(self, key: str, value: ntyping.ArrayLike) -> None:
        """Method for saving value corresponding to key

        Args:
            key (str): Key to new dataset
            value (ntyping.ArrayLike): New dataset
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
