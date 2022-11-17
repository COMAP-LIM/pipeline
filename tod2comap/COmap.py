import argparse
from typing import Dict, Any
import h5py
import numpy as np
import numpy.typing as ntyping
from dataclasses import dataclass, field
import re


@dataclass
class COmap:
    """COMAP map data class"""

    path: str
    _data: Dict[str, ntyping.ArrayLike] = field(default_factory=dict)

    def read_map(self) -> None:
        """Function for reading map data from file and fill data dictionary of Map class"""

        # Empty data dict
        self._data = {}

        # Open and read file
        with h5py.File(self.path, "r") as infile:
            for key, value in infile.items():
                if isinstance(value, h5py._hl.group.Group):
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

    def write_map(self, outpath: str) -> None:
        """Method for writing map data to file.

        Args:
            outpath (str): path to save output map file.
        """

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
                    # print(key)
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
