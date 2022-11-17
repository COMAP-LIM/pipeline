import argparse
from typing import Dict, Any
import h5py
import numpy as np
import numpy.typing as ntyping
from dataclasses import dataclass, field
import re


@dataclass
class L2file:
    """COMAP l2 data class"""

    path: str
    _data: Dict[str, ntyping.ArrayLike] = field(default_factory=dict)

    def read_l2(self) -> None:
        """Function for reading l2 data from file and fill data dictionary of Map class"""

        # Empty data dict
        self._data = {}

        # Open and read file
        with h5py.File(self.path, "r") as infile:
            for key, value in infile.items():
                # Copy dataset data to data dictionary
                self._data[key] = value[()]

        self.keys = self._data.keys()

    def write_l2(self, outpath: str) -> None:
        """Method for writing l2 data to file.

        Args:
            outpath (str): path to save output l2 file.
        """

        if len(self._data) == 0:
            raise ValueError("Cannot save l2 if data object is empty.")
        else:
            print("Saving l2 to: ", outpath)
            with h5py.File(outpath, "w") as outfile:
                for key in self.keys:
                    outfile.create_dataset(key, data=self._data[key])

    def __getitem__(self, key: str) -> ntyping.ArrayLike:
        """Method for indexing l2 data as dictionary

        Args:
            key (str): Dataset key, corresponds to HDF5 l2 data keys

        Returns:
            dataset (ntyping.ArrayLike): Dataset from HDF5 l2 file
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
