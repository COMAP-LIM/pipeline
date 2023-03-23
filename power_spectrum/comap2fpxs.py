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
import pickle


class COMAP2FPXS():
    def __init__(self):
        return NotImplemented

    def read_params(self):
        """Method reading and parsing the parameters from file or command line.

        Raises:
            ValueError: If no power spectrum directory is provided
            ValueError: if no COMAP map name is provided
            ValueError: _description_
            ValueError: _description_
            ValueError: _description_
        """
        from l2gen_argparser import parser

        self.params = parser.parse_args()

        self.power_spectrum_dir = self.params.power_spectrum_dir
        self.map_name = self.params.map_name
        self.map_dir = self.params.map_dir
        self.jk_def_file = self.params.jk_def_file
        self.accept_data_id_string = self.params.accept_data_id_string

        # Raising errors if required parameters are missing
        if self.power_spectrum_dir in None:
            raise ValueError(
                "A power spectrum data directory must be specified in parameter file or terminal."
            )
        if self.map_name is None:
            raise ValueError(
                "A map file name must be specified in parameter file or terminal."
            )
        if self.map_dir is None:
            raise ValueError(
                "A map data directory must be specified in parameter file or terminal."
            )
        if self.jk_def_file is None:
            raise ValueError(
                "Please specify a jk_def_file in parameter file or terminal."
            )
            
        if self.accept_data_id_string is None:
            raise ValueError(
                "Please specify a accept_data_id_string in parameter file or terminal."
            )

    
    def read_cosmology(self):
        """
        Method that reads in the standard cosmology to use form pickled astropy object.
        """
        
        cosmology_path = os.path.join(self.params.cosmology_path, self.params.cosmology_name)

        with open(cosmology_path, mode="rb") as file:
            self.cosmology = pickle.load(file)
        

