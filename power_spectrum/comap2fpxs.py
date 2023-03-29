from __future__ import annotations
from typing import Optional
import h5py
import numpy as np
from dataclasses import dataclass, field
import os
import sys
import pickle
import itertools
from mpi4py import MPI


import time 

current = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current)
sys.path.append(parent_directory)

import xs_class

class COMAP2FPXS():
    def __init__(self, omp_num_threads: int = 2):
        self.OMP_NUM_THREADS = omp_num_threads

        # Define MPI parameters as class attribites
        self.comm = MPI.COMM_WORLD
        self.Nranks = self.comm.Get_size()
        self.rank = self.comm.Get_rank()

        self.read_params()
        self.read_cosmology()
        self.read_jackknife_definition_file()
        self.generate_split_map_names()

    def run(self):
        feed_combinations = list(itertools.product(range(19), range(19)))

        mapnames = self.params.psx_map_names
        if self.params.null_cross_field:
            if len(mapnames) == 0:
                fields = self.params.fields
                mapnames = [f"{field_name}_{self.params.map_name}.h5" for field_name in fields]
            field_combinations = list(itertools.product(mapnames, mapnames))
            
            
        elif len(mapnames) == 0:
            fields = self.params.fields
            mapnames = [f"{field_name}_{self.params.map_name}.h5" for field_name in fields]
            field_combinations = [mapnames, mapnames]
        else:

            field_combinations = [mapnames, mapnames]


        all_combinations = list(itertools.product(field_combinations, self.split_map_combinations, feed_combinations))
        Number_of_combinations = len(all_combinations)
        

        

        for i in range(Number_of_combinations):
            if i % self.Nranks == self.rank:
                
                mapnames, splits, feeds = all_combinations[i]
                map1, map2 = mapnames
                split1, split2 = splits
                feed1, feed2 = feeds

                print(self.rank, map1, map2, split1, split2, feed1, feed2)

                mappaths = [
                    os.path.join(self.params.map_dir, map1),
                    os.path.join(self.params.map_dir, map2),
                ]

                cross_spectrum = xs_class.CrossSpectrum_nmaps(
                    mappaths, 
                    self.params, 
                    self.cosmology, 
                    splits, 
                    feed1, 
                    feed2
                    )
                

                ########################### REMOVE AFTER DEVELOPMENT
                time.sleep(5)
                ########################### REMOVE AFTER DEVELOPMENT
            if i ==5:
                break
            
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
        if self.power_spectrum_dir is None:
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
        
        cosmology_path = os.path.join(self.params.phy_cosmology_dir, self.params.phy_cosmology_name)

        with open(cosmology_path, mode="rb") as file:
            self.cosmology = pickle.load(file)

    def read_jackknife_definition_file(self):
        """Method that reads the jackknife/split definition file and outputs definition of split variables.
        """

        with open(self.params.jk_def_file, "r") as jk_file:
            all_lines = jk_file.readlines()

        # skip the first two lines (number of different jk and accr)
        all_lines = all_lines[
            2:
        ]  
        
        # Variables marked with "2" in jk_def_file. 
        # The first split in heirarchy/split-tree
        primary_variables = []  

        # Variables marked with "3" in jk_def_file.
        # The secondary successive splits in herarchy/split-tree
        secondary_variables = []  

        # The variable over which to perform cross-spectra. 
        # Marked with "1" in jk_def_file
        cross_variables = []  # extra 1

        # List of all split variables
        all_variables = []
        
        # Read and sort split variables to respective lists
        for line in all_lines:
            split_line = line.split()
            variable = split_line[0]
            number = split_line[1]

            if len(split_line) > 2:
                extra = split_line[2]
            
            if len(split_line) < 2 or len(split_line) == 2:
                extra = "#"

            all_variables.append(variable)

            if number == "3":
                secondary_variables.append(variable)

            if number == "2":
                primary_variables.append(variable)

            if extra == "1":
                cross_variables.append(variable)

        # Find all feed-feed variables that are also primary variables or secondary variables
        cross_and_primary = []
        cross_and_secondary = []

        for variable in all_variables:
            if variable in primary_variables and variable in cross_variables:
                cross_and_primary.append(variable)
            if variable in cross_variables and variable in secondary_variables:
                cross_and_secondary.append(variable)

        
        # Define as class attributes
        self.primary_variables  = primary_variables 
        self.secondary_variables  = secondary_variables 
        self.cross_variables  = cross_variables 
        self.all_variables  = all_variables 
        self.cross_and_primary  = cross_and_primary 
        self.cross_and_secondary  = cross_and_secondary 
    


    def generate_split_map_names(self):  
        secondary_variables = self.secondary_variables
        cross_and_primary = self.cross_and_primary
        
        split_map_combinations = []

        # If some primary variables are simultaneously feed-feed variables
        if (len(cross_and_primary) != 0):  
            
            number_of_secondary_variables = len(secondary_variables)  
            
            # Generate indices for all split combinations
            combinations = list(itertools.product(range(self.params.split_base_number), repeat = number_of_secondary_variables))  
            for primary_variable in cross_and_primary:

                # Generating names of split combinations
                for combo in combinations:
                    # name = primary_variable + "/"
                    name = ""
                    for i, bin_number in enumerate(combo):
                        name = name + f"{secondary_variables[i]}{bin_number}" 

                    split_map_combinations.append(
                        (f"multisplits/{primary_variable}/map_{primary_variable}{0}{name}",
                         f"multisplits/{primary_variable}/map_{primary_variable}{1}{name}",)
                    )

        self.split_map_combinations = split_map_combinations




        
if __name__ == "__main__":
    
    comap2fpxs = COMAP2FPXS()
    
    # print(comap2fpxs.primary_variables)
    # print(comap2fpxs.secondary_variables)
    # print(comap2fpxs.cross_variables)
    # print(comap2fpxs.all_variables)
    # print(comap2fpxs.cross_and_primary)
    # print(comap2fpxs.cross_and_secondary)
    # print(comap2fpxs.split_map_combinations)
    
    comap2fpxs.run()
