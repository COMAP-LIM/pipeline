from __future__ import annotations
from typing import Optional
import numpy as np
import numpy.typing as npt
from pixell import enmap
from dataclasses import dataclass, field
import os
import sys
import copy 
import time

from astropy import wcs
from astropy.io import fits

field_id_dict = {
    "co2": "field1",
    "co7": "field2",
    "co6": "field3",
}

current = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current)
sys.path.append(parent_directory)
sys.path.append(os.path.join(parent_directory, "tod2comap"))
from COmap import COmap

# Ignore RuntimeWarning
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


class MapEditor():
    def __init__(self):
        
        self.read_params()
        self.run()
        
    def read_params(self):
        from l2gen_argparser import parser

        params = parser.parse_args()
        print(params.medit_map_names)
        if len(params.medit_map_names) < 2:
            raise ValueError(
                "A maplist (--medit_map_names) must be specified in parameter file or terminal."
            )
        
        if len(params.medit_outpath) < 2:
            raise ValueError(
                "No output path provided."
            )    
        
        self.params = params

    def run(self):
        t0 = time.perf_counter()
        print("Start")
        mappath = self.params.map_dir
        mapnames = self.params.medit_map_names
        mapnames = [os.path.join(mappath, file) for file in mapnames]
        self.comaps = []
        
        for file in mapnames:
            if not os.path.exists(file):
                raise ValueError(f"Provided map {file} not found!")
            comap = COmap(file)
            comap.read_map_keys()
            self.comaps.append(comap)
        
        
        self.outmap = COmap(os.path.join(mappath, self.params.medit_outpath))
        metadata_keys = [] 
        for key in comap.keys:
            if "multisplits" in key:
                if not "pca" in key and "map" in key:
                    self.coadd_maps(key)
                    self.outmap.write_dataset(key, delete = True)
                    self.outmap.write_dataset(key.replace("map", "sigma_wn"), delete = True)
                    self.outmap.write_dataset(key.replace("map", "nhit"), delete = True)
                continue
            elif (("map" in key) or ("nhit" in key)) or ("sigma_wn" in key):
                if not "pca" in key and "map" in key:
                    self.coadd_maps(key)
                    self.outmap.write_dataset(key, delete = True)
                    self.outmap.write_dataset(key.replace("map", "sigma_wn"), delete = True)
                    self.outmap.write_dataset(key.replace("map", "nhit"), delete = True)
                continue
            elif "params" in key:
                continue
            elif "wcs" in key:
                continue
            else:
                metadata_keys.append(key)
                comap.read_and_append([key])
                self.outmap[key] = copy.deepcopy(comap[key])         
                self.outmap.write_dataset(key)

        print(f"Done! Run time {time.perf_counter() - t0}")

    def coadd_maps(self, map_key: str):
        """Method that accumulates noise weighted sum of data in 
        output map

        Args:
            map_key (str): Map key to accumulate
        """
        
        if self.params.verbose:
            print(f"Coadding {map_key}:")
        
        sigma_key = map_key.replace("map", "sigma_wn")
        hit_key = map_key.replace("map", "nhit")
        for comap in self.comaps:
            comap.read_and_append([map_key, sigma_key, hit_key])
        
        outmap_inv_var = 1 / (self.comaps[0][sigma_key] ** 2) 
        outmap_inv_var[~np.isfinite(outmap_inv_var)] = 0
        outmap_d = self.comaps[0][map_key] * outmap_inv_var
        outmap_d[~np.isfinite(outmap_d)] = 0
        outmap_hit = self.comaps[0][hit_key]
        outmap_hit[~np.isfinite(outmap_hit)] = 0
        
        for comap in self.comaps[1:]:
            inv_var = 1 / (comap[sigma_key] ** 2) 
            inv_var[~np.isfinite(inv_var)] = 0
            d = comap[map_key] * inv_var
            d[~np.isfinite(d)] = 0
            outmap_d += d
            hit = comap[hit_key]
            hit[~np.isfinite(hit)] = 0
            outmap_hit += hit
            outmap_inv_var += inv_var
        
        outmap_d /= outmap_inv_var
        outmap_sigma = 1 / np.sqrt(outmap_inv_var)
        outmap_d[outmap_d == 0] = np.nan
        outmap_d[~np.isfinite(outmap_d)] = np.nan
        outmap_sigma[~np.isfinite(outmap_sigma)] = np.nan
        
        self.outmap[map_key] = outmap_d
        self.outmap[sigma_key] = outmap_sigma
        self.outmap[hit_key] = outmap_hit
        
        for comap in self.comaps:
            del comap[map_key]
            del comap[sigma_key]
            del comap[hit_key]
        
    
if __name__ == "__main__":
    editor = MapEditor()