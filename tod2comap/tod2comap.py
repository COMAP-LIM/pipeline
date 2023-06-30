import numpy as np
import numpy.typing as npt
import ctypes
import time
import h5py
import re
from mpi4py import MPI
import datetime
from copy import copy, deepcopy
import itertools
from pixell import enmap, utils

from COmap import COmap
from L2file import L2file

import warnings
import tqdm 

# Ignore RuntimeWarning
warnings.filterwarnings("ignore", category=RuntimeWarning)

import os
import sys
import git

current = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current)

sys.path.append(parent_directory)
sys.path.append(os.path.join(parent_directory, "simpipeline/"))
from tools.read_runlist import read_runlist


class Mapmaker:
    """COMAP mapmaker class"""

    def __init__(self, omp_num_threads: int = 2):
        # Number of openMP threads
        self.OMP_NUM_THREADS = omp_num_threads

        # Define MPI parameters as class attribites
        self.comm = MPI.COMM_WORLD
        self.Nranks = self.comm.Get_size()
        self.rank = self.comm.Get_rank()

        # Read parameter file and runlist
        self.read_params()
        self.read_runlist()

        # Generate unique run ID to later enable easy identification of dataproducts
        runID = 0
        if self.rank == 0:
            runID = str(datetime.datetime.now())[2:].replace(" ", "-").replace(":", "-")
        runID = self.comm.bcast(runID, root=0)

        self.params.runID = int(runID.replace("-", "").replace(".", ""))

        # Define c-library used for binning maps
        mapbinner_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "../C_libs/tod2comap/mapbinner.so.1",
        )
        self.mapbinner = ctypes.cdll.LoadLibrary(mapbinner_path)

        # Computing allowed limits of noise in l2 TODs
        self.sample_time = 0.02  # seconds
        self.frequency_resolution = (
            1e9 * (34 - 26) / (4 * self.params.decimation_freqs)
        )  # Hz

        self.Tsys_limits = np.array([20, 100])  # [Min, Max] in K

        self.radiometer_limits = self.Tsys_limits / np.sqrt(
            self.sample_time * self.frequency_resolution
        )

        # Save git hash used
        dir_path = os.path.dirname(
            os.path.realpath(__file__)
        )  # Path to current directory.
        self.git_hash = git.Repo(
            dir_path, search_parent_directories=True
        ).head.object.hexsha  # Current git commit hash.
        self.params.git_hash = self.git_hash

        # Use unfiltered signal instead of data/data with signal
        self.use_signal_tod = False

    def read_params(self):
        from l2gen_argparser import parser

        params = parser.parse_args()
        if not params.runlist:
            raise ValueError(
                "A runlist must be specified in parameter file or terminal."
            )

        if not params.map_name:
            raise ValueError(
                "A map file name must be specified in parameter file or terminal."
            )

        self.jk_data_string = params.jk_data_string
        self.accept_data_id_string = params.accept_data_id_string
        self.accept_dir = params.accept_data_folder

        if not self.accept_data_id_string:
            message = (
                "Please specify a accept_data_id_string in parameter file or terminal."
            )
            raise ValueError(message)

        self.params = params

    def read_runlist(self):
        self.runlist = []
        self.fieldname = ""
        if self.rank == 0:
            self.runlist = read_runlist(self.params, ignore_existing=False)
            self.fieldname = self.runlist[0][4]
            
            for i in range(1, len(self.runlist)):
                if self.runlist[i][4] != self.fieldname:
                    raise ValueError(f"Mapmaker doesn't support multiple fields. Found both {self.fieldname} and {self.runlist[i][4]}.")
        self.runlist = self.comm.bcast(self.runlist, root=0)
        self.fieldname = self.comm.bcast(self.fieldname, root=0)
        

    def parse_accept_data(self):
        if len(self.jk_data_string) >= 1:
            self.jk_data_string = f"_{self.jk_data_string}"

        scan_data_path = f"scan_data_{self.accept_data_id_string}_{self.fieldname}.h5"
        split_data_path = f"jk_data_{self.accept_data_id_string}{self.jk_data_string}_{self.fieldname}.h5"

        scan_data_path = os.path.join(self.accept_dir, scan_data_path)
        with h5py.File(scan_data_path, "r") as scanfile:
            self.scandata = {}
            for key, value in scanfile.items():
                self.scandata[key] = value[()]

        split_data_path = os.path.join(self.accept_dir, split_data_path)

        with h5py.File(split_data_path, "r") as splitfile:
            self.splitdata = {}
            for key, value in splitfile.items():
                self.splitdata[key] = value[()]

        # List of map datasets we want to bin.
        # By default only the "non-split" maps will be binned
        self.maps_to_bin = ["numerator_map"]

        self.perform_splits = self.params.split
        if self.perform_splits:

            # Parse split definition file and save definitions to dictionary
            with open(self.params.jk_def_file, "r") as split_def_file:
                split_types_dict = {}
                # N_primary_splits = 0
                # N_secondary_splits = 0
                # N_temporal_splits = 0

                primary_splits = []
                secondary_splits = []
                temporal_splits = []
                temporal_primary_splits = []
                temporal_secondary_splits = []

                # Read split definition file
                for line in split_def_file.readlines()[2:]:
                    split_name, split_type = line.split()[:2]
                    split_types_dict[split_name] = int(split_type)
                    
                    extra = line.split()[-1]

                    # Count number of primary and secondary splits
                    if int(split_type) == 2:
                        if "$" in line:
                            # Temporal splits should be marked by dollar sign
                            temporal_splits.append(split_name)
                            temporal_primary_splits.append(split_name)
                            # N_temporal_splits += 1
                        # else:
                            # N_primary_splits += 1
                        primary_splits.append(split_name)
                        
                    elif int(split_type) == 3:
                        if "$" in line:
                            # Temporal splits should be marked by dollar sign
                            temporal_splits.append(split_name)
                            temporal_secondary_splits.append(split_name)
                            # N_temporal_splits += 1
                        # else:
                            # N_secondary_splits += 1
                        secondary_splits.append(split_name)

            N_primary_splits = len(primary_splits)
            N_secondary_splits = len(secondary_splits)
            N_temporal_splits = len(temporal_splits)
            N_temporal_primary_splits = len(temporal_primary_splits)
            N_temporal_secondary_splits = len(temporal_secondary_splits)
            
            split_names = self.splitdata["split_list"].astype(str)
            Nsplits = len(split_names)
            split_list = self.splitdata["jk_list"]

            # Define unique splits to be performed
            unique_numbers = np.unique(split_list)
            unique_numbers = unique_numbers[unique_numbers % 2 != 0]

            bits = np.ones((len(unique_numbers), Nsplits), dtype=np.int8)

            if self.rank == 0:
                print(f"Total number of splits: {unique_numbers.size}")
                print(f"Perform primary splits: {primary_splits}")
                print(f"For each primary split perform splits: {secondary_splits}")

            for i, number in enumerate(unique_numbers):
                # Convert number to bits
                bit_num = f"{number:0{Nsplits+1}b}"

                # Loop through all but last bit (accept/recect)
                for j in range(Nsplits):
                    bits[i, j] = int(bit_num[j])

            # Flipping bit digits to match split names read from file
            bits = bits[:, ::-1]

            # Dictionary to contain mapping between split id number and dataset name
            split_key_mapping = {}
            for i, number in enumerate(unique_numbers):
                # For each unique number generate mapping between number and split name
                for p, primary in enumerate(primary_splits):

                    if primary in temporal_splits:
                        key = f"{primary}"
                    else:
                        # Begin mapping name with primary key
                        key = f"{primary}"
                        key += f"{bits[i, N_secondary_splits + p]:d}"

                    # Add secondary keys to primary key
                    for s, secondary in enumerate(secondary_splits[::-1]):
                        
                        if secondary in temporal_splits:
                            key += f"{secondary}"
                        else:
                            key += f"{secondary}"
                            key += f"{bits[i,N_secondary_splits - 1 - s]:d}"

                    # Save mapping in dictionary
                    if key not in split_key_mapping.keys():
                        split_key_mapping[key] = np.array([number])
                    else:
                        split_key_mapping[key] = np.append(
                            split_key_mapping[key], [number]
                        )

            temporal_secondary_combinations = list(itertools.product(
                *(N_temporal_secondary_splits * [range(2)])
                ))

            temporal_secondary_splits = temporal_secondary_splits[::-1]

            # Adding temporal splits to key name but do not change bit flag
            # since bit flag only should map scan based splits. The temporal splits are implemented 
            # inside mapmaker at later stage based on the contained keys.
            _split_key_mapping = {}
            for key, num in split_key_mapping.items():
                if N_temporal_secondary_splits == 0:
                    new_keys = [key]
                else:
                    new_keys = []
                    for i, combos in enumerate(temporal_secondary_combinations):
                        sub_key = ""
                        sub_pattern = ""

                        # subtitute raw temporal key pattern with temporal keys 
                        for j, combo in enumerate(combos):
                            sub_key += f"{temporal_secondary_splits[j]}{combo}"
                            sub_pattern += fr"{temporal_secondary_splits[j]}"
                            # sub_key = f"{temporal_splits[j]}{combo}"
                            # sub_pattern = fr"{temporal_splits[j]}"
                        
                        new_keys.append(re.sub(rf"{sub_pattern}", f"{sub_key}", key))
                
                # subtitute raw temporal key pattern with temporal keys 
                for j in range(2):
                    if N_temporal_primary_splits == 0:                        
                        new_key = new_keys[0]
                        _split_key_mapping[new_key] = num 
                        continue

                    for i, temp_split in enumerate(temporal_primary_splits):
                
                        sub_key = ""
                        sub_pattern = ""
                        sub_key = f"{temp_split}{j}"
                        sub_pattern = fr"{temp_split}"

                        for new_key in new_keys:
                            new_key2 = re.sub(rf"{sub_pattern}", f"{sub_key}", new_key)
                            if len(new_key2) == 5 * (1 + N_secondary_splits):   # Igniring all new keys that miss the split bin number
                                _split_key_mapping[new_key2] = num
                            

            self.split_key_mapping = _split_key_mapping
            split_keys = list(self.split_key_mapping.keys())

            # Generating list of map keys that are to be binned up into maps
            # including the "non-split" map
            self.maps_to_bin += [
                f"/{self.maps_to_bin[0]}_{split_keys[i]}"
                for i in range(len(split_keys))
            ]

            # Removing dollar signs
            self.primary_splits = primary_splits
        

    def run(self):
        """Method running through the provided runlist and binning up maps."""
        self.comm.barrier()
        if self.rank == 0:
            start_time = time.perf_counter()
        # File name of full coadded output map
        full_map_name = f"{self.fieldname}_{self.params.map_name}.h5"
        full_map_name = os.path.join(self.params.map_dir, full_map_name)

        if os.path.exists(full_map_name):
            if self.rank == 0:
                print("Map already exists. Please delete or rename existing map to make new one.")
            sys.exit() 

        # Define and initialize empty map object to acumulate data
        full_map = COmap(full_map_name)
        full_map.init_emtpy_map(
            self.fieldname,
            self.params.decimation_freqs,
            self.params.res_factor,
            self.params.make_nhit,
            self.maps_to_bin,
            self.params.horizontal,
        )

        self.RA_min = full_map.ra_min
        self.DEC_min = full_map.dec_min

        time_array = np.zeros(6)
        rejection_number = np.zeros(1, dtype=np.int32)
        if self.rank == 0:
            time_buffer = time_array.copy()
            rejection_number_buffer = rejection_number.copy()
        else:
            time_buffer = None
            rejection_number_buffer = None

        if self.rank == 0:
            print("\n")
            progress_bar = tqdm.tqdm(
                total = len(self.runlist) // self.Nranks, 
                colour = "green", 
                ncols = 60
            )
        
        for i, scan in enumerate(self.runlist):

            scanid = scan[0]

            if i % self.Nranks == self.rank:
                # Cycle to next scan
                scan_idx = np.where(self.splitdata["scan_list"] == scanid)[0]
                if len(scan_idx) == 0:
                    print(f"Scan {scanid} in runlist but missing from accept_mod scanlist.")
                    # raise ValueError(f"Scan {scanid} in runlist but missing from accept_mod scanlist.")
                    rejection_number += 1
                    if self.rank == 0:
                        progress_bar.update(1)
                    continue

                scan_idx = scan_idx[0]
                if (
                    np.all(~self.splitdata["accept_list"][scan_idx])
                    and not self.params.override_accept
                ):
                    if self.params.verbose:
                        # Print in red forground color
                        print(
                            f"\033[91m Rejected scan {scanid} @ rank {self.rank} \033[00m"
                        )
                    rejection_number += 1
                    if self.rank == 0:
                        progress_bar.update(1)
                    continue

                if self.params.drop_first_scans and str(scanid)[-2:] == "02":
                    # Print in red forground color
                    if self.params.verbose:
                        print(
                            f"\033[91m Rejected scan {scanid} @ rank {self.rank} \033[00m"
                        )
                    rejection_number += 1
                    if self.rank == 0:
                        progress_bar.update(1)
                    continue

                if self.params.verbose:
                    # Print in green forground color
                    print(f"\033[92m Processing scan {scan[0]} @ rank {self.rank}\033[00m")
                
                l2path = scan[-1]

                ti = time.perf_counter()

                l2data = L2file(path=l2path, id=scanid)
                time_array[0] += time.perf_counter() - ti

                t0 = time.perf_counter()

                l2data.read_l2()
                time_array[1] += time.perf_counter() - t0

                try:
                    t0 = time.perf_counter()
                    self.preprocess_l2data(l2data)
                    time_array[2] += time.perf_counter() - t0
                except ValueError:
                    continue

                t0 = time.perf_counter()

                self.get_pointing_matrix(l2data, full_map)
                time_array[3] += time.perf_counter() - t0

                t0 = time.perf_counter()

                self.bin_map(full_map, l2data)
                time_array[4] += time.perf_counter() - t0

                time_array[5] += time.perf_counter() - ti


                if not self.params.verbose:
                    if self.rank == 0:
                        progress_bar.update(1)

        # Get frequency bin centers and edges from last level2 file.
        full_map["freq_centers"] = l2data["freq_bin_centers_lowres"]
        full_map["freq_edges"] = l2data["freq_bin_edges_lowres"]

        self.comm.Reduce(
            [time_array, MPI.DOUBLE], [time_buffer, MPI.DOUBLE], op=MPI.SUM, root=0
        )
        self.comm.Reduce(
            [rejection_number, MPI.INTEGER],
            [rejection_number_buffer, MPI.INTEGER],
            op=MPI.SUM,
            root=0,
        )
        if self.rank == 0:
            time_buffer /= len(self.runlist)

            print("\n" * 2 + "-" * 80)
            print(f"Average timing over {len(self.runlist)} scans:")
            print(
                f"Number of rejected scans {rejection_number_buffer[0]} of {len(self.runlist)}"
            )
            print("-" * 80)
            print("Time to define L2file:", 1e3 * time_buffer[0], "ms")
            print(
                "Time to read L2 data from HDF5:",
                1e3 * time_buffer[1],
                "ms",
            )
            print("Time to pre-process:", 1e3 * time_buffer[2], "ms")
            print("Time to compute pointing idx:", 1e3 * time_buffer[3], "ms")
            print("Time to bin map:", 1e3 * time_buffer[4], "ms")
            print("Total time for scan:", 1e3 * time_buffer[5], "ms")

            print("-" * 80)


        # Perform MPI reduce on map datasets
        self.reduce_maps(full_map)

        if self.rank == 0:

            if self.params.verbose and self.params.t2m_rms_mask_factor > 0:
                
                print(
                    f"Masking all sigma_wn > {self.params.t2m_rms_mask_factor} times the mean bottom 100 noise on each (feed, frequency):"
                )

            # self.postprocess_map(full_map, l2data)
            self.postprocess_map(full_map)
            if self.perform_splits:
                # Providing name of primary splits to make group names
                full_map.write_map(
                    primary_splits=self.primary_splits,
                    params=self.params,
                    save_hdf5=(not self.params.no_hdf5),
                    save_fits=(self.params.fits),
                )
            else:
                full_map.write_map(
                    params=self.params,
                    save_hdf5=(not self.params.no_hdf5),
                    save_fits=(self.params.fits),
                )
            finish_time = time.perf_counter()

            # Check if last l2data object contained simualtion and assume all did and make map of cube data only:
            if l2data["is_sim"] and self.params.populate_cube:
                self.populate_simulation_cube(full_map)

            print("=" * 80)
            print("=" * 80)
            print("Run time: ", finish_time - start_time, "s")
            print("=" * 80)
            print("=" * 80)

    def reduce_maps(self, mapdata: COmap) -> None:
        """Method that performes the MPI reduce on map datasets

        Args:
            mapdata (COmap): Map object with which to perform
            MPI reduce on. All data will be MPI reduced with MPI.SUM
            operation to rank 0 buffer.
        """

        for numerator_key in self.maps_to_bin:
            denominator_key = re.sub(r"numerator", "denominator", numerator_key)

            # Define buffer arrays
            if self.rank == 0:
                numerator_buffer = np.zeros_like(mapdata[numerator_key])
                denominator_buffer = np.zeros_like(mapdata[denominator_key])

                if self.params.make_nhit:
                    # Generate hit map keys
                    hit_key = re.sub(r"numerator_map", "nhit", numerator_key)
                    nhit_buffer = np.zeros_like(mapdata[hit_key])
            else:
                numerator_buffer = None
                denominator_buffer = None
                if self.params.make_nhit:
                    nhit_buffer = None

            # Perform MPI reduction
            self.comm.Reduce(
                [mapdata[numerator_key], MPI.FLOAT],
                [numerator_buffer, MPI.FLOAT],
                op=MPI.SUM,
                root=0,
            )

            self.comm.Reduce(
                [mapdata[denominator_key], MPI.FLOAT],
                [denominator_buffer, MPI.FLOAT],
                op=MPI.SUM,
                root=0,
            )

            if self.rank == 0:
                # Overwrite rank 0 datasets
                mapdata[numerator_key] = numerator_buffer
                mapdata[denominator_key] = denominator_buffer

            if self.params.make_nhit:
                # Generate hit map keys
                hit_key = re.sub(r"numerator_map", "nhit", numerator_key)

                self.comm.Reduce(
                    [mapdata[hit_key], MPI.INT],
                    [nhit_buffer, MPI.INT],
                    op=MPI.SUM,
                    root=0,
                )

                if self.rank == 0:
                    mapdata[hit_key] = nhit_buffer

    def preprocess_l2data(
        self,
        l2data: L2file,
    ):
        """Method that brings the feed infomation in needed datasets
        to correct correct order for binning to run optimally.

        Args:
            l2data (L2file): Level 2 file object to perform preprocessing on.
        """
        # NOTE: The level2 files will always contain a temporal mask.
        # The following if block is only there to add to this mask if temporal mask
        # keyword is provided in parameter file.
        if self.params.temporal_mask:
            l2data["mask_temporal"] *= self.get_temporal_mask(l2data)[None, :]

        _, NSB, NFREQ, NSAMP = l2data["tod"].shape

        # Defining empty buffers
        tod = np.zeros((20, NSB, NFREQ, NSAMP), dtype=np.float32)
        sigma0 = np.zeros((20, NSB, NFREQ), dtype=np.float32)
        freqmask = np.zeros((20, NSB, NFREQ), dtype=np.int32)
        pointing = np.zeros((20, NSAMP, 2), dtype=np.float32)
        temporal_mask = np.ones((NSAMP, 20), dtype=bool)

        # Get index to pixel mapping
        try:
            pixels = l2data["pixels"] - 1
        except KeyError:
            pixels = l2data["feeds"] - 1

        # Sort pixels to correct buffer position
        if "signal_simulation_tod" in l2data.keys and self.use_signal_tod:
            tod[pixels, ...] = l2data["signal_simulation_tod"]
        else:
            tod[pixels, ...] = l2data["tod"]

        sigma0[pixels, ...] = l2data["sigma0"]
        freqmask[pixels, ...] = l2data["freqmask"]
        pointing[pixels, ...] = l2data["point_cel"][..., :2]
        try:
            temporal_mask[:, pixels] = l2data["mask_temporal"].T
        except:
            pass

        # Check if noise level is above allowed limit
        if np.any(sigma0[sigma0 > 0] < self.radiometer_limits[0]):
            print(
                f"\033[95m WARNING: Scan: {l2data.id} lowest non-zero sigma_wn  {np.nanmin(sigma0[sigma0 > 0]):.5f} K < lower limit {self.radiometer_limits[0]:.5f} K! @ rank {self.rank} \033[00m"
            )

        # freqs = l2data["nu"][0, ...]

        # Flip sideband 0 and 2
        # tod[:, (0, 2), :, :] = tod[:, (0, 2), ::-1, :]
        # sigma0[:, (0, 2), :] = sigma0[:, (0, 2), ::-1]
        # freqmask[:, (0, 2), :] = freqmask[:, (0, 2), ::-1]
        # freqs[(0, 2), :] = freqs[(0, 2), ::-1]

        if not self.params.override_accept:
            # Masking accept mod rejected feeds and sidebands
            scan_idx = np.where(self.splitdata["scan_list"] == l2data.id)[0][0]
            rejected_feed, rejected_sideband = np.where(
                ~self.splitdata["accept_list"][scan_idx]
            )
            freqmask[rejected_feed, rejected_sideband, :] = 0

        # Ordering TOD axis so that fast freuquency axis is last
        tod = tod.transpose(
            0,
            3,
            1,
            2,
        )

        # Flatten frequency axis
        tod = tod.reshape(20, NSAMP, NSB * NFREQ)
        sigma0 = sigma0.reshape(20, NSB * NFREQ)
        freqmask = freqmask.reshape(20, NSB * NFREQ)

        # Enfore transposition in memory
        tod = np.ascontiguousarray(tod, dtype=np.float32)

        # Pre-compute masking
        masked_feeds, masked_freqs = np.where(freqmask == 0)
        inv_var = 1 / sigma0**2  # inverse variance
        inv_var[masked_feeds, masked_freqs] = 0
        tod[masked_feeds, :, masked_freqs] = 0

        nan_in_tod = np.any(~np.isfinite(tod))
        nan_in_inv_var = np.any(~np.isfinite(inv_var))

        if nan_in_tod or nan_in_inv_var:
            print(
                "\n"
                + "#" * 50
                + "\n"
                + f"Scan: {l2data.id}: found NaNs in; TOD {np.sum(~np.isfinite(tod))}, inv_var {np.sum(~np.isfinite(inv_var))}"
                + "\n"
                + "#" * 50
                + "\n"
            )
            tod[~np.isfinite(tod)] = 0
            inv_var[~np.isfinite(inv_var)] = 0
            # raise ValueError

        # Overwrite old data
        l2data["tod"] = tod
        l2data["inv_var"] = inv_var
        l2data["freqmask"] = freqmask
        l2data["mask_temporal"] = temporal_mask

        # Found that transposing the pixels here makes the binning
        # faster for some reason
        l2data["point_cel"] = np.ascontiguousarray(pointing.transpose(1, 0, 2))
        # l2data["nu"] = freqs

    # def postprocess_map(self, mapdata: COmap, l2data: L2file) -> None:
    def postprocess_map(self, mapdata: COmap) -> None:
        """Method that performes a post-processing of map object.
        In the post processing the map and noise maps are computed
        from the numerator and denominator maps, and feed-coadded maps
        are made. Optionally hit maps are also made. Lastly all map
        datasets are reshaped to the correct dimensionallity.

        Args:
            mapdata (COmap): Map object to perform the post-processing on.
            l2data (L2file): L2 file object from which to get frequency grid
            to be saved to map object.
        """

        # For each split dataset
        for numerator_key in self.maps_to_bin:
            # Generating map keys
            denominator_key = re.sub(r"numerator", "denominator", numerator_key)
            map_key = re.sub(r"numerator_map", "map", numerator_key)
            sigma_wn_key = re.sub(r"numerator_map", "sigma_wn", numerator_key)

            map_saddlebag_key = re.sub(r"numerator_map", "map_saddlebag", numerator_key)
            sigma_wn_saddlebag_key = re.sub(r"numerator_map", "sigma_wn_saddlebag", numerator_key)
            hit_saddlebag_key = re.sub(r"numerator_map", "nhit_saddlebag", numerator_key)


            if self.params.t2m_rms_mask_factor > 0:
                if mapdata["n_ra"] * mapdata["n_dec"] > 100: # Need more than 100 pixels to make propper noise mask
                    self.mask_map(mapdata, numerator_key, denominator_key)

            inv_var = mapdata[denominator_key]

            # Full map is made by computing map = sum(TOD * inv_var) / sum(inv_var)
            map = mapdata[numerator_key] / inv_var

            # White noise level map
            sigma = 1 / np.sqrt(inv_var)

            # Mask all non-hit regions for feed-coaddition
            map[inv_var == 0] = 0

            # Computing feed-coadded map and white noise map
            map_coadd = np.sum(map * inv_var, axis=0)
            sigma_coadd = np.sum(inv_var, axis=0)

            map_coadd /= sigma_coadd
            sigma_coadd = 1 / np.sqrt(sigma_coadd)

            sigma_coadd[np.isinf(sigma_coadd)] = np.nan

            if self.params.make_nhit:
                # Generate hit map keys
                hit_key = re.sub(r"numerator_map", "nhit", numerator_key)

                nhit = mapdata[hit_key]

                # Computing coadded hit map
                nhit_coadd = np.sum(nhit, axis=0)

                nhit = nhit.reshape(
                    20,
                    mapdata["n_ra"],
                    mapdata["n_dec"],
                    mapdata["n_sidebands"],
                    mapdata["n_channels"],
                )

                mapdata[hit_key] = nhit.transpose(0, 3, 4, 1, 2)

                if hit_key == "nhit":
                    nhit_coadd = nhit_coadd.reshape(
                        mapdata["n_ra"],
                        mapdata["n_dec"],
                        mapdata["n_sidebands"],
                        mapdata["n_channels"],
                    )
                    mapdata[f"{hit_key}_coadd"] = nhit_coadd.transpose(2, 3, 0, 1)

            if map_key == "map":
                # Saving coadded and reshaped data to map object
                map_coadd = map_coadd.reshape(
                    mapdata["n_ra"],
                    mapdata["n_dec"],
                    mapdata["n_sidebands"],
                    mapdata["n_channels"],
                )

                sigma_coadd = sigma_coadd.reshape(
                    mapdata["n_ra"],
                    mapdata["n_dec"],
                    mapdata["n_sidebands"],
                    mapdata["n_channels"],
                )

                # Changing axis order to old standard
                mapdata[f"{map_key}_coadd"] = map_coadd.transpose(2, 3, 0, 1)
                mapdata[f"{sigma_wn_key}_coadd"] = sigma_coadd.transpose(2, 3, 0, 1)

            # Masking non-hit regions with NaNs again
            map[inv_var == 0] = np.nan
            sigma[inv_var == 0] = np.nan

            # Saving and reshaping map data to map object
            map = map.reshape(
                20,
                mapdata["n_ra"],
                mapdata["n_dec"],
                mapdata["n_sidebands"],
                mapdata["n_channels"],
            )

            sigma = sigma.reshape(
                20,
                mapdata["n_ra"],
                mapdata["n_dec"],
                mapdata["n_sidebands"],
                mapdata["n_channels"],
            )

            # Changing axis order to old standard
            mapdata[f"{map_key}"] = map.transpose(0, 3, 4, 1, 2)
            mapdata[f"{sigma_wn_key}"] = sigma.transpose(0, 3, 4, 1, 2)

            # Deleting numerator and denominator from map object
            del mapdata[numerator_key]
            del mapdata[denominator_key]
        
            mapdata[f"{map_saddlebag_key}"] = np.zeros((mapdata.saddlebag_feeds.shape[0], *mapdata["map_coadd"].shape))
            mapdata[f"{sigma_wn_saddlebag_key}"] = np.zeros((mapdata.saddlebag_feeds.shape[0],  *mapdata["sigma_wn_coadd"].shape))
            
            if self.params.make_nhit:
                mapdata[f"{hit_saddlebag_key}"] = np.zeros((mapdata.saddlebag_feeds.shape[0], *mapdata["nhit_coadd"].shape), dtype = np.int32)

            for i in range(mapdata.saddlebag_feeds.shape[0]):
                # Current saddle bag feed indices
                feeds_in_saddlebag = mapdata.saddlebag_feeds[i] - 1
                
                weights = 1 / mapdata[f"{sigma_wn_key}"][feeds_in_saddlebag, ...] ** 2 
                data = mapdata[f"{map_key}"][feeds_in_saddlebag, ...] * weights
                inv_var = np.nansum(weights, axis = 0)

                mapdata[f"{map_saddlebag_key}"][i] = np.nansum(data, axis = 0) / inv_var
                mapdata[f"{sigma_wn_saddlebag_key}"][i] = 1 / np.sqrt(inv_var)
                if self.params.make_nhit:
                    mapdata[f"{hit_saddlebag_key}"][i] = np.sum( mapdata[f"{hit_key}"][feeds_in_saddlebag, ...], axis = 0)
            
            where = ~np.isfinite(mapdata[f"{sigma_wn_saddlebag_key}"])
            mapdata[f"{sigma_wn_saddlebag_key}"][where] = np.nan
            


    def mask_map(self, mapdata: COmap, numerator_key: str, denominator_key) -> None:
        """Method that takes in a map object and masks out the high noise regions. The masking is doing the following;
        
        for any map dataset per feed {map, rms and nhit} do
            1) coadd the rms over frequencies 
            2) compute the arithmetic mean of bottom-100 sigma_wn
            3) mask all map datasets where sigma_wn is below average bottom-100 channel-coadded sigma_wn per channel times self.params.t2m_rms_mask_factor.
        repeat for all feed and split maps
        done 

        Args:
            mapdata (COmap): Map object that contains map data to apply sigma_wn mask to.
            map_key (str): Key to map dataset to mask.
        """

        
        

        inv_var = mapdata[denominator_key]
        
        # Make keys for rms and nhit datasets that corresponds to map dataset
        nhit_key = re.sub(r"numerator_map", "nhit", numerator_key)
        
        feed_noise_freq_coadded = inv_var.copy()
        mask = ~np.isfinite(feed_noise_freq_coadded)
        feed_noise_freq_coadded[mask] = 0

        feed_noise_freq_coadded = 1 / np.sqrt(np.sum(feed_noise_freq_coadded, axis = 3))


        nfeed, nra, ndec, _ = mapdata[denominator_key].shape


        sorted_rms = feed_noise_freq_coadded.reshape(nfeed, nra * ndec)

        bottom100_idx = np.argpartition(sorted_rms, 100, axis=-1)[..., :100]
        bottom100 = np.take_along_axis(sorted_rms, bottom100_idx, axis=-1)

        mean_bottom100_rms = np.nanmean(bottom100, axis=-1) 

        noise_lim = self.params.t2m_rms_mask_factor * mean_bottom100_rms 

        mask = feed_noise_freq_coadded[:, :, :, None] * np.ones_like(mapdata[denominator_key]) > noise_lim[:, None, None, None] 

        mapdata[numerator_key][mask] = 0

        if self.params.make_nhit:
            mapdata[nhit_key] = mapdata[nhit_key].astype(np.float32)
            mapdata[nhit_key][mask] = 0

        mapdata[denominator_key][mask] = 0


        # Masking all channels that have a factor self.params.t2m_rms_mask_factor larger aritmetic average noise per channel than the totally arithmetically averaged sigma_wn.

        sigma = 1 / np.sqrt(inv_var)
        sigma[~np.isfinite(sigma)] = np.nan
        
        average_noise = np.nanmean(sigma)
        
        average_noise_per_channel = np.nanmean(sigma, axis = (1, 2))
        
        mask = average_noise_per_channel > (average_noise * self.params.t2m_rms_mask_factor)

        feeds_to_mask, channels_to_mask = np.where(mask)

        mapdata[numerator_key][feeds_to_mask, :, :, channels_to_mask] = 0
        mapdata[denominator_key][feeds_to_mask, :, :, channels_to_mask] = 0
        if self.params.make_nhit:
            mapdata[nhit_key][feeds_to_mask, :, :, channels_to_mask] = 0

    def get_pointing_matrix(
        self,
        l2data: L2file,
        mapdata: COmap,
    ) -> None:
        """Method which converts the {RA, DEC} pointing of the
        level 2 file into a pixel-feed index for later binning the maps.

        Args:
            l2data (L2file): Input level 2 file object for which
            to compute the pixel-feed index.
            mapdata (COmap): Map object from which to get grid parameters.
        """

        # Get pointing from level 2 object
        if self.params.horizontal:
            pointing = l2data["point_tel"]
        else:
            pointing = l2data["point_cel"]

        ra = pointing[:, :, 0].astype(np.float64)
        dec = pointing[:, :, 1].astype(np.float64)

        # # Get WCS grid parameters from map
        # DRA = mapdata["wcs"]["CDELT1"].astype(np.float64)
        # DDEC = mapdata["wcs"]["CDELT2"].astype(np.float64)

        # CRVAL_RA = mapdata["wcs"]["CRVAL1"].astype(np.float64)
        # CRVAL_DEC = mapdata["wcs"]["CRVAL2"].astype(np.float64)

        # CRPIX_RA = mapdata["wcs"]["CRPIX1"].astype(np.float64)
        # CRPIX_DEC = mapdata["wcs"]["CRPIX2"].astype(np.float64)

        # # Define {RA, DEC} indecies
        # idx_ra_allfeed = (np.round((ra - CRVAL_RA) / DRA + (CRPIX_RA - 1))).astype(
        #     np.int32
        # )
        # idx_dec_allfeed = (np.round((dec - CRVAL_DEC) / DDEC + (CRPIX_DEC - 1))).astype(
        #     np.int32
        # )

        coords = np.array((dec, ra))
        coords = np.deg2rad(coords)
        idx_dec_allfeed, idx_ra_allfeed = utils.nint(mapdata.standard_geometry.sky2pix(coords)).astype(np.int32)

        l2data["pointing_ra_index"] = idx_ra_allfeed
        l2data["pointing_dec_index"] = idx_dec_allfeed

    def get_temporal_mask(self, l2data: L2file, numerator_key: str) -> npt.NDArray:
        """Method to use to extend defualt level 2 file mask with for example
        a directional scan mask or to cut the beginning and end of scans.

        NOTE: This function is not yet finished and thus under construction

        Raises:
            ValueError: When provided percentile is outside allowed range.

        Returns:
            npt.NDArray: Additonal temporal mask array.
        """
        temporal_mask = l2data["mask_temporal"].copy()

        if "azdr0" in numerator_key:
            az = l2data["point_tel"][0,:,0]  # All pixels have same pointing behavior, so we simply use feed 1.
            temporal_mask = np.logical_and(temporal_mask, np.gradient(az)[:, None] > 0)

        elif "azdr1" in numerator_key:
            az = l2data["point_tel"][0,:,0]  # All pixels have same pointing behavior, so we simply use feed 1.
            temporal_mask = np.logical_and(temporal_mask, np.gradient(az)[:, None] < 0)
        
        if "azmd0" in numerator_key:
            az = l2data["point_tel"][0,:,0]
            az_median = np.median(az)
            temporal_mask = np.logical_and(temporal_mask, (az > az_median)[:, None])
        elif "azmd1" in numerator_key:
            az = l2data["point_tel"][0,:,0]
            az_median = np.median(az)
            temporal_mask = np.logical_and(temporal_mask, (az > az_median)[:, None])
            
        if "schf0" in numerator_key:
            Ntod = l2data["point_tel"][0,:,0][()].shape[-1]
            temporal_mask[Ntod//2:,:] = False
        elif "schf1" in numerator_key:
            Ntod = l2data["point_tel"][0,:,0][()].shape[-1]
            temporal_mask[:Ntod//2,:] = False

        if "azdi0" in numerator_key:
            az = l2data["point_tel"][0,:,0]
            az_median = np.median(az)
            az_dist_from_median = np.abs(az - az_median)
            az_median_dist_from_median = np.median(az_dist_from_median)
            temporal_mask[az_median_dist_from_median < az_dist_from_median, :] = False
        elif "azdi1" in numerator_key:
            az = l2data["point_tel"][0,:,0]
            az_median = np.median(az)
            az_dist_from_median = np.abs(az - az_median)
            az_median_dist_from_median = np.median(az_dist_from_median)
            temporal_mask[az_median_dist_from_median > az_dist_from_median, :] = False

        return temporal_mask

    def bin_map(
        self,
        mapdata: COmap,
        l2data: L2file,
    ):
        """Method that takes in a level 2 file object and binned its TOD up into feed maps inside COmap object

        Args:
            mapdata (COmap): Map object to fill with binned maps.
            l2data (L2file): Level 2 file object to bin up.
        """

        # Defining pointers for arrays to send to C++ modules
        float32_array4 = np.ctypeslib.ndpointer(
            dtype=ctypes.c_float, ndim=4, flags="contiguous"
        )  # 4D array 32-bit float pointer object.

        float32_array3 = np.ctypeslib.ndpointer(
            dtype=ctypes.c_float, ndim=3, flags="contiguous"
        )  # 4D array 32-bit float pointer object.

        float32_array2 = np.ctypeslib.ndpointer(
            dtype=ctypes.c_float, ndim=2, flags="contiguous"
        )  # 4D array 32-bit float pointer object.

        int32_array2 = np.ctypeslib.ndpointer(
            dtype=ctypes.c_int, ndim=2, flags="contiguous"
        )  # 4D array 32-bit integer pointer object.
        int64_array2 = np.ctypeslib.ndpointer(
            dtype=ctypes.c_long, ndim=2, flags="contiguous"
        )  # 4D array 32-bit integer pointer object.

        bool_array2 = np.ctypeslib.ndpointer(
            dtype=ctypes.c_bool, ndim=2, flags="contiguous"
        )  # 4D array 32-bit integer pointer object.
        scan_idx = np.where(self.splitdata["scan_list"] == l2data.id)[0][0]
        
        # If no hit map is needed:
        if self.params.make_nhit:

            int32_array4 = np.ctypeslib.ndpointer(
                dtype=ctypes.c_int, ndim=4, flags="contiguous"
            )  # 4D array 32-bit integer pointer object.
            # self.mapbinner.bin_map.argtypes = [

            # Specifying input types and shapes for C++ shared library
            self.mapbinner.bin_nhit_and_map.argtypes = [
                float32_array3,  # tod
                float32_array2,  # sigma
                int32_array2,  # freqmask
                int32_array2,  # idx_ra_pix
                int32_array2,  # idx_dec_pix
                bool_array2,  # temporal_mask
                int32_array4,  # hit map
                float32_array4,  # numerator map
                float32_array4,  # denominator map
                ctypes.c_int,  # nfreq
                ctypes.c_int,  # nsamp
                ctypes.c_int,  # nside_ra
                ctypes.c_int,  # nside_dec
                ctypes.c_int,  # nfeed
                ctypes.c_int,  # nthread
                ctypes.c_int,  # scanid
            ]

            NFEED, NSAMP, NFREQ = l2data["tod"].shape

            # For all split maps to bin:
            for numerator_key in self.maps_to_bin:
                # Generating map keys
                denominator_key = re.sub(r"numerator", "denominator", numerator_key)
                hit_key = re.sub(r"numerator_map", "nhit", numerator_key)

                # Generating feed-frequency mask for split maps
                if numerator_key == "numerator_map":
                    freqmask = l2data["freqmask"].copy()
                else:
                    freqmask = l2data["freqmask"].copy()
                    split_key = re.sub(r"/numerator_map_", "", numerator_key)

                    split_list = self.splitdata["jk_list"][scan_idx, ...]

                    split_feed_mask, split_sideband_mask = np.where(
                        np.isin(
                            split_list, self.split_key_mapping[split_key], invert=True
                        )
                    )

                    NCHANNEL = mapdata["n_channels"]
                    NSB = mapdata["n_sidebands"]

                    freqmask = (
                        l2data["freqmask"]
                        .reshape(
                            NFEED,
                            NSB,
                            NCHANNEL,
                        )
                        .copy()
                    )

                    freqmask[split_feed_mask, split_sideband_mask, :] = 0

                    if np.all(freqmask == 0):
                        continue

                    freqmask = freqmask.reshape(NFEED, NFREQ)
                

                temporal_mask = self.get_temporal_mask(l2data, numerator_key)

                # Calling C++ shared library to bin up maps
                self.mapbinner.bin_nhit_and_map(
                    l2data["tod"],
                    l2data["inv_var"],
                    freqmask,
                    l2data["pointing_ra_index"],
                    l2data["pointing_dec_index"],
                    temporal_mask,
                    mapdata[hit_key],
                    mapdata[numerator_key],
                    mapdata[denominator_key],
                    NFREQ,
                    NFEED,
                    NSAMP,
                    mapdata["n_ra"],
                    mapdata["n_dec"],
                    self.OMP_NUM_THREADS,
                    l2data.id,
                )
            
        else:
            # If we want to make maps and hit maps

            # Specifying input types and shapes for C++ shared library
            self.mapbinner.bin_map.argtypes = [
                float32_array3,  # tod
                float32_array2,  # sigma
                int32_array2,  # freqmask
                int32_array2,  # idx_ra_pix
                int32_array2,  # idx_dec_pix
                bool_array2,  # temporal_mask
                float32_array4,  # numerator map
                float32_array4,  # denominator map
                ctypes.c_int,  # nfreq
                ctypes.c_int,  # nsamp
                ctypes.c_int,  # nside_ra
                ctypes.c_int,  # nside_dec
                ctypes.c_int,  # nfeed
                ctypes.c_int,  # nthread
            ]

            NFEED, NSAMP, NFREQ = l2data["tod"].shape

            for numerator_key in self.maps_to_bin:
                # Generating map keys
                denominator_key = re.sub(r"numerator", "denominator", numerator_key)

                # Generating feed-frequency mask for split maps
                if numerator_key == "numerator_map":
                    freqmask = l2data["freqmask"].copy()
                else:
                    freqmask = l2data["freqmask"].copy()
                    split_key = re.sub(r"/numerator_map_", "", numerator_key)

                    split_list = self.splitdata["jk_list"][scan_idx, ...]
                    split_feed_mask, split_sideband_mask = np.where(
                        np.isin(
                            split_list, self.split_key_mapping[split_key], invert=True
                        )
                    )
                    NCHANNEL = mapdata["n_channels"]
                    NSB = mapdata["n_sidebands"]

                    freqmask = (
                        l2data["freqmask"]
                        .reshape(
                            NFEED,
                            NSB,
                            NCHANNEL,
                        )
                        .copy()
                    )

                    freqmask[split_feed_mask, split_sideband_mask, :] = 0

                    if np.all(freqmask == 0):
                        continue

                    freqmask = freqmask.reshape(NFEED, NFREQ)

                temporal_mask = self.get_temporal_mask(l2data, numerator_key)

                # Calling C++ shared library to bin up maps
                self.mapbinner.bin_map(
                    l2data["tod"],
                    l2data["inv_var"],
                    freqmask,
                    l2data["pointing_ra_index"],
                    l2data["pointing_dec_index"],
                    temporal_mask,
                    mapdata[numerator_key],
                    mapdata[denominator_key],
                    NFREQ,
                    NFEED,
                    NSAMP,
                    mapdata["n_ra"],
                    mapdata["n_dec"],
                    self.OMP_NUM_THREADS,
                )

    def populate_simulation_cube(self, mapdata: COmap):
        """Method that generates a map file object from a simulation cube object defined by
        self.params.signal_path

        Args:
            mapdata (COmap): Map file object to fill with data from simulation cube.
        """

        # Marking map object as simulation
        mapdata["is_sim"] = True
        mapdata["is_simcube"] = True

        from l2_simulations import SimCube

        NFEED, NSIDEBAND, _, NDEC, NRA = mapdata["map"].shape

        # Simulation cube object to fill map object with
        signal_path = self.params.signal_path
        if signal_path is None:
            signal_path = os.path.join(self.params.sim_output_dir, self.params.sim_map_output_file_name)

        simdata = SimCube(signal_path)

        # Reading simulation cube data from file
        simdata.read()

        # Defining simulation cube geometry using standard geometies and boost signal
        simdata.prepare_geometry(self.fieldname, self.params.boost_factor)

        NCHANNEL_sim = simdata.simdata.shape[1]

        # Rotate and bin up simulation cube data to target map geometry
        signal = simdata.bin_cube2field_geometry(mapdata.standard_geometry)

        # Adapting cube to map frequency resolution
        signal = signal.reshape(
            NSIDEBAND,
            self.params.decimation_freqs,
            NCHANNEL_sim // self.params.decimation_freqs,
            NDEC,
            NRA,
        ).mean((2))

        # Asigning map datasets
        mask = np.isfinite(mapdata["map_coadd"])
        mapdata["map_coadd"][mask] = signal[mask]

        # Filling feed maps with simulation data
        for i in range(NFEED):
            mask = np.isfinite(mapdata["map"][i, ...])
            mapdata["map"][i][mask] = signal[mask]

        # Fill in split maps with simulation data
        if self.perform_splits:
            for key in mapdata.keys:
                if "/" in key and ("map" in key and "saddlebag" not in key):
                    for i in range(NFEED):
                        mask = np.isfinite(mapdata[key][i, ...])
                        mapdata[key][i][mask] = signal[mask]
                if "/" in key and ("map" in key and "saddlebag" in key):
                    for i in range(4):
                        mask = np.isfinite(mapdata[key][i, ...])
                        mapdata[key][i][mask] = signal[mask]

            # Providing name of primary splits to make group names
            mapdata.write_map(primary_splits=self.primary_splits, params=self.params)
        else:
            mapdata.write_map(params=self.params)


def main():
    if "OMP_NUM_THREADS" in os.environ:
        omp_num_threads = int(os.environ["OMP_NUM_THREADS"])
    else:
        omp_num_threads = 1

    tod2comap = Mapmaker(omp_num_threads=omp_num_threads)
    tod2comap.parse_accept_data()

    if tod2comap.params.directional:
        tod2comap.which_az_direction = 1
        base_name = tod2comap.params.map_name
        tod2comap.params.map_name = base_name + "_positive"
        tod2comap.run()

        tod2comap.which_az_direction = -1
        tod2comap.params.map_name = base_name + "_negative"
        tod2comap.run()
    if tod2comap.params.temporal_chunking > 0:
        obsids = np.sort(tod2comap.obsid_list)
        n_obsids = tod2comap.n_obsids
        tod2comap.params.obsid_start = obsids[0]
        tod2comap.params.obsid_stop = obsids[tod2comap.params.temporal_chunking]

        base_name = tod2comap.params.map_name
        tod2comap.params.map_dir = os.path.join(
            tod2comap.params.map_dir, "temporal_chunking"
        )
        for chunk, i in enumerate(
            range(
                tod2comap.params.temporal_chunking + 1,
                n_obsids,
                tod2comap.params.temporal_chunking,
            )
        ):
            tod2comap.params.obsid_stop = obsids[i]
            tod2comap.read_runlist()
            tod2comap.params.map_name = base_name + f"_t{chunk}"
            tod2comap.run()
            tod2comap.params.obsid_start = obsids[i + 1]
    else:
        tod2comap.run()

        if tod2comap.params.bin_signal_tod:
            tod2comap.use_signal_tod = True
            tod2comap.params.populate_cube = False
            tod2comap.params.map_name += "_signal_tod" 
            tod2comap.run()

if __name__ == "__main__":
    main()

    # TODO:
    # * Fix documentations
