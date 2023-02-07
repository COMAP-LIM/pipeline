import numpy as np
import numpy.typing as ntyping
import ctypes
import time
import h5py
import re
from mpi4py import MPI
import datetime
from copy import copy, deepcopy

from COmap import COmap
from L2file import L2file

import warnings

# Ignore RuntimeWarning
warnings.filterwarnings("ignore", category=RuntimeWarning)

import os
import sys
import git

current = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current)

sys.path.append(parent_directory)
sys.path.append(os.path.join(parent_directory, "simpipeline/"))


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
        if self.rank == 0:
            print(
                f"Creating runlist in specified obsid range [{self.params.obsid_start}, {self.params.obsid_stop}]"
            )
            print(f"Runlist file: {self.params.runlist}\n")

        # Create list of already processed scanids.
        existing_scans = []
        for dir in os.listdir(self.params.level2_dir):
            if os.path.isdir(os.path.join(self.params.level2_dir, dir)):
                for file in os.listdir(os.path.join(self.params.level2_dir, dir)):
                    if file[-3:] == ".h5" or file[-4:] == ".hd5":
                        if (
                            len(file) == 16 or len(file) == 17
                        ):  # In order to not catch the intermediate debug files.
                            existing_scans.append(int(file.split(".")[0].split("_")[1]))

        with open(self.params.runlist) as my_file:
            lines = [line.split() for line in my_file]
        i = 0
        runlist = []
        n_fields = int(lines[i][0])
        i = i + 1

        self.obsid_list = []
        for i_field in range(n_fields):
            runlist = []
            n_scans_tot = 0
            n_scans_outside_range = 0
            n_scans_already_processed = 0
            fieldname = lines[i][0]
            n_obsids = int(lines[i][1])
            i = i + 1
            for j in range(n_obsids):
                obsid_int = int(lines[i][0])
                self.obsid_list.append(obsid_int)
                obsid = "0" + lines[i][0]
                n_scans = int(lines[i][3])

                if (
                    obsid_int < self.params.obsid_start
                    or obsid_int > self.params.obsid_stop
                ):
                    i = i + n_scans + 1
                    continue  # Skip this loop iteration if obsid is outside parameter file specified obsid range.
                l1_filename = lines[i][-1]
                l1_filename = l1_filename.strip(
                    "/"
                )  # The leading "/" will stop os.path.join from joining the filenames.
                l1_filename = os.path.join(self.params.level1_dir, l1_filename)
                for k in range(n_scans):
                    scantype = int(float(lines[i + k + 1][3]))
                    if scantype != 8192:
                        if scantype != 32 and self.rank == 0:
                            scan_warning = (
                                "\n"
                                + "=" * 50
                                + "\nYou are running a runlist with scanning strategies different than CES!"
                                + " \nLissajous and circular scans are deprecated!\n"
                                + "=" * 50
                            )
                            warnings.warn(scan_warning, category=FutureWarning)

                        n_scans_tot += 1
                        if (
                            self.params.obsid_start
                            <= int(obsid)
                            <= self.params.obsid_stop
                        ):
                            scanid = int(lines[i + k + 1][0])

                            l2_filename = f"{fieldname}_{scanid:09}.h5"
                            l2_filename = os.path.join(
                                self.params.level2_dir,
                                fieldname,
                                l2_filename,
                            )

                            mjd_start = float(lines[i + k + 1][1])
                            mjd_stop = float(lines[i + k + 1][2])
                            runlist.append(
                                [
                                    scanid,
                                    mjd_start,
                                    mjd_stop,
                                    scantype,
                                    fieldname,
                                    l1_filename,
                                    l2_filename,
                                ]
                            )
                        else:
                            n_scans_outside_range += 1
                i = i + n_scans + 1

        if self.rank == 0:
            print(f"Field name:                 {fieldname}")
            print(f"Obsids in runlist file:     {n_obsids}")
            print(f"Scans in runlist file:      {n_scans_tot}")
            print(f"Scans included in run:      {len(runlist)}")
            print(f"Scans outside obsid range:  {n_scans_outside_range}")

        self.fieldname = fieldname
        self.runlist = runlist
        self.n_obsids = n_obsids

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
                N_primary_splits = 0
                N_secondary_splits = 0

                primary_splits = []
                secondary_splits = []

                # Read split definition file
                for line in split_def_file.readlines()[2:]:
                    split_name, split_type = line.split()[:2]
                    split_types_dict[split_name] = int(split_type)

                    # Count number of primary and secondary splits
                    if int(split_type) == 2:
                        N_primary_splits += 1
                        primary_splits.append(split_name)
                    elif int(split_type) == 3:
                        N_secondary_splits += 1
                        secondary_splits.append(split_name)

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
                    # Begin mapping name with primary key
                    key = f"{primary}"
                    key += f"{bits[i, N_secondary_splits + p]:d}"

                    # Add secondary keys to primary key
                    for s, secondary in enumerate(secondary_splits[::-1]):
                        key += f"{secondary}"
                        key += f"{bits[i,N_secondary_splits - 1 - s]:d}"

                    # Save mapping in dictionary
                    if key not in split_key_mapping.keys():
                        split_key_mapping[key] = np.array([number])
                    else:
                        split_key_mapping[key] = np.append(
                            split_key_mapping[key], [number]
                        )

            self.split_key_mapping = split_key_mapping
            split_keys = list(self.split_key_mapping.keys())

            # Generating list of map keys that are to be binned up into maps
            # including the "non-split" map
            self.maps_to_bin += [
                f"/{self.maps_to_bin[0]}_{split_keys[i]}"
                for i in range(len(split_keys))
            ]

            self.primary_splits = primary_splits

    def run(self):
        """Method running through the provided runlist and binning up maps."""
        self.comm.barrier()
        if self.rank == 0:
            start_time = time.perf_counter()
        # File name of full coadded output map
        full_map_name = f"{self.fieldname}_{self.params.map_name}.h5"
        full_map_name = os.path.join(self.params.map_dir, full_map_name)

        # Define and initialize empty map object to acumulate data
        full_map = COmap(full_map_name)
        full_map.init_emtpy_map(
            self.fieldname,
            self.params.decimation_freqs,
            self.params.res_factor,
            self.params.make_no_nhit,
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

        for i, scan in enumerate(self.runlist):

            scanid = scan[0]

            if i % self.Nranks == self.rank:
                # Cycle to next scan
                scan_idx = np.where(self.splitdata["scan_list"] == scanid)[0][0]
                if (
                    np.all(~self.splitdata["accept_list"][scan_idx])
                    and not self.params.override_accept
                ):
                    # Print in red forground color
                    print(
                        f"\033[91m Rejected scan {scanid} @ rank {self.rank} \033[00m"
                    )
                    rejection_number += 1
                    continue

                if self.params.drop_first_scans and str(scanid)[-2:] == "02":
                    # Print in red forground color
                    print(
                        f"\033[91m Rejected scan {scanid} @ rank {self.rank} \033[00m"
                    )
                    rejection_number += 1
                    continue

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

            print("=" * 80)
            print(f"Average timing over {len(self.runlist)} scans:")
            print(
                f"Number of rejected scans {rejection_number_buffer[0]} of {len(self.runlist)}"
            )
            print("=" * 80)
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

            print("=" * 80)

        # Perform MPI reduce on map datasets
        self.reduce_maps(full_map)

        if self.rank == 0:
            # self.postprocess_map(full_map, l2data)
            self.postprocess_map(full_map)
            if self.perform_splits:
                # Providing name of primary splits to make group names
                full_map.write_map(
                    primary_splits=self.primary_splits,
                    params=self.params,
                    save_hdf5=(not self.params.no_hdf5),
                    save_fits=(not self.params.no_fits),
                )
            else:
                full_map.write_map(
                    params=self.params,
                    save_hdf5=(not self.params.no_hdf5),
                    save_fits=(not self.params.no_fits),
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

                if self.params.make_no_nhit:
                    # Generate hit map keys
                    hit_key = re.sub(r"numerator_map", "nhit", numerator_key)
                    nhit_buffer = np.zeros_like(mapdata[hit_key])
            else:
                numerator_buffer = None
                denominator_buffer = None
                if self.params.make_no_nhit:
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

            if self.params.make_no_nhit:
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

        temporal_mask = l2data["mask_temporal"]

        if self.params.temporal_mask:
            temporal_mask *= self.get_temporal_mask(l2data)[None, :]

        feeds_to_mask, times_to_mask = np.where(temporal_mask)
        l2data["tod"] = l2data["tod"][feeds_to_mask, :, :, times_to_mask]
        l2data["point_tel"] = l2data["point_tel"][feeds_to_mask, times_to_mask, :]
        l2data["point_cel"] = l2data["point_cel"][feeds_to_mask, times_to_mask, :]

        _, NSB, NFREQ, NSAMP = l2data["tod"].shape

        # Defining empty buffers
        tod = np.zeros((20, NSB, NFREQ, NSAMP), dtype=np.float32)
        sigma0 = np.zeros((20, NSB, NFREQ), dtype=np.float32)
        freqmask = np.zeros((20, NSB, NFREQ), dtype=np.int32)
        pointing = np.zeros((20, NSAMP, 2), dtype=np.float32)

        # Get index to pixel mapping
        try:
            pixels = l2data["pixels"] - 1
        except KeyError:
            pixels = l2data["feeds"] - 1

        # Sort pixels to correct buffer position
        tod[pixels, ...] = l2data["tod"]
        sigma0[pixels, ...] = l2data["sigma0"]
        freqmask[pixels, ...] = l2data["freqmask"]
        pointing[pixels, ...] = l2data["point_cel"][..., :2]

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

            if self.params.make_no_nhit:
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

        # Get WCS grid parameters from map
        DRA = mapdata["wcs"]["CDELT1"].astype(np.float64)
        DDEC = mapdata["wcs"]["CDELT2"].astype(np.float64)

        CRVAL_RA = mapdata["wcs"]["CRVAL1"].astype(np.float64)
        CRVAL_DEC = mapdata["wcs"]["CRVAL2"].astype(np.float64)

        CRPIX_RA = mapdata["wcs"]["CRPIX1"].astype(np.float64)
        CRPIX_DEC = mapdata["wcs"]["CRPIX2"].astype(np.float64)

        # Define {RA, DEC} indecies
        idx_ra_allfeed = (np.round((ra - CRVAL_RA) / DRA + (CRPIX_RA - 1))).astype(
            np.int32
        )
        idx_dec_allfeed = (np.round((dec - CRVAL_DEC) / DDEC + (CRPIX_DEC - 1))).astype(
            np.int32
        )

        l2data["pointing_ra_index"] = idx_ra_allfeed
        l2data["pointing_dec_index"] = idx_dec_allfeed

    def get_temporal_mask(self, l2data: L2file):
        az_percentile = self.params.az_mask_percentile  # Between 0 and 100

        if az_percentile > 0.0:

            el_cut = self.params.el_mask_cut  # Degrees

            if az_percentile < 0 or az_percentile > 100:
                raise ValueError(
                    "Azimuth masking percentile must be between 0 and 100."
                )

            # Since the time of turn around should be the same for all detectors,
            # only the azimuth of feed index 0 is used.
            az = l2data["point_tel"][0, :, 0]
            el = l2data["point_tel"][0, :, 1]

            # Define aximuth and elevation limits
            az_lims = [
                np.percentile(az, 100 - az_percentile),
                np.percentile(az, az_percentile),
            ]

            el_lims = [
                np.median(el) - el_cut,
                np.median(el) + el_cut,
            ]

            az_mask = np.logical_and(az > az_lims[0], az < az_lims[1])
            el_mask = np.logical_and(el > el_lims[0], el < el_lims[1])

            temporal_mask = np.logical_and(az_mask, el_mask)

        if self.params.directional:
            az_grad = np.gradient(l2data["point_tel"][0, :, 0])
            temporal_mask = np.sign(az_grad) == self.which_az_direction

        # Manually removing 0.5 seconds of the data at the scan edges
        # to avoid potential repointing leakage
        sample_time = 3600 * 24 * (l2data["time"][1] - l2data["time"][0])
        Ncut = np.round(30 / sample_time).astype(np.int32)

        temporal_mask[:Ncut] = False
        temporal_mask[-Ncut:] = False

        return temporal_mask

    def bin_map(
        self,
        mapdata: COmap,
        l2data: L2file,
    ):

        float32_array4 = np.ctypeslib.ndpointer(
            dtype=ctypes.c_float, ndim=4, flags="contiguous"
        )  # 4D array 32-bit float pointer object.

        float32_array3 = np.ctypeslib.ndpointer(
            dtype=ctypes.c_float, ndim=3, flags="contiguous"
        )  # 4D array 32-bit float pointer object.

        float32_array2 = np.ctypeslib.ndpointer(
            dtype=ctypes.c_float, ndim=2, flags="contiguous"
        )  # 4D array 32-bit float pointer object.

        # float64_array3 = np.ctypeslib.ndpointer(
        #     dtype=ctypes.c_double, ndim=3, flags="contiguous"
        # )  # 4D array 32-bit float pointer object.

        # float64_array2 = np.ctypeslib.ndpointer(
        #     dtype=ctypes.c_double, ndim=2, flags="contiguous"
        # )  # 4D array 32-bit float pointer object.

        int32_array2 = np.ctypeslib.ndpointer(
            dtype=ctypes.c_int, ndim=2, flags="contiguous"
        )  # 4D array 32-bit integer pointer object.

        scan_idx = np.where(self.splitdata["scan_list"] == l2data.id)[0][0]

        if self.params.make_no_nhit:
            int32_array4 = np.ctypeslib.ndpointer(
                dtype=ctypes.c_int, ndim=4, flags="contiguous"
            )  # 4D array 32-bit integer pointer object.
            # self.mapbinner.bin_map.argtypes = [
            self.mapbinner.bin_nhit_and_map.argtypes = [
                float32_array3,  # tod
                float32_array2,  # sigma
                int32_array2,  # freqmask
                int32_array2,  # idx_ra_pix
                int32_array2,  # idx_dec_pix
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

            for numerator_key in self.maps_to_bin:
                # Generating map keys
                denominator_key = re.sub(r"numerator", "denominator", numerator_key)
                hit_key = re.sub(r"numerator_map", "nhit", numerator_key)

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

                    # print(np.unique(split_list))
                    # print(split_list)
                    # print(~np.isin(split_list, self.split_key_mapping[split_key]))

                    # print(
                    #     "hallo",
                    #     numerator_key,
                    #     np.all(freqmask == 0),
                    #     np.all(mapdata[hit_key] == 0),
                    #     freqmask.dtype,
                    # )
                    # sys.exit()

                self.mapbinner.bin_nhit_and_map(
                    l2data["tod"],
                    l2data["inv_var"],
                    freqmask,
                    l2data["pointing_ra_index"],
                    l2data["pointing_dec_index"],
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
            self.mapbinner.bin_map.argtypes = [
                float32_array3,  # tod
                float32_array2,  # sigma
                int32_array2,  # freqmask
                int32_array2,  # idx_ra_pix
                int32_array2,  # idx_dec_pix
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

                self.mapbinner.bin_map(
                    l2data["tod"],
                    l2data["inv_var"],
                    freqmask,
                    l2data["pointing_ra_index"],
                    l2data["pointing_dec_index"],
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

        from l2_simulations import SimCube

        NFEED, NSIDEBAND, NCHANNEL, NDEC, NRA = mapdata["map"].shape

        simdata = SimCube(self.params.signal_path)
        # Reading simulation cube data from file
        simdata.read()

        # Defining simulation cube geometry using standard geometies and boost signal
        simdata.prepare_geometry(self.fieldname, self.params.boost_factor)

        # Get pixel center meshgrid from target standard geometry
        dec_grid, ra_grid = np.rad2deg(mapdata.standard_geometry.posmap())

        # Euler rotaion of telescope pointing to equatorial origin
        ra_grid_rotated, dec_grid_rotated = simdata.rotate_pointing_to_equator(
            ra_grid.flatten(),
            dec_grid.flatten(),
        )

        simdata.interpolate_cube(self.fieldname)

        NCHANNEL_sim = simdata.simdata.shape[1]

        signal = np.zeros((NSIDEBAND, NCHANNEL_sim, NDEC, NRA))

        for sb in range(NSIDEBAND):
            for channel in range(NCHANNEL_sim):
                signal[sb, channel, :, :] = simdata.signal[sb][channel](
                    dec_grid_rotated, ra_grid_rotated, grid=False
                ).reshape(NDEC, NRA)

        # cube = {}
        # with h5py.File(self.params.signal_path, "r") as infile:
        #     cube["simulation"] = infile["simulation"][()]

        # signal = cube["simulation"]

        # _, NCHANNEL_sim, NDEC_sim, NRA_sim = signal.shape

        # Marking map object as simulation
        mapdata["is_sim"] = True
        mapdata["is_simcube"] = True

        # NDOWNSAMPLE_RA = NRA_sim // NRA
        # NDOWNSAMPLE_DEC = NDEC_sim // NDEC
        # NDOWNSAMPLE_CHANNEL = NCHANNEL_sim // NCHANNEL

        # # Adapting cube to map resolution
        # signal = signal.reshape(
        #     NSIDEBAND,
        #     NCHANNEL,
        #     NDOWNSAMPLE_CHANNEL,
        #     NDEC,
        #     NDOWNSAMPLE_DEC,
        #     NRA,
        #     NDOWNSAMPLE_RA,
        # ).mean((2, 4, 6))

        # Adapting cube to map frequency resolution
        signal = signal.reshape(
            NSIDEBAND,
            self.params.decimation_freqs,
            NCHANNEL_sim // self.params.decimation_freqs,
            NDEC,
            NRA,
        ).mean((2))

        # From muK to K and optional signal boost
        # signal *= 1e-6 * self.params.boost_factor

        # Asigning map datasets
        mask = np.isfinite(mapdata["map_coadd"])
        mapdata["map_coadd"][mask] = signal[mask]

        for i in range(NFEED):
            mask = np.isfinite(mapdata["map"][i, ...])
            mapdata["map"][i][mask] = signal[mask]

        if self.perform_splits:
            for key in mapdata.keys:
                if "multisplits" in key and "map" in key:
                    for i in range(NFEED):
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


if __name__ == "__main__":
    main()

    # TODO:
    # * Fix documentations
    # * Implement splits
    # * Save frequency edges and
    #   centers when Jonas
    #   fixes this in l2gen
    # * HP filter?
    #
