import numpy as np
import numpy.typing as ntyping
import ctypes
import time
import h5py
from mpi4py import MPI

from COmap import COmap
from L2file import L2file

import warnings

# Ignore RuntimeWarning
warnings.filterwarnings("ignore", category=RuntimeWarning)

import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current)

sys.path.append(parent_directory)


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

        # Define map grid parameters
        self.GRID_SIZE = self.params.grid_size

        self.FIELD_CENTER = self.params.field_center[self.fieldname]

        # Converting Delta RA to physical degrees
        self.DRA = self.params.grid_res[0] / np.abs(
            np.cos(np.radians(self.FIELD_CENTER[1]))
        )
        self.DDEC = self.params.grid_res[1]

        if not self.params.map_name:
            raise ValueError(
                "A map file name must be specified in parameter file or terminal."
            )

        # Define c-library used for binning maps
        mapbinner_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "../C_libs/tod2comap/mapbinner.so.1",
        )
        self.mapbinner = ctypes.cdll.LoadLibrary(mapbinner_path)

    def read_params(self):
        from l2gen_argparser import parser

        params = parser.parse_args()
        if not params.runlist:
            raise ValueError(
                "A runlist must be specified in parameter file or terminal."
            )
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
        for i_field in range(n_fields):
            runlist = []
            n_scans_tot = 0
            n_scans_outside_range = 0
            n_scans_already_processed = 0
            fieldname = lines[i][0]
            n_obsids = int(lines[i][1])
            i = i + 1
            for j in range(n_obsids):
                obsid = "0" + lines[i][0]
                n_scans = int(lines[i][3])
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

    def parse_accept_data(self):
        split_data_path = self.params.split_data
        scan_data_path = self.params.scan_data
        accept_dir = self.params.accept_dir

        if not split_data_path:
            message = "Please specify a split_data file name."
            raise ValueError(message)
        elif not scan_data_path:
            message = "Please specify a scan_data file name."
            raise ValueError(message)

        scan_data_path = os.path.join(accept_dir, scan_data_path)
        with h5py.File(scan_data_path, "r") as scanfile:
            self.scandata = {}
            for key, value in scanfile.items():
                self.scandata[key] = value[()]

        split_data_path = os.path.join(accept_dir, split_data_path)
        with h5py.File(split_data_path, "r") as splitfile:
            self.splitdata = {}
            for key, value in splitfile.items():
                self.splitdata[key] = value[()]

        self.perform_splits = self.params.split
        if self.perform_splits:

            split_names = self.splitdata["split_list"].astype(str)
            Nsplits = len(split_names)
            split_list = self.splitdata["jk_list"]
            _split_list = split_list.copy()

            # split_bits = np.zeros((Nsplits + 1,) + split_list.shape)

            # for i in range(Nsplits + 1):
            #    split_bits[i, ...] = _split_list >= 2**i
            #    _split_list -= 2**i

            unique_numbers = np.unique(split_list)
            unique_numbers = unique_numbers[unique_numbers != 0]
            bits = np.zeros((len(unique_numbers), Nsplits + 1))

            split_key_mapping = {}

            for i, number in enumerate(unique_numbers):
                # Convert number to bits
                bit_num = f"{number:0{Nsplits+1}b}"

                key = ""

                # Loop through all but last bit (accept/recect)
                for j in range(Nsplits):
                    bits[i, j] = int(bit_num[j])
                    key += str(split_names[j]) + bit_num[j] + "_"

                split_key_mapping[key] = number

        #         print(number, bits[i, :], bit_num, key)
        # print(split_key_mapping)

        ## Need to parse split_def file to get keys right. In particular "parent" split must be first
        # sys.exit()

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
            (self.GRID_SIZE, (self.DRA, self.DDEC), self.FIELD_CENTER),
            self.params.decimation_freqs,
            self.params.make_nhit,
        )
        self.RA_min = full_map.ra_min
        self.DEC_min = full_map.dec_min

        time_array = np.zeros(6)
        if self.rank == 0:
            time_buffer = time_array.copy()
        else:
            time_buffer = None

        for i, scan in enumerate(self.runlist):
            if i % self.Nranks == self.rank:
                print(f"Processing scan {scan[0]} @ rank {self.rank}")
                l2path = scan[-1]

                ti = time.perf_counter()

                l2data = L2file(path=l2path, id=scan[0])
                time_array[0] += time.perf_counter() - ti

                t0 = time.perf_counter()

                l2data.read_l2()
                time_array[1] += time.perf_counter() - t0

                t0 = time.perf_counter()

                self.preprocess_l2data(l2data)
                time_array[2] += time.perf_counter() - t0

                t0 = time.perf_counter()

                self.get_pointing_matrix(l2data)
                time_array[3] += time.perf_counter() - t0

                t0 = time.perf_counter()

                self.bin_map(full_map, l2data)
                time_array[4] += time.perf_counter() - t0

                time_array[5] += time.perf_counter() - ti

        self.comm.Reduce(
            [time_array, MPI.DOUBLE], [time_buffer, MPI.DOUBLE], op=MPI.SUM, root=0
        )

        if self.rank == 0:
            time_buffer /= len(self.runlist)

            print("=" * 80)
            print(f"Average timing over {len(self.runlist)} scans:")
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
            self.postprocess_map(full_map, l2data)
            full_map.write_map()
            finish_time = time.perf_counter()

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
        # Define buffer arrays
        if self.rank == 0:
            numerator_buffer = np.zeros_like(mapdata["numerator_map"])
            denominator_buffer = np.zeros_like(mapdata["denominator_map"])
            if self.params.make_nhit:
                nhit_buffer = np.zeros_like(mapdata["nhit"])
        else:
            numerator_buffer = None
            denominator_buffer = None
            if self.params.make_nhit:
                nhit_buffer = None

        # Perform MPI reduction
        self.comm.Reduce(
            [mapdata["numerator_map"], MPI.FLOAT],
            [numerator_buffer, MPI.FLOAT],
            op=MPI.SUM,
            root=0,
        )

        self.comm.Reduce(
            [mapdata["denominator_map"], MPI.FLOAT],
            [denominator_buffer, MPI.FLOAT],
            op=MPI.SUM,
            root=0,
        )

        if self.rank == 0:
            # Overwrite rank 0 datasets
            mapdata["numerator_map"] = numerator_buffer
            mapdata["denominator_map"] = denominator_buffer

        if self.params.make_nhit:
            self.comm.Reduce(
                [mapdata["nhit"], MPI.INT],
                [nhit_buffer, MPI.INT],
                op=MPI.SUM,
                root=0,
            )

            if self.rank == 0:
                mapdata["nhit"] = nhit_buffer

    def preprocess_l2data(
        self,
        l2data: L2file,
    ):
        """Method that brings the feed infomation in needed datasets
        to correct correct order for binning to run optimally.

        Args:
            l2data (L2file): Level 2 file object to perform preprocessing on.
        """

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
        pointing[pixels, ...] = l2data["point_cel"][..., :-1]
        freqs = l2data["nu"][0, ...]

        # Flip sideband 0 and 2
        tod[:, (0, 2), :, :] = tod[:, (0, 2), ::-1, :]
        sigma0[:, (0, 2), :] = sigma0[:, (0, 2), ::-1]
        freqmask[:, (0, 2), :] = freqmask[:, (0, 2), ::-1]
        freqs[(0, 2), :] = freqs[(0, 2), ::-1]

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
        # sigma0 = np.ascontiguousarray(sigma0, dtype=np.float32)

        # Pre-compute masking
        inv_var = 1 / sigma0**2  # inverse variance
        inv_var[freqmask == 0] = 0

        # Overwrite old data
        l2data["tod"] = tod
        l2data["inv_var"] = inv_var
        l2data["freqmask"] = freqmask

        # Found that transposing the pixels here makes the binning
        # faster for some reason
        l2data["point_cel"] = np.ascontiguousarray(pointing.transpose(1, 0, 2))
        l2data["nu"] = freqs

    def postprocess_map(self, mapdata: COmap, l2data: L2file) -> None:
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

        inv_var = mapdata["denominator_map"]

        # Full map is made by computing map = sum(TOD * inv_var) / sum(inv_var)
        map = mapdata["numerator_map"] / inv_var

        # White noise level map
        sigma = 1 / np.sqrt(inv_var)

        # Mask all non-hit regions for feed-coaddition
        map[inv_var == 0] = 0

        # Computing feed-coadded map and white noise map
        map_coadd = np.sum(map * inv_var, axis=0)
        sigma_coadd = np.sum(inv_var, axis=0)

        map_coadd /= sigma_coadd
        sigma_coadd = 1 / np.sqrt(sigma_coadd)

        if self.params.make_nhit:
            # Computing coadded hit map
            nhit_coadd = np.sum(mapdata["nhit"], axis=0)

            mapdata["nhit_coadd"] = nhit_coadd.reshape(
                mapdata["n_ra"],
                mapdata["n_dec"],
                mapdata["n_sidebands"],
                mapdata["n_channels"],
            )

        # Saving coadded and reshaped data to map object
        mapdata["map_coadd"] = map_coadd.reshape(
            mapdata["n_ra"],
            mapdata["n_dec"],
            mapdata["n_sidebands"],
            mapdata["n_channels"],
        )

        mapdata["sigma_wn_coadd"] = sigma_coadd.reshape(
            mapdata["n_ra"],
            mapdata["n_dec"],
            mapdata["n_sidebands"],
            mapdata["n_channels"],
        )

        # Masking non-hit regions with NaNs again
        map[inv_var == 0] = np.nan
        sigma[inv_var == 0] = np.nan

        # Saving and reshaping map data to map object
        mapdata["map"] = map.reshape(
            20,
            mapdata["n_ra"],
            mapdata["n_dec"],
            mapdata["n_sidebands"],
            mapdata["n_channels"],
        )

        mapdata["sigma_wn"] = sigma.reshape(
            20,
            mapdata["n_ra"],
            mapdata["n_dec"],
            mapdata["n_sidebands"],
            mapdata["n_channels"],
        )

        # Deleting numerator and denominator from map object
        del mapdata["numerator_map"]
        del mapdata["denominator_map"]

    def get_pointing_matrix(
        self,
        l2data: L2file,
    ) -> None:
        """Method which converts the {RA, DEC} pointing of the
        level 2 file into a pixel-feed index for later binning the maps.

        Args:
            l2data (L2file): Input level 2 file object for which
            to compute the pixel-feed index.
        """
        # Get pointing from level 2 object
        pointing = l2data["point_cel"]
        ra = pointing[:, :, 0]
        dec = pointing[:, :, 1]

        # Read these from file in future
        NSIDE_RA = self.GRID_SIZE[0]
        NSIDE_DEC = self.GRID_SIZE[1]
        NPIX = NSIDE_RA * NSIDE_DEC
        NFEED = 20

        # Define {RA, DEC} indecies
        idx_ra_allfeed = np.round((ra - self.RA_min) / self.DRA).astype(np.int32)
        idx_dec_allfeed = np.round((dec - self.DEC_min) / self.DDEC).astype(np.int32)

        l2data["pointing_ra_index"] = idx_ra_allfeed
        l2data["pointing_dec_index"] = idx_dec_allfeed

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

        int32_array2 = np.ctypeslib.ndpointer(
            dtype=ctypes.c_int, ndim=2, flags="contiguous"
        )  # 4D array 32-bit integer pointer object.

        if self.params.make_nhit:
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

            # self.mapbinner.bin_map(
            self.mapbinner.bin_nhit_and_map(
                l2data["tod"],
                l2data["inv_var"],
                l2data["freqmask"],
                l2data["pointing_ra_index"],
                l2data["pointing_dec_index"],
                mapdata["nhit"],
                mapdata["numerator_map"],
                mapdata["denominator_map"],
                NFREQ,
                NFEED,
                NSAMP,
                self.GRID_SIZE[0],
                self.GRID_SIZE[1],
                self.OMP_NUM_THREADS,
                l2data.id,
            )
        else:
            self.mapbinner.bin_map.argtypes = [
                float32_array3,  # tod
                float32_array2,  # sigma
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

            self.mapbinner.bin_map(
                l2data["tod"],
                l2data["inv_var"],
                l2data["pointing_ra_index"],
                l2data["pointing_dec_index"],
                mapdata["numerator_map"],
                mapdata["denominator_map"],
                NFREQ,
                NFEED,
                NSAMP,
                self.GRID_SIZE[0],
                self.GRID_SIZE[1],
                self.OMP_NUM_THREADS,
            )


def main():
    if "OMP_NUM_THREADS" in os.environ:
        omp_num_threads = int(os.environ["OMP_NUM_THREADS"])
    else:
        omp_num_threads = 1

    tod2comap = Mapmaker(omp_num_threads=omp_num_threads)
    tod2comap.parse_accept_data()
    tod2comap.run()


if __name__ == "__main__":
    main()

    # TODO:
    # * Cycle when all of scan is masked/rejected
    # * Fix documentations
    # * Make map saver
    # * Implement MPI over scans
    # * Implement splits
    # * Save frequency edges and
    #   centers when Jonas
    #   fixes this in l2gen
    # * HP filter?
    #
