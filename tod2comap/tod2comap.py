import numpy as np
import numpy.typing as ntyping
import ctypes
import time

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

        # Read parameter file and runlist
        self.read_params()
        self.read_runlist()

        # Define map grid parameters
        self.GRID_SIZE = self.params.grid_size
        self.DRA, self.DDEC = self.params.grid_res

        self.FIELD_CENTER = self.params.field_center[self.fieldname]

        if not self.params.map_name:
            raise ValueError(
                "A map file name must be specified in parameter file or terminal."
            )

        self.mapbinner = ctypes.cdll.LoadLibrary("mapbinner.so.1")

    def read_params(self):
        from l2gen_argparser import parser

        params = parser.parse_args()
        if not params.runlist:
            raise ValueError(
                "A runlist must be specified in parameter file or terminal."
            )
        self.params = params

    def read_runlist(self):
        print(
            f"Creating runlist in specified obsid range [{self.params.obsid_start}, {self.params.obsid_stop}]"
        )
        print(f"Runlist file: {self.params.runlist}")

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

        print(f"Field name:                 {fieldname}")
        print(f"Obsids in runlist file:     {n_obsids}")
        print(f"Scans in runlist file:      {n_scans_tot}")
        print(f"Scans included in run:      {len(runlist)}")
        print(f"Scans outside obsid range:  {n_scans_outside_range}")

        self.fieldname = fieldname
        self.runlist = runlist

    def run(self):
        """Method running through the provided runlist and binning up maps."""

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

        for scan in self.runlist:
            print("-" * 20)
            print(f"Processing scan {scan[0]}")

            l2path = scan[-1]

            t0 = time.perf_counter()

            l2data = L2file(path=l2path)
            print("Time to define L2file:", 1e3 * (time.perf_counter() - t0), "ms")
            t0 = time.perf_counter()

            l2data.read_l2()
            print(
                "Time to read L2 data from HDF5:",
                1e3 * (time.perf_counter() - t0),
                "ms",
            )
            t0 = time.perf_counter()

            self.preprocess_l2data(l2data)
            print("Time to pre-process:", 1e3 * (time.perf_counter() - t0), "ms")
            t0 = time.perf_counter()

            self.get_pointing_matrix(l2data)
            print(
                "Time to compute pointing idx:", 1e3 * (time.perf_counter() - t0), "ms"
            )
            t0 = time.perf_counter()

            self.bin_map(full_map, l2data)
            print("Time to bin map:", 1e3 * (time.perf_counter() - t0), "ms")
            print("-" * 20)

        self.postprocess_map(full_map)

    def preprocess_l2data(
        self,
        l2data: L2file,
    ):
        """Method that brings the feed infomation in needed datasets
        to correct correct order."""

        _, NSB, NFREQ, NSAMP = l2data["tod"].shape

        # Defining empty buffers
        tod = np.zeros((20, NSB, NFREQ, NSAMP), dtype=np.float32)
        sigma0 = np.zeros((20, NSB, NFREQ), dtype=np.float32)
        freqmask = np.zeros((20, NSB, NFREQ), dtype=np.int32)
        pointing = np.zeros((20, NSAMP, 3), dtype=np.float32)

        # Get index to pixel mapping
        pixels = l2data["pixels"] - 1

        # Sort pixels to correct buffer position
        tod[pixels, ...] = l2data["tod"]
        sigma0[pixels, ...] = l2data["sigma0"]
        freqmask[pixels, ...] = l2data["freqmask"]
        pointing[pixels, ...] = l2data["point_cel"]

        # Flip sideband 0 and 2
        tod[:, (0, 2), :, :] = tod[:, (0, 2), ::-1, :]
        sigma0[:, (0, 2), :] = sigma0[:, (0, 2), ::-1]
        freqmask[:, (0, 2), :] = freqmask[:, (0, 2), ::-1]

        tod = tod.transpose(
            0,
            3,
            1,
            2,
        )

        # Flatten frequency axis
        tod = tod.reshape(20, NSAMP, NSB * NFREQ)
        sigma0 = sigma0.reshape(20, NSB * NFREQ).T
        freqmask = freqmask.reshape(20, NSB * NFREQ).T

        # Enfore transposition in memory
        tod = np.ascontiguousarray(tod, dtype=np.float32)
        sigma0 = np.ascontiguousarray(sigma0, dtype=np.float32)
        freqmask = np.ascontiguousarray(freqmask, dtype=np.float32)

        # Pre-compute masking
        inv_var = 1 / sigma0**2  # Define inverse variance
        sigma0[freqmask == 0] = 0

        # Overwrite old
        l2data["tod"] = tod
        l2data["sigma0"] = sigma0
        l2data["inv_var"] = inv_var
        l2data["freqmask"] = freqmask
        l2data["point_cel"] = pointing

    def postprocess_map(self, mapdata: COmap) -> None:
        return NotImplemented

    def get_pointing_matrix(
        self,
        l2data: L2file,
    ) -> None:

        pointing = l2data["point_cel"]
        ra = pointing[:, :, 0]
        dec = pointing[:, :, 1]

        # Read these from file in future
        NSIDE = self.GRID_SIZE
        NPIX = NSIDE**2
        NFEED = 20

        idx_ra_allfeed = np.round((ra - self.RA_min) / self.DRA).astype(np.int32)
        idx_dec_allfeed = np.round((dec - self.DEC_min) / self.DDEC).astype(np.int32)

        idx_pix = idx_dec_allfeed * NSIDE + idx_ra_allfeed
        pointing_mask = ~np.logical_or(idx_pix < 0, idx_pix >= NPIX)
        pointing_mask = np.where(pointing_mask)

        # Clip pixel index to 0 for pointing outside field grid.
        # See further down for how corresponding TOD values are zeroed out.
        idx_pix[pointing_mask] = 0

        pointing_idx = NSIDE**2 * np.arange(NFEED)[:, None] + idx_pix

        l2data["pointing_index"] = pointing_idx.astype(np.int32)

        # Set TOD values outside field to zero
        l2data["tod"][pointing_mask[0], pointing_mask[1], :] = 0.0

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

        if self.params.make_nhit:
            int32_array4 = np.ctypeslib.ndpointer(
                dtype=ctypes.c_int, ndim=4, flags="contiguous"
            )  # 4D array 32-bit integer pointer object.

        int32_array2 = np.ctypeslib.ndpointer(
            dtype=ctypes.c_int, ndim=2, flags="contiguous"
        )  # 4D array 32-bit integer pointer object.

        self.mapbinner.bin_map.argtypes = [
            float32_array3,  # tod
            float32_array2,  # sigma
            int32_array2,  # idx_pix
            float32_array4,  # numerator map
            float32_array4,  # denominator map
            ctypes.c_int,  # nfreq
            ctypes.c_int,  # nsamp
            ctypes.c_int,  # nside
            ctypes.c_int,  # nfeed
            ctypes.c_int,  # nthread
        ]

        NFEED, NSAMP, NFREQ = l2data["tod"].shape

        self.mapbinner.bin_map(
            l2data["tod"],
            l2data["inv_var"],
            l2data["pointing_index"],
            mapdata["numerator_map"],
            mapdata["denominator_map"],
            NFREQ,
            NFEED,
            NSAMP,
            self.GRID_SIZE,
            self.OMP_NUM_THREADS,
        )


def main():
    if "OMP_NUM_THREADS" in os.environ:
        omp_num_threads = int(os.environ["OMP_NUM_THREADS"])
    else:
        omp_num_threads = 1

    tod2comap = Mapmaker(omp_num_threads=omp_num_threads)
    tod2comap.run()


if __name__ == "__main__":
    main()
