import argparse
from typing import Dict, Any
import h5py
import numpy as np
import numpy.typing as ntyping
from COmap import COmap

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
        return NotImplemented

    def get_pointing_matrix(
        self, ra: ntyping.ArrayLike, dec: ntyping.ArrayLike
    ) -> ntyping.ArrayLike:
        # Read these from file in future
        FIELDCENT = self.FIELD_CENTER
        NSIDE = self.GRID_SIZE
        NPIX = NSIDE**2

        # RA/Dec grid
        RA = np.zeros(NSIDE)
        DEC = np.zeros(NSIDE)
        dRA = self.DRA / np.abs(np.cos(np.radians(FIELDCENT[1])))
        dDEC = self.DDEC

        # Min values in RA/Dec. directions
        if NSIDE % 2 == 0:
            RA_min = FIELDCENT[0] - dRA * NSIDE / 2.0
            DEC_min = FIELDCENT[1] - dDEC * NSIDE / 2.0

        else:
            RA_min = FIELDCENT[0] - dRA * NSIDE / 2.0 - dRA / 2.0
            DEC_min = FIELDCENT[1] - dDEC * NSIDE / 2.0 - dDEC / 2.0
        print(FIELDCENT[0] - dRA * NSIDE / 2.0, FIELDCENT[1] - dDEC * NSIDE / 2.0)

        print(
            FIELDCENT[0] - dRA * NSIDE / 2.0 - dRA / 2.0,
            FIELDCENT[1] - dDEC * NSIDE / 2.0 - dDEC / 2.0,
        )

        # Defining piRAel centers
        RA[0] = RA_min + dRA / 2
        DEC[0] = DEC_min + dDEC / 2

        for i in range(1, NSIDE):
            RA[i] = RA[i - 1] + dRA
            DEC[i] = DEC[i - 1] + dDEC

        RA_min, DEC_min = RA[0], DEC[0]

        idx = np.round((ra - RA_min) / dRA) * NSIDE + np.round((dec - DEC_min) / dDEC)
        mask = ~np.logical_or(idx < 0, idx >= NPIX)
        mask = np.where(mask)[0]
        return idx.astype(np.int32), mask

    def bin_map(
        self, data: COmap, idx: ntyping.ArrayLike, mask: ntyping.ArrayLike
    ) -> COmap:
        tod = data["tod"][..., mask]
        sigma = data["sigma"]

        inv_var = np.ones_like(tod) / sigma[..., None] ** 2
        nanmask = ~np.isfinite(inv_var)

        tod[nanmask] = 0.0
        inv_var[nanmask] = 0.0

        sidebands, channels, _samples = tod.shape

        numinator = np.zeros((sidebands, channels, NSIDE * NSIDE))
        denominator = np.zeros_like(numinator)
        hits = np.ones(denominator.shape, dtype=np.int32)

        for sb in range(sidebands):
            for freq in range(channels):
                hits[sb, freq] = np.bincount(idx, minlength=NSIDE * NSIDE)

                numinator[sb, freq, :] = np.bincount(
                    idx,
                    minlength=NSIDE * NSIDE,
                    weights=tod[sb, freq, ...] * inv_var[sb, freq, ...],
                )

                denominator[sb, freq, :] = np.bincount(
                    idx, minlength=NSIDE * NSIDE, weights=inv_var[sb, freq, ...]
                )

        return numinator, denominator, hits


def main():
    if "OMP_NUM_THREADS" in os.environ:
        omp_num_threads = int(os.environ["OMP_NUM_THREADS"])
    else:
        omp_num_threads = 1
    tod2comap = Mapmaker(omp_num_threads=omp_num_threads)


if __name__ == "__main__":
    main()
