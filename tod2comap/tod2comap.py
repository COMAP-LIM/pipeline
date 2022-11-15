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
        self.omp_num_threads = omp_num_threads

    def read_params(self):
        from l2gen_argparser import parser

        params = parser.parse_args()
        if not params.runlist:
            raise ValueError(
                "A runlist must be specified in parameter file or terminal."
            )
        self.params = params

    def run(self):
        pass

    def get_pointing_matrix(
        self, ra: ntyping.ArrayLike, dec: ntyping.ArrayLike
    ) -> ntyping.ArrayLike:
        # Read these from file in future
        FIELDCENT = [226.00, 55.00]
        DPIX = 2 / 60
        NSIDE = 120
        NPIX = NSIDE * NSIDE

        # RA/Dec grid
        RA = np.zeros(NSIDE)
        DEC = np.zeros(NSIDE)
        dRA = DPIX / np.abs(np.cos(np.radians(FIELDCENT[1])))
        dDEC = DPIX

        # Min values in RA/Dec. directions
        if NSIDE % 2 == 0:
            RA_min = FIELDCENT[0] - dRA * NSIDE / 2.0
            DEC_min = FIELDCENT[1] - dDEC * NSIDE / 2.0

        else:
            RA_min = FIELDCENT[0] - dRA * NSIDE / 2.0 - dRA / 2.0
            DEC_min = FIELDCENT[1] - dDEC * NSIDE / 2.0 - dDEC / 2.0

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
    tod2comap.read_params()

    print(tod2comap.params.level2_dir)


if __name__ == "__main__":
    main()
