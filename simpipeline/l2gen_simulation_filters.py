import warnings
import numpy as np

# Ignore RuntimeWarning
warnings.filterwarnings("ignore", category=RuntimeWarning)


import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current)

sys.path.append(parent_directory)

from l2gen_l2class import level2_file
from simpipeline.l2_simulations import SimCube


class Cube2TOD:
    """General signal cube injection class.

    Args:
        Filter (Filter): Filter super class on which to build signal injector.

    NOTE: This filter class should only be used after computing the system temperature
    as this is needed to inject singal
    """

    name = "cube2tod"
    name_long = "Signal cube injector filter"
    run_when_masking = False  # If set to True, this filter will be applied to a local copy of the data before masking.
    has_corr_template = False  # Set to True if the template impacts the correlation matrix and must be corrected (through the "get_corr_template" function).

    def __init__(self, params, omp_num_threads):
        self.boost = params.boost_factor
        self.simpath = params.signal_path
        self.verbose = params.verbose

    def run(self, l2file: level2_file):
        """Method that injects signal into the TOD of provided level2 file object.

        Args:
            l2file (level2_file): Level 2 file object into which signal is injected.
        """
        # if self.verbose:
        #     print(f"Scan {l2file.scanid} -- Injecting cube signal into TOD")

        # Marking l2file as simulation by setting;
        l2file.is_sim = True

        if self.simpath is None:
            raise TypeError(
                "To run signal injection from a signal cube make sure to specify the path to the signal cube HDF5 file."
            )

        # Setting up simulation cube object
        simdata = SimCube(self.simpath)

        # Reading simulation cube data from file
        simdata.read()

        # Defining simulation cube geometry using standard geometies and boost signal
        simdata.prepare_geometry(l2file.fieldname, self.boost)

        # Euler rotaion of telescope pointing to equatorial origin
        rotated_ra, rotated_dec = simdata.rotate_pointing_to_equator(
            l2file.ra.flatten(), l2file.dec.flatten()
        )

        rotated_ra = rotated_ra.reshape(l2file.ra.shape)
        rotated_dec = rotated_dec.reshape(l2file.dec.shape)

        simdata.interpolate_cube(l2file.fieldname)

        for sb in range(l2file.Nsb):
            for freq in range(l2file.Nfreqs):

                # signal = simdata.signal[sb][freq](l2file.dec, l2file.ra, grid=False)
                signal = simdata.signal[sb][freq](rotated_dec, rotated_ra, grid=False)
                l2file.tod[:, sb, freq, :] *= (
                    1 + signal / l2file.Tsys[:, sb, freq, None]
                )

        # Reading off signal from cube given telescope pointing.
        # signal = simdata.sim2tod(l2file.ra, l2file.dec)

        # # Injecting signal to TOD
        # l2file.tod *= 1 + signal / l2file.Tsys[..., None]
