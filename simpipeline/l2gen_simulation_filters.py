import warnings
import numpy as np
import ctypes

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
        self.model_name = params.model_name
        self.verbose = params.verbose
        self.omp_num_threads = omp_num_threads

        # Define c-library used for binning maps
        C_lib_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "../C_libs/cube2tod/cube2tod.so.1",
        )
        self.injector = ctypes.cdll.LoadLibrary(C_lib_path)


    def run(self, l2file: level2_file):
        """Method that injects signal into the TOD of provided level2 file object.

        Args:
            l2file (level2_file): Level 2 file object into which signal is injected.
        """

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
        # print("Time read cube:", time.perf_counter() - t0, "s")

        # Defining simulation cube geometry using standard geometies and boost signal
        simdata.prepare_geometry(l2file.fieldname, self.boost)

        # Euler rotaion of telescope pointing to equatorial origin
        rotated_ra, rotated_dec = simdata.rotate_pointing_to_equator(
            l2file.ra.flatten(), l2file.dec.flatten()
        )

        # Back to original shape {feed, time samples}
        rotated_ra = rotated_ra.reshape(l2file.ra.shape)
        rotated_dec = rotated_dec.reshape(l2file.dec.shape)
                
        # Defining pointers for arrays to send to C++ modules
        float32_array4 = np.ctypeslib.ndpointer(
            dtype=ctypes.c_float, ndim=4, flags="contiguous"
        )  # 4D array 32-bit float pointer object.
        float64_array4 = np.ctypeslib.ndpointer(
            dtype=ctypes.c_double, ndim=4, flags="contiguous"
        )  # 3D array 32-bit float pointer object.

        float64_array3 = np.ctypeslib.ndpointer(
            dtype=ctypes.c_double, ndim=3, flags="contiguous"
        )  # 2D array 32-bit float pointer object.

        float64_array2 = np.ctypeslib.ndpointer(
            dtype=ctypes.c_double, ndim=2, flags="contiguous"
        )  # 2D array 32-bit float pointer object.

        self.injector.cube2tod.argtypes = [
                        float32_array4,  # tod
                        float64_array3,  # Tsys
                        float64_array4,  # simdata
                        float64_array2,  # ra
                        float64_array2,  # dec
                        ctypes.c_float,  # dra
                        ctypes.c_float,  # ddec
                        ctypes.c_float,  # ra0
                        ctypes.c_float,  # dec0
                        ctypes.c_int,    # nra
                        ctypes.c_int,    # ndec
                        ctypes.c_int,    # nfreq
                        ctypes.c_int,    # nfeed
                        ctypes.c_int,    # nsamp
                        ctypes.c_int,    # nthread
                    ]

        # Grid edges of simulation box
        ra_edges = simdata["x"]
        dec_edges = simdata["y"]

        # From bin edges to centers
        ra_grid, dec_grid = simdata.get_bin_centers(ra_edges, dec_edges)
        
        # Grid resolution
        dra = ra_grid[1] - ra_grid[0]
        ddec = dec_grid[1] - dec_grid[0]
        
        # Box dimentions
        nsb, nchannel, ndec, nra = simdata["simulation"].shape
        nfreq = nsb * nchannel

        # Call C++ library by reference and inject TOD with signal
        self.injector.cube2tod(
                    l2file.tod,
                    l2file.Tsys,
                    simdata["simulation"],
                    rotated_ra,
                    rotated_dec,
                    dra, 
                    ddec,
                    ra_grid[0],
                    dec_grid[0],
                    nra,
                    ndec,
                    nfreq,
                    l2file.tod.shape[0],
                    l2file.tod.shape[-1],
                    self.omp_num_threads,
                )        
                
