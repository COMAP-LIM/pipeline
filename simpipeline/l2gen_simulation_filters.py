import warnings
import numpy as np
import ctypes
from pixell import enmap, utils

# Ignore RuntimeWarning
warnings.filterwarnings("ignore", category=RuntimeWarning)


import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current)

sys.path.append(parent_directory)

from l2gen_l2class import level2_file
from simpipeline.l2_simulations import SimCube


class Cube2TOD_nn:
    """General signal cube nearest neighbor (nn) injection class.

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
        
        if params.signal_path is not None:
            self.simpath = params.signal_path
        else:
            self.simpath = params.sim_output_dir
            self.simpath = os.path.join(self.simpath, params.sim_map_output_file_name)

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

        simdata.read_cube_geometry()

        # Defining simulation cube geometry using standard geometies and boost signal
        simdata.prepare_geometry(l2file.fieldname)
        # print("Read and prepare geometry:", time.perf_counter() - t00, "s")
        
        # Euler rotaion of telescope pointing to equatorial origin
        rotated_ra, rotated_dec = simdata.rotate_pointing_to_equator(
            l2file.ra.flatten(), l2file.dec.flatten()
        )

        # Find extrema of rotated equator pointing
        max_ra = np.deg2rad(np.max(rotated_ra))
        min_ra = np.deg2rad(np.min(rotated_ra))
        max_dec = np.deg2rad(np.max(rotated_dec))
        min_dec = np.deg2rad(np.min(rotated_dec))

        # Find indecies the extrema of rotated pointing correspond to
        max_dec_idx, max_ra_idx = utils.nint(simdata.equator_geometry.sky2pix((max_dec, max_ra)))
        min_dec_idx, min_ra_idx = utils.nint(simdata.equator_geometry.sky2pix((min_dec, min_ra)))
        
        N_dec, N_ra = simdata.equator_geometry.shape
    
        if max_dec_idx < 0 or max_dec_idx > N_dec:
            max_dec_idx = N_dec
        if min_dec_idx < 0 or min_dec_idx > N_dec: 
            min_dec_idx = 0
    
        if max_ra_idx < 0 or max_ra_idx > N_dec:
            max_ra_idx = N_ra
        if min_ra_idx < 0 or min_ra_idx > N_dec: 
            min_ra_idx = 0

        # Slice equatorial box geometry. Pixel centers will automatically also be updated
        # if they are produced by simdata.equator_geometry.posmap or the like. 
        simdata.equator_geometry = simdata.equator_geometry[min_dec_idx:max_dec_idx, min_ra_idx:max_ra_idx]

        # Reading simulation cube data from file
        simdata.read(
            self.boost, 
            (min_dec_idx, max_dec_idx), 
            (min_ra_idx, max_ra_idx)
        )

        # Testing if frequency array in new simulation format has same bin centers as level2 data
        if ".npz" in self.simpath:
            np.testing.assert_allclose(l2file.freq_bin_centers, simdata["frequencies"], atol = 0)
        
        # Get new pixel centers of sliced equatorial geometry 
        ra_grid, dec_grid = simdata.get_bin_centers()

        # Grid resolution
        dra = ra_grid[1] - ra_grid[0]
        ddec = dec_grid[1] - dec_grid[0]

        # Box dimentions
        nsb, nchannel, ndec, nra = simdata["simulation"].shape
        nfreq = nsb * nchannel


        # Emplty signal TOD that is to be filled by C++ module.
        signal_tod = np.zeros_like(l2file.tod) 
        
        # Back to original shape {feed, time samples}
        rotated_ra = rotated_ra.reshape(l2file.ra.shape)
        rotated_dec = rotated_dec.reshape(l2file.dec.shape)
        
        # Defining pointers for arrays to send to C++ modules
        float32_array4 = np.ctypeslib.ndpointer(
            dtype=ctypes.c_float, ndim=4, flags="contiguous"
        )  
        float64_array2 = np.ctypeslib.ndpointer(
            dtype=ctypes.c_double, ndim=2, flags="contiguous"
        )  

        self.injector.replace_tod_with_nearest_neighbor_signal.argtypes = [
                        float32_array4,  # tod
                        float32_array4,  # simdata
                        float64_array2,  # ra
                        float64_array2,  # dec
                        ctypes.c_float,  # dra
                        ctypes.c_float,  # ddec
                        ctypes.c_float,  # ra0
                        ctypes.c_float,  # dec0
                        ctypes.c_int,    # nra
                        ctypes.c_int,    # ndec
                        ctypes.c_long,    # nfreq
                        ctypes.c_long,    # nfeed
                        ctypes.c_long,    # nsamp
                        ctypes.c_int,    # nthread
                    ]
        
        # Call C++ library by reference and inject TOD with signal
        self.injector.replace_tod_with_nearest_neighbor_signal(
                    signal_tod,
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


        l2file.tod *= (1 + signal_tod / l2file.Tsys[..., None])
        l2file.signal_tod = signal_tod

class Cube2TOD_bilinear:
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
        
        if params.signal_path is not None:
            self.simpath = params.signal_path
        else:
            self.simpath = params.sim_output_dir
            self.simpath = os.path.join(self.simpath, params.sim_map_output_file_name)

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

        simdata.read_cube_geometry()

        # Defining simulation cube geometry using standard geometies and boost signal
        simdata.prepare_geometry(l2file.fieldname)
        # print("Read and prepare geometry:", time.perf_counter() - t00, "s")
        
        # Euler rotaion of telescope pointing to equatorial origin
        rotated_ra, rotated_dec = simdata.rotate_pointing_to_equator(
            l2file.ra.flatten(), l2file.dec.flatten()
        )

        # Find extrema of rotated equator pointing
        max_ra = np.deg2rad(np.max(rotated_ra))
        min_ra = np.deg2rad(np.min(rotated_ra))
        max_dec = np.deg2rad(np.max(rotated_dec))
        min_dec = np.deg2rad(np.min(rotated_dec))

        # Find indecies the extrema of rotated pointing correspond to
        max_dec_idx, max_ra_idx = utils.nint(simdata.equator_geometry.sky2pix((max_dec, max_ra)))
        min_dec_idx, min_ra_idx = utils.nint(simdata.equator_geometry.sky2pix((min_dec, min_ra)))

        N_dec, N_ra = simdata.equator_geometry.shape
    
        if max_dec_idx < 0 or max_dec_idx > N_dec:
            max_dec_idx = N_dec
        if min_dec_idx < 0 or min_dec_idx > N_dec: 
            min_dec_idx = 0
    
        if max_ra_idx < 0 or max_ra_idx > N_dec:
            max_ra_idx = N_ra
        if min_ra_idx < 0 or min_ra_idx > N_dec: 
            min_ra_idx = 0

        # Slice equatorial box geometry. Pixel centers will automatically also be updated
        # if they are produced by simdata.equator_geometry.posmap or the like. 
        simdata.equator_geometry = simdata.equator_geometry[min_dec_idx:max_dec_idx, min_ra_idx:max_ra_idx]

        # Reading simulation cube data from file
        simdata.read(
            self.boost, 
            (min_dec_idx, max_dec_idx), 
            (min_ra_idx, max_ra_idx)
        )

        # Testing if frequency array in new simulation format has same bin centers as level2 data
        if ".npz" in self.simpath:
            np.testing.assert_allclose(l2file.freq_bin_centers, simdata["frequencies"], atol = 0)
        
        # Get new pixel centers of sliced equatorial geometry 
        ra_grid, dec_grid = simdata.get_bin_centers()

        # Grid resolution
        dra = ra_grid[1] - ra_grid[0]
        ddec = dec_grid[1] - dec_grid[0]

        # Box dimentions
        nsb, nchannel, ndec, nra = simdata["simulation"].shape
        nfreq = nsb * nchannel


        # Emplty signal TOD that is to be filled by C++ module.
        signal_tod = np.zeros_like(l2file.tod) 
        
        # Back to original shape {feed, time samples}
        rotated_ra = rotated_ra.reshape(l2file.ra.shape)
        rotated_dec = rotated_dec.reshape(l2file.dec.shape)
                
        # Defining pointers for arrays to send to C++ modules
        float32_array4 = np.ctypeslib.ndpointer(
            dtype=ctypes.c_float, ndim=4, flags="contiguous"
        )  
        float64_array2 = np.ctypeslib.ndpointer(
            dtype=ctypes.c_double, ndim=2, flags="contiguous"
        )  

        self.injector.replace_tod_with_bilinear_interp_signal.argtypes = [
                        float32_array4,  # tod
                        float32_array4,  # simdata
                        float64_array2,  # ra
                        float64_array2,  # dec
                        ctypes.c_float,  # dra
                        ctypes.c_float,  # ddec
                        ctypes.c_float,  # ra0
                        ctypes.c_float,  # dec0
                        ctypes.c_int,    # nra
                        ctypes.c_int,    # ndec
                        ctypes.c_long,    # nfreq
                        ctypes.c_long,    # nfeed
                        ctypes.c_long,    # nsamp
                        ctypes.c_int,    # nthread
                    ]
        
        # Call C++ library by reference and inject TOD with signal
        self.injector.replace_tod_with_bilinear_interp_signal(
                    signal_tod,
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


        l2file.tod *= (1 + signal_tod / l2file.Tsys[..., None])
        l2file.signal_tod = signal_tod
    
class Replace_TOD_With_Signal:
    """General signal cube injection class.

    Args:
        Filter (Filter): Filter super class on which to build signal injector.

    NOTE: This filter class should only be used after computing the system temperature
    as this is needed to inject singal
    """

    name = "cube2tod_signal_only"
    name_long = "Signal cube substitutuin filter"
    run_when_masking = False  # If set to True, this filter will be applied to a local copy of the data before masking.
    has_corr_template = False  # Set to True if the template impacts the correlation matrix and must be corrected (through the "get_corr_template" function).

    def __init__(self, params, omp_num_threads):
        self.boost = params.boost_factor
        
        if params.signal_path is not None:
            self.simpath = params.signal_path
        else:
            self.simpath = params.sim_output_dir
            self.simpath = os.path.join(self.simpath, params.sim_map_output_file_name)

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

        # Testing if frequency array in new simulation format has same bin centers as level2 data
        if ".npz" in self.simpath:
            np.testing.assert_allclose(l2file.freq_bin_centers, simdata["frequencies"], atol = 0)
            
        simdata.read_cube_geometry()

        # Defining simulation cube geometry using standard geometies and boost signal
        simdata.prepare_geometry(l2file.fieldname)

        # Euler rotaion of telescope pointing to equatorial origin
        rotated_ra, rotated_dec = simdata.rotate_pointing_to_equator(
            l2file.ra.flatten(), l2file.dec.flatten()
        )

        # Find extrema of rotated equator pointing
        max_ra = np.deg2rad(np.max(rotated_ra))
        min_ra = np.deg2rad(np.min(rotated_ra))
        max_dec = np.deg2rad(np.max(rotated_dec))
        min_dec = np.deg2rad(np.min(rotated_dec))

        # Find indecies the extrema of rotated pointing correspond to
        max_dec_idx, min_ra_idx = utils.nint(simdata.equator_geometry.sky2pix((max_dec, max_ra)))
        min_dec_idx, max_ra_idx = utils.nint(simdata.equator_geometry.sky2pix((min_dec, min_ra)))

        # Expand geometry by four pixels to be sure all needed simualtion pixels are included.
        max_dec_idx += 4
        max_ra_idx += 4
        min_dec_idx -= 4
        min_ra_idx -= 4
        
        # Slice equatorial box geometry. Pixel centers will automatically also be updated
        # if they are produced by simdata.equator_geometry.posmap or the like. 
        simdata.equator_geometry = simdata.equator_geometry[min_dec_idx:max_dec_idx, min_ra_idx:max_ra_idx]

        # Reading simulation cube data from file
        simdata.read(
            self.boost, 
            (min_dec_idx, max_dec_idx), 
            (min_ra_idx, max_ra_idx)
        )

        # Get new pixel centers of sliced equatorial geometry 
        ra_grid, dec_grid = simdata.get_bin_centers()
        
        # Grid resolution
        dra = ra_grid[1] - ra_grid[0]
        ddec = dec_grid[1] - dec_grid[0]
        
        # Box dimentions
        nsb, nchannel, ndec, nra = simdata["simulation"].shape
        nfreq = nsb * nchannel

        # Emplty signal TOD that is to be filled by C++ module.
        signal_tod = np.zeros_like(l2file.tod) 
        
        # Back to original shape {feed, time samples}
        rotated_ra = rotated_ra.reshape(l2file.ra.shape)
        rotated_dec = rotated_dec.reshape(l2file.dec.shape)
                
        # Defining pointers for arrays to send to C++ modules
        float32_array4 = np.ctypeslib.ndpointer(
            dtype=ctypes.c_float, ndim=4, flags="contiguous"
        )  

        float64_array2 = np.ctypeslib.ndpointer(
            dtype=ctypes.c_double, ndim=2, flags="contiguous"
        )  

        self.injector.replace_tod_with_bilinear_interp_signal.argtypes = [
                        float32_array4,  # tod
                        float32_array4,  # simdata
                        float64_array2,  # ra
                        float64_array2,  # dec
                        ctypes.c_float,  # dra
                        ctypes.c_float,  # ddec
                        ctypes.c_float,  # ra0
                        ctypes.c_float,  # dec0
                        ctypes.c_int,    # nra
                        ctypes.c_int,    # ndec
                        ctypes.c_long,    # nfreq
                        ctypes.c_long,    # nfeed
                        ctypes.c_long,    # nsamp
                        ctypes.c_int,    # nthread
                    ]

        # Call C++ library by reference and inject TOD with signal
        self.injector.replace_tod_with_bilinear_interp_signal(
                    l2file.tod,
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
        

class Replace_TOD_with_Tsys_WN:
    """ Replace the entire time-stream with white noise, from the radiometer equation (using Tsys).
        NOTE: This filter class should only be used after computing the system temperature
    """

    name = "tsysWNsim"
    name_long = "Tsys white noise simulation"
    run_when_masking = False  # If set to True, this filter will be applied to a local copy of the data before masking.
    has_corr_template = False  # Set to True if the template impacts the correlation matrix and must be corrected (through the "get_corr_template" function).

    def __init__(self, params, omp_num_threads=2):
        self.params = params
        self.omp_num_threads = omp_num_threads


    def run(self, l2):
        l2.is_sim = False
        l2.is_wn_sim = True

        Gain = 10000
        tsys = l2.Tsys
        dt = l2.tod_times_seconds[1] - l2.tod_times_seconds[0]
        dnu = (l2.freqs[0][1] - l2.freqs[0][0])*1e9

        if not self.params.wn_sim_seed is None:
            np.random.seed(self.params.wn_sim_seed*l2.scanid)

        for ifeed in range(l2.Nfeeds):
            radiometer_noise = tsys[ifeed]/np.sqrt(dt*dnu)
            l2.freqmask[ifeed, radiometer_noise < 0] = False
            radiometer_noise[radiometer_noise < 0] = 1.0
            l2.tod[ifeed] = Gain*(tsys[ifeed][:,:,None] + np.random.normal(0, radiometer_noise[:,:,None], (l2.Nsb, l2.Nfreqs, l2.Ntod)))




class Replace_TOD_with_WN:
    """ Replace the entire time-stream with pure white noise, centered around zero, without using Tsys.
        NOTE: This filter should not be used in combination with normalization!
    """

    name = "WNsim"
    name_long = "White noise simulation"
    run_when_masking = False  # If set to True, this filter will be applied to a local copy of the data before masking.
    has_corr_template = False  # Set to True if the template impacts the correlation matrix and must be corrected (through the "get_corr_template" function).

    def __init__(self, params, omp_num_threads=2):
        self.params = params
        self.omp_num_threads = omp_num_threads


    def run(self, l2):
        l2.is_sim = False
        l2.is_wn_sim = True
        if not self.params.wn_sim_seed is None:
            np.random.seed(self.params.wn_sim_seed*l2.scanid)

        for ifeed in range(l2.Nfeeds):
            l2.tod[ifeed] = np.random.normal(0, 17.0, (l2.Nsb, l2.Nfreqs, l2.Ntod))
