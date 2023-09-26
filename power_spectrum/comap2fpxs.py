

from __future__ import annotations
from typing import Sequence
import numpy.typing as npt
import h5py
import numpy as np
from dataclasses import dataclass, field
import os
import sys
import pickle
import itertools
from mpi4py import MPI
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import time 
import scipy.interpolate as interpolate

import tqdm

current = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current)
sys.path.append(parent_directory)

from TransferFunction import TransferFunction
import xs_class

class COMAP2FPXS():
    """COMAP feed-feed pseudo cross-spectrum (FPXS) class
    """
    def __init__(self, omp_num_threads: int = 1):
        """Class initializer defining essential class attributes and calls important setup methods.

        Args:
            omp_num_threads (int, optional): Number of openMP threads to use when computing with scipy.fft. Defaults to 1.

        Raises:
            ValueError: Error raised when more than one feed-cross-varaiable is used. I.e. can only compute cross-spectrum across one variable currently.
            ValueError: Error raised when more than one secondary variable is used when computing map-difference null tests.
            ValueError: Error raised when using a split base larger than 2. May change in the future!
        """
        # Number of openMP threads
        self.OMP_NUM_THREADS = omp_num_threads

        # Define MPI parameters as class attribites
        self.comm = MPI.COMM_WORLD
        self.Nranks = self.comm.Get_size()
        self.rank = self.comm.Get_rank()

        # Read parameter file/command-line
        self.read_params()

        # Whether to verbose print
        self.verbose = self.params.verbose == 1
        
        # Load and define interpolation of transfer functions
        self.define_transfer_function()

        # Read in pre-defined cosmology from file
        self.read_cosmology()

        # Read and sort splits according to jackknive defenition file
        self.read_jackknife_definition_file()

        if len(self.cross_and_secondary) > 1:
            #####
            # CURRENTLY THERE IS NO SUPPORT FOR MORE THAN ONE FEED CROSS-CORRELATION VARIABLE
            ####
            raise ValueError("Cannot have more than one feed cross-correlation variable")
        

        if self.params.psx_null_diffmap:
            if len(self.secondary_variables) > 1:
                ######
                # FOR NOW IF DIFFERENCE MAP NULLTESTS ARE 
                # PERFORMED ONLY ONE SECONDARY 
                # SPLIT VARIABLE VAR SUPPORTED. 
                # ADJUST THIS LATER IF NEEDED 
                ######
                raise ValueError("Cannot have more than one secondary split variable when performing difference map null tests.")
            if self.params.split_base_number > 2:
                raise ValueError("Cannot currently perform difference map null test with split base number > 2")

        # Generate all combinations of split maps 
        self.generate_split_map_names()

        if self.params.psx_generate_white_noise_sim:
            if self.params.psx_white_noise_sim_seed is None:
                self.generate_new_monte_carlo_seed()

    def run(self):
        """Main run method to perform computation of cross-spectra
        """

        if self.params.psx_mode == "feed":
            # Only want to compute FPXS for included feeds (detectors)
            self.included_feeds = self.params.included_feeds

            # Define all combinations between detectors
            feed_combinations = list(itertools.product(self.included_feeds, self.included_feeds))
            self.feed_combinations = feed_combinations
        elif self.params.psx_mode == "saddlebag":
            self.included_feeds = [1,2,3,4]
            self.feed_combinations = list(itertools.product(self.included_feeds, self.included_feeds))

        # Compute FPXS for maps in psx_map_names list
        mapnames = self.params.psx_map_names

        mapnames = [name for name in mapnames]

        if self.params.psx_null_cross_field:
            ##### CURRENTLY NOT IN USE #####
            # Compute cross-field spectra

            if len(mapnames) == 0:
                # If provided map name file list is empty, just use map-maker map name 
                # and all included fields (assuming that all fields have same filename pattern).
                fields = self.params.fields
                mapnames = [f"{field_name}_{self.params.map_name}{self.params.psx_map_name_postfix}.h5" for field_name in fields]
            
            # Map file name combinations
            field_combinations = list(itertools.product(mapnames, mapnames))
            
        elif len(mapnames) == 0 and not self.params.psx_null_cross_field:
            # If now custom map file name list is provided, and no cross-field spectra are computed,
            # assume that map file name follows mapmaker file name pattern for all fields.

            fields = self.params.fields
            field_combinations = [(f"{field_name}_{self.params.map_name}{self.params.psx_map_name_postfix}.h5", f"{field_name}_{self.params.map_name}{self.params.psx_map_name_postfix}.h5")
                        for field_name in fields]            
        else:
            # Else use file names from map file name list (when to compute FPXS of custom map list)
            field_combinations = [(name, name) for name in mapnames]
        
        self.field_combinations = field_combinations        

        # Generate tuple of (field/map filename, split map key, feed combination) to use for computing FPXS 
        if self.params.psx_only_feed_splits:
            self.split_map_combinations = [(None, None)]
        all_combinations = list(itertools.product(self.field_combinations, self.split_map_combinations, self.feed_combinations))
        
        Number_of_combinations = len(all_combinations)
        
        if self.rank == 0:
            print("#" * 70)
            print(f"Primary splits: {self.primary_variables}")
            print(f"Secondary splits: {self.secondary_variables}")
            print(f"Computing cross-spectra for {Number_of_combinations} combinations with {self.Nranks} MPI processes:")
            print("#" * 70)

        self.comm.Barrier()

        self.params.primary_variables = self.primary_variables
        
        if self.rank == 0 and not self.verbose:
            pbar = tqdm.tqdm(
                total = Number_of_combinations, 
                colour = "red", 
                ncols = 80,
                desc = f"Total",
                position = 0,
            )
        # MPI parallel run over all FPXS combinations
        for i in range(Number_of_combinations):

            # If not verbose use progress bars instead of informative prints
            if not self.verbose:
                prog_tot = self.comm.reduce(
                    1,
                    op = MPI.SUM,
                    root = 0
                )
                
                if self.rank == 0:
                    pbar.refresh()
                    pbar.n = int(pbar.n + prog_tot / self.Nranks)
                
            if i % self.Nranks == self.rank:
                
                # Extract file names, split keys and feed combinations from current combination
                mapnames, splits, feeds = all_combinations[i]
       
                map1, map2 = mapnames
                split1, split2 = splits
                feed1, feed2 = feeds

                # Construnct full map file paths
                mappaths = [
                    os.path.join(self.params.map_dir, map1),
                    os.path.join(self.params.map_dir, map2),
                ]

                # Generate name of outpute data directory
                mapname1 = map1.split("/")[-1]
                mapname2 = map2.split("/")[-1]
                if self.params.psx_null_cross_field:
                    outdir = f"{mapname1[:-3]}_X_{mapname2[:-3]}"
                else:
                    outdir = f"{mapname1[:-3]}"

                if self.params.psx_generate_white_noise_sim:
                    outdir_data = f"{outdir}"
                    outdir = f"{outdir}/white_noise_seed{self.params.psx_white_noise_sim_seed}"
                
                if self.verbose:
                    # Print some usefull information while computing FPXS
                    if self.params.psx_null_diffmap:
                        if split1 is None or split2 is None:
                            print(f"\033[91m Rank {self.rank} ({i + 1} / {Number_of_combinations}): \033[00m \033[94m {mapname1.split('_')[0]} X {mapname2.split('_')[0]} \033[00m \033[00m \033[92m \033[00m \033[93m Feed {feed1} X Feed {feed2} \033[00m")
                        else:
                            print(f"\033[91m Rank {self.rank} ({i + 1} / {Number_of_combinations}): \033[00m \033[94m {mapname1.split('_')[0]} X {mapname2.split('_')[0]} \033[00m \033[00m \033[92m ({split1[0].split('/map_')[-1]} - {split1[1].split('/map_')[-1]}) X ({split2[0].split('/map_')[-1]} - {split2[1].split('/map_')[-1]}) \033[00m \033[93m Feed {feed1} X Feed {feed2} \033[00m")
                    else:
                        if split1 is None or split2 is None:
                            print(f"\033[91m Rank {self.rank} ({i + 1} / {Number_of_combinations}): \033[00m \033[94m {mapname1.split('_')[0]} X {mapname2.split('_')[0]} \033[00m \033[00m \033[92m \033[00m \033[93m Feed {feed1} X Feed {feed2} \033[00m")  
                        else:
                            print(f"\033[91m Rank {self.rank} ({i + 1} / {Number_of_combinations}): \033[00m \033[94m {mapname1.split('_')[0]} X {mapname2.split('_')[0]} \033[00m \033[00m \033[92m {split1.split('/map_')[-1]} X {split2.split('/map_')[-1]} \033[00m \033[93m Feed {feed1} X Feed {feed2} \033[00m")  
                    
                # Generate cross-spectrum instance from split keys and feeds of current FPXS combo 
                cross_spectrum = xs_class.CrossSpectrum_nmaps(
                        self.params, 
                        splits, 
                        feed1, 
                        feed2
                    )
                
                # If map difference null test is computed the output is saved in separate directory
                if self.params.psx_null_diffmap:
                    outdir = os.path.join(outdir, f"null_diffmap/{cross_spectrum.null_variable}")
                
                    
                # Skip loop iteration if cross-spectrum of current combination already exists
                if os.path.exists(os.path.join(self.params.power_spectrum_dir, "spectra_2D", outdir, cross_spectrum.outname)):
                    continue
                else:
                    # Read maps from generated map paths, and provide cosmology to translate from
                    # observational units to cosmological ones.
                    cross_spectrum.read_map(
                        mappaths, 
                        self.cosmology, 
                    )

                    self.full_transfer_function_interp

                    # Compute cross-spectrum for current FPXS combination
                    cross_spectrum.calculate_xs_2d(
                        no_of_k_bins=self.params.psx_number_of_k_bins + 1,
                    )

                    k_bin_centers_perp, k_bin_centers_par  = cross_spectrum.k[0]
                    
                    if not self.params.psx_generate_white_noise_sim:
                        # Run noise simulations to generate FPXS errorbar
                        
                        seed = self.params.psx_error_bar_seed
                        if seed is None:
                            t = time.time()
                            seed = int(np.round((t - np.floor(t)) * 1e4))

                        # self.params.psx_white_noise_transfer_function = transfer_function_wn

                        cross_spectrum.run_noise_sims_2d(
                            self.params.psx_noise_sim_number,
                            no_of_k_bins=self.params.psx_number_of_k_bins + 1,
                            seed = seed,
                        )

                    else:
                        cross_spectrum.xs *= transfer_function_wn
                        cross_spectrum.read_and_append_attribute(["rms_xs_mean_2D", "rms_xs_std_2D", "white_noise_covariance", "white_noise_simulation"], outdir_data)
                    
                    # Save resulting FPXS from current combination to file
                    cross_spectrum.make_h5_2d(outdir)
                
        # MPI barrier to prevent thread 0 from computing average FPXS before all individual combinations are finished.
        self.comm.Barrier()
        
        if self.rank == 0:
            # Compute average FPXS and finished data product plots
            print("\nComputing averages:")
            self.compute_averages()

    def compute_averages(self):
        if self.params.psx_mode == "saddlebag":
            average_spectrum_dir = os.path.join(self.power_spectrum_dir, "average_spectra_saddlebag")
        else:
            average_spectrum_dir = os.path.join(self.power_spectrum_dir, "average_spectra")

        fig_dir = os.path.join(self.power_spectrum_dir, "figs")

        if not os.path.exists(average_spectrum_dir):
            os.mkdir(average_spectrum_dir)
    
        if not os.path.exists(fig_dir):
            os.mkdir(fig_dir)

        if self.params.psx_mode == "saddlebag":
            if not os.path.exists(os.path.join(fig_dir, "chi2_grid_saddlebag")):
                os.mkdir(os.path.join(fig_dir, "chi2_grid_saddlebag"))

            if not os.path.exists(os.path.join(fig_dir, "average_spectra_saddlebag")):
                os.mkdir(os.path.join(fig_dir, "average_spectra_saddlebag"))
        else:
            if not os.path.exists(os.path.join(fig_dir, "chi2_grid")):
                os.mkdir(os.path.join(fig_dir, "chi2_grid"))

            if not os.path.exists(os.path.join(fig_dir, "average_spectra")):
                os.mkdir(os.path.join(fig_dir, "average_spectra"))
        
        N_feed = len(self.included_feeds)
        N_splits = len(self.split_map_combinations)
        N_k = self.params.psx_number_of_k_bins

        for map1, map2 in self.field_combinations:
            # Generate name of outpute data directory
            mapname1 = map1.split("/")[-1]
            mapname2 = map2.split("/")[-1]
            
            if self.params.psx_null_cross_field:
                indir = f"{mapname1[:-3]}_X_{mapname2[:-3]}"
            else:
                indir = f"{mapname1[:-3]}"
            
            mapname = f"{indir}"

            outdir = os.path.join(average_spectrum_dir, indir)
            if self.params.psx_generate_white_noise_sim:
                #outdir_data = f"{outdir}"
                outdir = f"{outdir}/white_noise_seed{self.params.psx_white_noise_sim_seed}"
                        
            if self.params.psx_generate_white_noise_sim:
                #indir_data = f"{indir}"
                indir = f"{indir}/white_noise_seed{self.params.psx_white_noise_sim_seed}"

            if not os.path.exists(outdir):
                os.mkdir(outdir)

            xs_mean = np.zeros((N_splits, N_k, N_k))
            xs_error = np.zeros((N_splits, N_k, N_k))
            xs_covariance = np.zeros((N_splits, N_k ** 2, N_k ** 2))
            xs_full_operator = np.zeros((N_splits, N_k ** 2, N_k ** 2))
            
            xs_wn_covariance = np.zeros((N_splits, N_k ** 2, N_k ** 2))
            xs_wn_mean = np.zeros((N_splits, self.params.psx_noise_sim_number, N_k, N_k))
            chi2_wn = np.zeros((N_splits, self.params.psx_noise_sim_number))
            chi2_wn_cov = np.zeros((N_splits, self.params.psx_noise_sim_number))

            xs_mean_1d = np.zeros((N_splits, N_k))
            xs_error_1d = np.zeros((N_splits, N_k))
            xs_wn_mean_1d = np.zeros((N_splits, self.params.psx_noise_sim_number, N_k))
            xs_wn_covariance_1d = np.zeros((N_splits, N_k, N_k))
            chi2_wn_1d = np.zeros((N_splits, self.params.psx_noise_sim_number))
            chi2_wn_cov_1d = np.zeros((N_splits, self.params.psx_noise_sim_number))



            chi2_grids = np.zeros((N_splits, N_feed, N_feed))
            
            if self.params.psx_null_diffmap:
                loaded_chi2_grids = np.zeros((N_splits, N_feed, N_feed))
            
            cross_variable_names = [] 

            if self.params.psx_null_diffmap:
                if len(self.params.psx_chi2_import_path) <= 0 or not os.path.exists(self.params.psx_chi2_import_path):
                    raise ValueError("No chi2 import file provided to perform null test chi2 cuts!")

                with h5py.File(self.params.psx_chi2_import_path, "r") as infile:
                    loaded_names = infile["cross_variable_names"][()].astype(str)
                    loaded_chi2_grid = infile["chi2_grid"][()]

            for i, splits in enumerate(self.split_map_combinations):
                if not self.params.psx_null_diffmap:
                    cross_variable = splits[0].split("/")[1]
                    cross_variable_names.append(cross_variable)
                else:
                    cross_variable = splits[0][0].split("/")[-1]
                    cross_variable = cross_variable[-5:-1]

                if self.params.psx_use_full_wn_covariance:
                    xs_sum = np.zeros(N_k ** 2)
                    xs_inv_cov = np.zeros((N_k ** 2, N_k ** 2))
                else:
                    wn_xs_sum = np.zeros((self.params.psx_noise_sim_number, N_k, N_k))
                    xs_sum = np.zeros((N_k, N_k))
                    xs_inv_var = np.zeros((N_k, N_k))
                
                chi2 = np.zeros((N_feed, N_feed))
                accepted_chi2 = np.zeros((N_feed, N_feed))

                # xs_mean_wn_sim, xs_wn_sim_error = self.average_wn_ensemble(splits, indir)
                # print(xs_mean_wn_sim.shape, xs_wn_sim_error.shape)
                # np.save(f"mean_wn_sim_cov_v2_{cross_variable}", xs_wn_sim_error)

                #sys.exit()
                
                for feed1 in range(N_feed):
                    for feed2 in range(N_feed):
                
                        cross_spectrum = xs_class.CrossSpectrum_nmaps(
                            self.params, 
                            splits, 
                            self.included_feeds[feed1], 
                            self.included_feeds[feed2],
                        )

                        cross_spectrum.read_spectrum(indir)
                        cross_spectrum.read_and_append_attribute(["white_noise_simulation"], indir)
                        #cross_spectrum.read_and_append_attribute(["rms_xs_mean_2D", "rms_xs_std_2D"], indir_data)

                        xs_wn = cross_spectrum.white_noise_simulation
                        xs = cross_spectrum.xs_2D
                        xs_sigma = cross_spectrum.rms_xs_std_2D

                        k_bin_centers_perp, k_bin_centers_par  = cross_spectrum.k
                        
                        k_bin_edges_par = cross_spectrum.k_bin_edges_par
                        
                        k_bin_edges_perp = cross_spectrum.k_bin_edges_perp

                        transfer_function = self.full_transfer_function_interp(k_bin_centers_perp, k_bin_centers_par)

                        # Applying white noise transfer function
                        transfer_function_wn = self.transfer_function_wn_interp(k_bin_centers_perp, k_bin_centers_par)
                        xs_sigma *= transfer_function_wn
                        xs_wn *= transfer_function_wn[None, ...]
                        
                        tf_cutoff = self.params.psx_tf_cutoff * np.nanmax(transfer_function[1:-1, 1:-1])

                        transfer_function_mask = np.logical_and(transfer_function > tf_cutoff, np.sign(transfer_function) >= 0) 

                        _transfer_function_mask = np.ones_like(transfer_function, dtype = bool)
                        # transfer_function_mask = np.ones_like(transfer_function, dtype = bool)
                        _transfer_function_mask[:4, :] = False
                        
                        transfer_function_mask = np.logical_and(transfer_function_mask, _transfer_function_mask)


                        chi3 = np.nansum(
                        (xs[transfer_function_mask] / xs_sigma[transfer_function_mask]) ** 3
                        )
                            
                        number_of_samples = np.sum(transfer_function_mask)

                        if np.all(xs / xs_sigma == 0) or np.all(~np.isfinite(xs / xs_sigma)):
                            chi2[feed1, feed2] = np.nan
                            continue
                        else:
                            chi2[feed1, feed2] = np.sign(chi3) * abs(
                                (np.nansum((xs[transfer_function_mask]  / xs_sigma[transfer_function_mask] ) ** 2) - number_of_samples)
                                / np.sqrt(2 * number_of_samples)
                            )
                        
                        
                        if self.params.psx_null_diffmap:
                            # If null test is run the same chi2 as in data run must be used!
                            current_cross_variable, = np.where(loaded_names == cross_variable)
                            accept_chi2 = np.abs(loaded_chi2_grid[current_cross_variable, feed1, feed2]) < self.params.psx_chi2_cut_limit
                            # accept_chi2 = np.abs(chi2[feed1, feed2]) < self.params.psx_chi2_cut_limit
                        else:
                            accept_chi2 = np.abs(chi2[feed1, feed2]) < self.params.psx_chi2_cut_limit
                        
                        accepted_chi2[feed1, feed2] = accept_chi2

                        if (np.isfinite(chi2[feed1, feed2]) and chi2[feed1, feed2] != 0) and feed1 != feed2:
                            if accept_chi2:
                                if self.params.psx_use_full_wn_covariance:
                                    xs = xs.flatten()

                                    cov = cross_spectrum.white_noise_covariance


                                    new_cov = np.zeros_like(cov)
                                    for d in range(0, 14 * 14, 14):
                                        new_cov += np.diag(cov.diagonal(d), d)
                                        new_cov += np.diag(cov.diagonal(-d), -d)
                                    cov = new_cov
                                                                        
                                    cov_inv = np.linalg.inv(cov)
                                    xs_sum += cov_inv @ xs 
                                    xs_inv_cov += cov_inv

                                else:
                                    xs_sum += xs / xs_sigma ** 2
                                    wn_xs_sum += xs_wn / xs_sigma[None, ...] ** 2
                                    xs_inv_var += 1 / xs_sigma ** 2
                                   
                if self.params.psx_null_diffmap:
                    print(f"{indir} {splits} \n# |chi^2| < {self.params.psx_chi2_cut_limit}:", np.sum(np.abs(loaded_chi2_grid[current_cross_variable, ...]) < self.params.psx_chi2_cut_limit))
                    # print(f"{indir} {splits} \n# |chi^2| < {self.params.psx_chi2_cut_limit}:", np.sum(np.abs(chi2) < self.params.psx_chi2_cut_limit))
                else:
                    print(f"{indir} {splits} \n# |chi^2| < {self.params.psx_chi2_cut_limit}:", np.sum(np.abs(chi2) < self.params.psx_chi2_cut_limit))

                if self.params.psx_use_full_wn_covariance:
                    xs_covariance[i, ...] = np.linalg.inv(xs_inv_cov)
                    xs_mean[i, ...] = (xs_covariance[i, ...] @ xs_sum).reshape(N_k, N_k)
                    xs_error[i, ...] = np.sqrt(xs_covariance[i, ...].diagonal().reshape(N_k, N_k))
                    xs_full_operator[i, ...] = xs_covariance[i, ...]
                else:
                    xs_mean[i, ...] = xs_sum / xs_inv_var
                    xs_error[i, ...] = 1.0 / np.sqrt(xs_inv_var)

                    xs_wn_mean[i, ...] = wn_xs_sum / xs_inv_var[None, ...]

                    xs_wn_covariance[i, ...] = np.cov(xs_wn_mean[i, ...].reshape(self.params.psx_noise_sim_number, N_k * N_k).T, ddof = 1)

                    # _error = np.sqrt(xs_wn_covariance[i, ...].diagonal())[None, transfer_function_mask.flatten()]

                    chi2_wn_data = xs_wn_mean[i, :, transfer_function_mask].T
                    chi2_wn[i, :] = np.nansum((chi2_wn_data  / xs_error[i, None, transfer_function_mask].T) ** 2, axis = 1)
                    # chi2_wn[i, :] = np.nansum((chi2_wn_data  / _error) ** 2, axis = 1)
                    
                    flattened_mask = transfer_function_mask.flatten()
                    flattened_chi2_wn_data = xs_wn_mean[i, :, ...].reshape(self.params.psx_noise_sim_number, N_k * N_k)
                    
                    # flattened_chi2_wn_data[:, ~flattened_mask] = 0
                    flattened_chi2_wn_data = flattened_chi2_wn_data[:, flattened_mask]
                    
                    cov_2d = xs_wn_covariance[i, flattened_mask, :]
                    cov_2d = cov_2d[:, flattened_mask]
                    
                    # inv_xs_wn_covariance = np.linalg.inv(xs_wn_covariance[i, ...])
                    inv_xs_wn_covariance = np.linalg.inv(cov_2d)


                    # inv_xs_wn_covariance[:, ~flattened_mask] = 0
                    # inv_xs_wn_covariance[~flattened_mask, :] = 0

                    for k in range(self.params.psx_noise_sim_number):
                        chi2_wn_cov[i, k] = flattened_chi2_wn_data[k, :].T @ inv_xs_wn_covariance @ flattened_chi2_wn_data[k, :]
                        # chi2_wn_cov[i, k] = flattened_chi2_wn_data[k, :].T @ _cov_inv @ flattened_chi2_wn_data[k, :]
                
                if np.all(~np.isfinite(chi2)):
                    continue

                weights = 1 / (xs_error[i, ...] / transfer_function) ** 2

                xs_1d = xs_mean[i, ...].copy()
                xs_wn_1d = xs_wn_mean[i, ...].copy()

                # transfer_function_mask = np.logical_and(transfer_function_mask, xs_error[i, ...] < xs_error[i, ...].max() * 0.5)

                weights[~transfer_function_mask] = 0.0
            
                xs_1d /= transfer_function
                xs_1d *= weights

                xs_wn_1d /= transfer_function[None, ...]
                xs_wn_1d *= weights[None, ...]

                k_bin_edges = np.logspace(-2.0, np.log10(1.5), len(k_bin_centers_perp) + 1)

                kgrid = np.sqrt(sum(ki**2 for ki in np.meshgrid(k_bin_centers_perp, k_bin_centers_par, indexing="ij")))


                Ck_nmodes_1d = np.histogram(
                    kgrid[kgrid > 0], bins=k_bin_edges, weights=xs_1d[kgrid > 0]
                )[0]
                inv_var_nmodes_1d = np.histogram(
                    kgrid[kgrid > 0], bins=k_bin_edges, weights=weights[kgrid > 0]
                )[0]
                nmodes_1d = np.histogram(kgrid[kgrid > 0], bins=k_bin_edges)[0]

                # Binning up the 2D noise spectra
                Ck_wn_nmodes_1d = np.zeros((self.params.psx_noise_sim_number, k_bin_edges.size - 1))
                for n in range(self.params.psx_noise_sim_number):
                    Ck_wn_nmodes_1d[n, :] = np.histogram(
                    kgrid[kgrid > 0], bins=k_bin_edges, weights=xs_wn_1d[n, kgrid > 0]
                )[0]

                # Ck = Ck_nmodes / nmodes
                k_1d = (k_bin_edges[1:] + k_bin_edges[:-1]) / 2.0
                
                Ck_1d = np.zeros_like(k_1d)
                Ck_wn_1d = np.zeros((self.params.psx_noise_sim_number, k_1d.size))
                rms_1d = np.zeros_like(k_1d)
                
                Ck_1d[np.where(nmodes_1d > 0)] = (
                    Ck_nmodes_1d[np.where(nmodes_1d > 0)]
                    / inv_var_nmodes_1d[np.where(nmodes_1d > 0)]
                )
                rms_1d[np.where(nmodes_1d > 0)] = np.sqrt(
                    1 / inv_var_nmodes_1d[np.where(nmodes_1d > 0)]
                )

                Ck_wn_1d[:, np.where(nmodes_1d > 0)] = (
                    Ck_wn_nmodes_1d[:, np.where(nmodes_1d > 0)]
                    / inv_var_nmodes_1d[None, np.where(nmodes_1d > 0)]
                )

                xs_error_1d[i, ...] = rms_1d

                xs_mean_1d[i, ...] = Ck_1d
                xs_wn_mean_1d[i, ...] = Ck_wn_1d
                xs_wn_covariance_1d[i, ...] = np.cov(Ck_wn_1d.T)

                mask = np.where(np.isfinite(Ck_1d / rms_1d))[0]
                cov_1d = xs_wn_covariance_1d[i, mask, :]
                cov_1d = cov_1d[:, mask]
                
                chi2_wn_data_1d = xs_wn_mean_1d[i, :, mask].T
                
                _error_1d = np.sqrt(cov_1d.diagonal())
                self._error_1d = _error_1d.copy()

                chi2_wn_1d[i, :] = np.nansum((chi2_wn_data_1d  / _error_1d) ** 2, axis = 1)
                # chi2_wn_1d[i, :] = np.nansum((chi2_wn_data_1d  / rms_1d[None, mask]) ** 2, axis = 1)

                inv_xs_wn_covariance_1d = np.linalg.inv(cov_1d)

                for k in range(self.params.psx_noise_sim_number):
                    chi2_wn_cov_1d[i, k] = chi2_wn_data_1d[k, :].T @ inv_xs_wn_covariance_1d @ chi2_wn_data_1d[k, :]

                # sys.exit()
                if not self.params.psx_generate_white_noise_sim:
                    if self.params.psx_mode == "saddlebag":
                        chi2_name = os.path.join(fig_dir, "chi2_grid_saddlebag")
                    else:
                        chi2_name = os.path.join(fig_dir, "chi2_grid")

                    chi2_name = os.path.join(chi2_name, indir)
                    chi2_name = chi2_name + f"_{self.params.psx_plot_name_suffix}"

                    if not os.path.exists(chi2_name):
                        os.mkdir(chi2_name)
                    
                    if self.params.psx_null_diffmap:
                        chi2_name = os.path.join(chi2_name, "null_diffmap")
                        if not os.path.exists(chi2_name):
                            os.mkdir(chi2_name)
                        
                        chi2_name = os.path.join(chi2_name, f"{cross_spectrum.null_variable}")
                        if not os.path.exists(chi2_name):
                            os.mkdir(chi2_name)
                    
                    chi2_grids[i, ...] = chi2
                    if self.params.psx_null_diffmap:
                        loaded_chi2_grids[i, ...] = loaded_chi2_grid[current_cross_variable, ...]

                    self.plot_chi2_grid(chi2, accepted_chi2, splits, chi2_name)

                    if self.params.psx_mode == "saddlebag":
                        average_name = os.path.join(fig_dir, "average_spectra_saddlebag")
                    else:
                        average_name = os.path.join(fig_dir, "average_spectra")
                    
                    average_name = os.path.join(average_name, indir)
                    

                    average_name = average_name + f"_{self.params.psx_plot_name_suffix}"
                    if not os.path.exists(average_name):
                        os.mkdir(average_name)

                    if self.params.psx_null_diffmap:
                        average_name = os.path.join(average_name, "null_diffmap")
                        if not os.path.exists(average_name):
                            os.mkdir(average_name)

                        average_name = os.path.join(average_name, f"{cross_spectrum.null_variable}")
                    
                    if not os.path.exists(average_name):
                        os.mkdir(average_name)
                else:
                    if self.params.psx_mode == "saddlebag":
                        average_name = os.path.join(fig_dir, "average_spectra")
                    else:
                        average_name = os.path.join(fig_dir, "average_spectra")

                    if not os.path.exists(average_name):
                        os.mkdir(average_name)
                    average_name = os.path.join(average_name, indir)
                    if not os.path.exists(average_name):
                        os.mkdir(average_name)

                    if self.params.psx_null_diffmap:
                        average_name = os.path.join(average_name, "null_diffmap")
                        if not os.path.exists(average_name):
                            os.mkdir(average_name)

                        average_name = os.path.join(average_name, f"{cross_spectrum.null_variable}")
                        if not os.path.exists(average_name):
                            os.mkdir(average_name)

                    
                    
                self.angle2Mpc = cross_spectrum.angle2Mpc
                self.map_dx = cross_spectrum.dx
                self.map_dy = cross_spectrum.dy
                self.map_dz = cross_spectrum.dz


                #if not self.params.psx_generate_white_noise_sim:
                self.plot_2D_mean(
                    k_bin_edges_par,
                    k_bin_edges_perp,
                    k_bin_centers_par,
                    k_bin_centers_perp,
                    xs_mean[i, ...],
                    xs_error[i, ...],
                    xs_wn_covariance[i, ...],
                    transfer_function_mask,
                    (chi2_wn[i, ...], chi2_wn_cov[i, ...]),
                    splits,
                    (mapname1, mapname2),
                    average_name,
                )

                if self.params.psx_null_diffmap:
                    plot_chi2 = loaded_chi2_grid[current_cross_variable, ...]                
                    # plot_chi2 = chi2                
                else:
                    plot_chi2 = chi2                
                
                self.plot_1D_mean(
                    k_1d,
                    xs_mean_1d[i, ...],
                    xs_error_1d[i, ...],
                    xs_wn_covariance_1d[i, ...],
                    plot_chi2,
                    (chi2_wn_1d[i, ...], chi2_wn_cov_1d[i, ...]),
                    splits,
                    (mapname1, mapname2),
                    average_name,
                )
            
            if not self.params.psx_null_diffmap:
                cross_variable_names = np.array(cross_variable_names, dtype = "S")
            else:
                cross_variable_names = np.array([*self.cross_variables], dtype = "S")

            with h5py.File(os.path.join(outdir, mapname + "_average_fpxs.h5"), "w") as outfile:
                outfile.create_dataset("k_1d", data = k_1d)             
                outfile.create_dataset("k_centers_par", data = k_bin_centers_par)
                outfile.create_dataset("k_centers_perp", data = k_bin_centers_perp)
                outfile.create_dataset("k_edges_par", data = k_bin_edges_par)      
                outfile.create_dataset("k_edges_perp", data = k_bin_edges_perp)     
                outfile.create_dataset("xs_mean_1d", data = xs_mean_1d)       
                outfile.create_dataset("xs_mean_2d", data = xs_mean)
                outfile.create_dataset("xs_sigma_1d", data = xs_error_1d)      
                outfile.create_dataset("xs_sigma_2d", data = xs_error)
                outfile.create_dataset("cross_variable_names", data = cross_variable_names)
                outfile.create_dataset("white_noise_covariance", data = xs_covariance)
                outfile.create_dataset("transfer_function_mask", data = transfer_function_mask)
                outfile.create_dataset("chi2_grid", data = chi2_grids)

                if not self.params.psx_null_diffmap:                
                    outfile.create_dataset("num_chi2_below_cutoff", data = np.sum(np.abs(chi2_grids) < self.params.psx_chi2_cut_limit, axis = (1, 2)))
                else:
                    outfile.create_dataset("num_chi2_below_cutoff", data = np.sum(np.abs(loaded_chi2_grids) < self.params.psx_chi2_cut_limit, axis = (1, 2)))
                    outfile.create_dataset("loaded_chi2_grid", data = loaded_chi2_grids)
                

                outfile.create_dataset("angle2Mpc", data = self.angle2Mpc)
                outfile.create_dataset("dx", data = self.map_dx)
                outfile.create_dataset("dy", data = self.map_dx)
                outfile.create_dataset("dz", data = self.map_dx)

                outfile["angle2Mpc"].attrs["unit"] = "Mpc/arcmin"
                outfile["dx"].attrs["unit"] = "Mpc"
                outfile["dy"].attrs["unit"] = "Mpc"
                outfile["dz"].attrs["unit"] = "Mpc"

                if self.params.psx_white_noise_sim_seed is not None:
                    outfile.create_dataset("white_noise_seed", data = self.params.psx_white_noise_sim_seed)

    def plot_2D_mean(self,
                    k_bin_edges_par: npt.NDArray,
                    k_bin_edges_perp: npt.NDArray,
                    k_bin_centers_par: npt.NDArray,
                    k_bin_centers_perp: npt.NDArray,
                    xs_mean: npt.NDArray,
                    xs_sigma: npt.NDArray,
                    cov_wn: npt.NDArray,
                    transfer_function_mask: npt.NDArray,
                    chi2_wn_list: list[npt.NDArray, npt.NDArray],
                    splits: Sequence[str],
                    fields: Sequence[str],
                    outname: str,
                    ):
        """Method that plots 1D, i.e. cylindrically averaged mean FPXS.

        Args:
            k_bin_edges_par (npt.NDArray): Array of k-bin edges of parallel (line-of-sight) dimension in 1/Mpc
            k_bin_edges_perp (npt.NDArray): Array of k-bin edges of perpendicular (angular, i.e.             k_bin_centers_par (npt.NDArray): Array of k-bin centers of parallel (line-of-sight) dimension in 1/Mpc
            k_bin_centers_perp (npt.NDArray): Array of k-bin centers of perpendicular (angular, i.e. perpendicular to line-of-sight) dimension in 1/Mpc.
            xs_mean (npt.NDArray): Array of mean spherically averaged FPXS 
            xs_sigma (npt.NDArray): Array of errors of mean spherically averaged FPXS
            transfer_function_mask (npt.NDArray): Array of bools marking where transfer function is below specified level.
            splits (Sequence[str]): Sequence of strings with split names used for as cross-spectrum variable
            fields (Sequence[str]): Sequence of strings with field names used in cross correlation
            outname (str): String with output directory for plot
        """

        from scipy.stats import chi2, norm

        matplotlib.rcParams["xtick.labelsize"] = 16
        matplotlib.rcParams["ytick.labelsize"] = 16
        matplotlib.rcParams["hatch.color"] = "gray"
        matplotlib.rcParams["hatch.linewidth"] = 0.3

        if self.params.psx_null_diffmap:
            try:
                split1 = splits[0][0].split("map_")[-1][5:]
                split2 = splits[1][0].split("map_")[-1][5:]
            except:
                split1 = ""
                split2 = ""
        else:
            try:
                split1 = splits[0].split("map_")[-1]
                split2 = splits[1].split("map_")[-1]
            except:
                split1 = ""
                split2 = ""

        outname_corr = os.path.join(
            outname, 
            f"corr_2d_{split1}_X_{split2}.png"
            )

        outname = os.path.join(
            outname, 
            f"xs_mean_2d_{split1}_X_{split2}.png"
        )

        
        # fig, ax = plt.subplots(1, 3, figsize=(16, 5.6))
        fig, ax = plt.subplots(2, 3, figsize=(16, 13))

        fig.suptitle(f"Fields: {fields[0]} X {fields[1]} | {split1} X {split2}", fontsize=16)
        
        limit_idx = int(np.round(3 / 14 * xs_mean.shape[0])) 

        if limit_idx != 0:
            lim = np.nanmax(np.abs(xs_mean[limit_idx:-limit_idx, limit_idx:-limit_idx]))
            lim_error = np.nanmax(xs_sigma[limit_idx:-limit_idx, limit_idx:-limit_idx])
            lim_significance = np.nanmax(np.abs((xs_mean / xs_sigma)[limit_idx:-limit_idx, limit_idx:-limit_idx]))
        else:
            lim = np.nanmax(np.abs(xs_mean))
            lim_error = np.nanmax(xs_sigma)
            lim_significance = np.nanmax(np.abs((xs_mean / xs_sigma)))

        norm = matplotlib.colors.Normalize(vmin=-1.1 * lim, vmax=1.1 * lim)
        lim_error = matplotlib.colors.Normalize(vmin=0, vmax=lim_error)
        lim_significance = matplotlib.colors.Normalize(vmin=-lim_significance, vmax=lim_significance)
        
        k_contour_levels = np.logspace(-2.0, np.log10(1.5), len(k_bin_centers_perp) + 1)
        k_1d = np.logspace(-2.0, np.log10(1.5), 1000)
        K_x, K_y = np.meshgrid(k_1d, k_1d)
        for i in range(3):
            ax[0, i].contour(K_x, K_y, np.sqrt(K_x ** 2 + K_y ** 2), levels = k_contour_levels, colors = "gray", alpha = 0.4, zorder = 4)
        
        X_perp, X_par = np.meshgrid(k_bin_centers_perp, k_bin_centers_par)

        img1 = ax[0, 0].pcolormesh(
            X_par,
            X_perp, 
            xs_mean.T,
            cmap="RdBu_r",
            norm=norm,
            rasterized=True,
            zorder = 1,
        )

        ax[0, 0].fill_between(
            [0, 1], 
            0, 
            1, 
            hatch='xxxx', 
            transform = ax[0, 0].transAxes, 
            alpha = 0, 
            zorder = 2
        )

        xs_mean_masked = np.ma.masked_where(~transfer_function_mask, xs_mean)
        ax[0, 0].pcolormesh(
            X_par,
            X_perp,
            xs_mean_masked.T,
            cmap="RdBu_r",
            norm=norm,
            rasterized=True,
            zorder = 3,
        )

        cbar = fig.colorbar(img1, ax=ax[0, 0], fraction=0.046, pad=0.18, location = "bottom")
        cbar.set_label(
            r"$\tilde{C}\left(k_{\bot},k_{\parallel}\right)$ [$\mu$K${}^2$ (Mpc)${}^3$]",
            size=16,
        )
        cbar.ax.tick_params(rotation=45)

        ###############################

        img2 = ax[0, 1].pcolormesh(
            X_par,
            X_perp, 
            xs_sigma.T,
            cmap = "CMRmap",
            norm=lim_error,
            rasterized=True,
            zorder = 1,
        )

        ax[0, 1].fill_between(
            [0, 1], 
            0, 
            1, 
            hatch='xxxx', 
            transform = ax[0, 1].transAxes, 
            alpha = 0, 
            zorder = 2
        )

        xs_sigma_masked = np.ma.masked_where(~transfer_function_mask, xs_sigma)
        ax[0, 1].pcolormesh(
            X_par,
            X_perp,
            xs_sigma_masked.T,
            cmap = "CMRmap",
            norm=lim_error,
            rasterized=True,
            zorder = 3,
        )

        cbar = fig.colorbar(img2, ax=ax[0, 1], fraction=0.046, pad=0.18, location = "bottom")
        cbar.set_label(
            r"$\sigma\left(k_{\bot},k_{\parallel}\right)$[$\mu$K$^2$ (Mpc)${}^3$]",
            size=16,
        )
        cbar.ax.tick_params(rotation=45)

        ###############################

        img3 = ax[0, 2].pcolormesh(
            X_par,
            X_perp, 
            (xs_mean / xs_sigma).T,
            cmap="RdBu_r",
            norm=lim_significance,
            rasterized=True,
            zorder = 1,
        )

        ax[0, 2].fill_between(
            [0, 1], 
            0, 
            1, 
            hatch='xxxx', 
            transform = ax[0, 2].transAxes, 
            alpha = 0, 
            zorder = 2
        )

        ax[0, 2].pcolormesh(
            X_par,
            X_perp,
            (xs_mean_masked / xs_sigma_masked).T,
            cmap="RdBu_r",
            norm=lim_significance,
            rasterized=True,
            zorder = 3,
        )

        cbar = fig.colorbar(img3, ax=ax[0, 2], fraction=0.046, pad=0.18, location = "bottom")
        cbar.set_label(
            r"$\tilde{C}/\sigma\left(k_{\bot},k_{\parallel}\right)$",
            size=16,
        )
        cbar.ax.tick_params(rotation=45)
       
        for i in range(3):
            ax[0, i].set_xscale("log")
            ax[0, i].set_yscale("log")

            ticks = [0.03, 0.1, 0.3, 1]
            ticklabels = ["0.03", "0.1", "0.3", "1"]

            ax[0, i].set_xticks(ticks)
            ax[0, i].set_xticklabels(ticklabels)
            ax[0, i].set_yticks(ticks)
            ax[0, i].set_yticklabels(ticklabels)

            ax[0, i].set_xlim(k_bin_edges_par[0], k_bin_edges_par[-1])
            ax[0, i].set_ylim(k_bin_edges_perp[0], k_bin_edges_perp[-1])


            ax[0, i].set_xlabel(r"$k_\parallel$ [Mpc${}^{-1}$]", fontsize=16)
        
        ax[0, 0].set_ylabel(r"$k_\bot$ [Mpc${}^{-1}$]", fontsize=16)


        majorticks = [0.03, 0.1, 0.3, 1]

        ax2 = ax[0, 2].twinx()
        ax2.set_yscale("log")

        ax2.set_yticks(majorticks)
        ax2.set_yticklabels(majorticks)
        
        ax2.set_ylim(k_bin_edges_perp[0], k_bin_edges_perp[-1])
        ax2.set_yticklabels(np.round(2 * np.pi / (np.array(majorticks) * self.angle2Mpc), 2).astype(str))
        ax2.set_ylabel(r"angular scale [$\mathrm{arcmin}$]", fontsize = 16)

        ###############################
        # chi2 of white noise spectra #
        ###############################
        ax[1, 0].axis("off")
        ax[1, 1].axis("off")
        ax[1, 2].axis("off")

        ax1 = fig.add_subplot(212)

        x_lim = [0, 200]
        x = np.linspace(x_lim[0], x_lim[1], 1000)
        ax1.hist(
            chi2_wn_list[0], 
            histtype = "step", 
            bins = int(np.round(chi2_wn_list[0].size * 0.05)), 
            density = True, 
            lw = 3,
            label = r"$\sum_i (d_i/\sigma_i)^2$",
        )
        
        ax1.hist(
            chi2_wn_list[1],
            histtype = "step",
            bins = int(np.round(chi2_wn_list[1].size * 0.05)),
            density = True,
            lw = 3,
            label = r"$\mathbf{d}^T\mathbf{N}^{-1}\mathbf{d}$",
            )
        
        chi2_analytical = chi2.pdf(x, df = np.sum(transfer_function_mask))
        
        ax1.plot(
            x,
            chi2_analytical,
            color = "r",
            linestyle = "dashed",
            lw = 3,
            label = rf"$\chi^2(dof = {np.sum(transfer_function_mask)})$",
        )
        

        chi2_sum = np.nansum((xs_mean_masked / xs_sigma_masked) ** 2)

        if chi2_sum >= x_lim[0] and chi2_sum <= x_lim[1]:
            ax1.axvline(
                chi2_sum, 
                label = r"$\chi^2_\mathrm{data}$",
                linestyle = "dashed",
                lw = 3,
                color = "k",
            )
        elif chi2_sum < x_lim[0]:
            ax1.arrow(
                x_lim[0] + 50,
                chi2_analytical.max() / 2,
                -45,
                0,
                width = 0.0005,
                head_length = 5,
                label = r"$\chi^2_\mathrm{data}$",
                color = "k",
            )
        elif chi2_sum > x_lim[1]:
            ax1.arrow(
                x_lim[1] - 50,
                chi2_analytical.max() / 2,
                45,
                0,
                width = 0.0005,
                head_length = 5,
                label = r"$\chi^2_\mathrm{data}$",
                color = "k",
            )

        ax1.legend(fontsize = 16, ncol = 4)

        ax1.set_xlabel(r"$\chi^2$", fontsize = 16)
        ax1.set_ylabel(r"$P(\chi^2)$", fontsize = 16)
        ax1.set_xlim(x_lim)
        # ax1.set_yscale("log")

        fig.tight_layout()
        fig.savefig(outname, bbox_inches = "tight")
        

        fig, ax = plt.subplots(figsize = (10, 8))
        img = ax.imshow(
            cov_wn / np.sqrt(np.outer(cov_wn.diagonal(), cov_wn.diagonal(),)),
            interpolation = "none",
            vmin = -1,
            vmax = 1,
            cmap = "bwr",
        )
        cbar = fig.colorbar(img, ax=ax, fraction=0.046, pad=0.018)
        cbar.set_label(
            r"bin correlation",
            size=16,
        )
        
        ax.set_xlabel("bin number", fontsize = 16)
        ax.set_ylabel("bin number", fontsize = 16)
        
        fig.savefig(outname_corr, bbox_inches = "tight")


    def plot_1D_mean(self,
                    k_1d: npt.NDArray,
                    xs_mean: npt.NDArray,
                    xs_sigma: npt.NDArray,
                    cov_wn: npt.NDArray,
                    chi2_grid: npt.NDArray,
                    chi2_wn_list: list[npt.NDArray, npt.NDArray],
                    splits: Sequence[str],
                    fields: Sequence[str],
                    outname: str
                    ):
        
        """Method that plots 1D, i.e. spherically averaged mean FPXS.

        Args:
            k_1d (npt.NDArray): Array of k-bin centers in 1/Mpc
            xs_mean (npt.NDArray): Array of mean spherically averaged FPXS 
            xs_sigma (npt.NDArray): Array of errors of mean spherically averaged FPXS
            chi2_grid (npt.NDArray): Array of errors of normalized chi2 values for each feed-split combo
            splits (Sequence[str]): Sequence of strings with split names used for as cross-spectrum variable
            fields (Sequence[str]): Sequence of strings with field names used in cross correlation
            outname (str): String with output directory for plot
        """

        from scipy.stats import chi2, norm

        # Define default tick label fontsize
        matplotlib.rcParams["xtick.labelsize"] = 16
        matplotlib.rcParams["ytick.labelsize"] = 16

        # Define the two split names that were cross-correlated
        if self.params.psx_null_diffmap:
            try:
                split1 = splits[0][0].split("map_")[-1][5:]
                split2 = splits[1][0].split("map_")[-1][5:]
            except:
                split1 = ""
                split2 = ""
        else:
            try:
                split1 = splits[0].split("map_")[-1]
                split2 = splits[1].split("map_")[-1]
            except:
                split1 = ""
                split2 = ""

        outname_corr = os.path.join(
            outname, 
            f"corr_1d_{split1}_X_{split2}.png"
        )
        # Add output name to output path
        outname = os.path.join(
            outname, 
            f"xs_mean_1d_{split1}_X_{split2}.png"
            )
        
        # Plot spherically averaged mean FPXS
        fig, ax = plt.subplots(3, 1, figsize=(16, 15))

        # Figure title
        fig.suptitle(f"Fields: {fields[0]} X {fields[1]} | {split1} X {split2}", fontsize=16)
        
        # Only want to use points between 0.04 and 1.0 /Mpc
        where = np.logical_and(k_1d > 0.1, k_1d < 1.0)
        
        # Plot y-limits
        lim = np.nanmax(np.abs((k_1d * (xs_mean + xs_sigma))[where]))
        lim = np.nanmax((np.nanmax(np.abs((k_1d * (xs_mean - xs_sigma))[where])), lim))
        
        # lim = 2e4

        lim_significance = 2 * np.nanmax(np.abs((xs_mean / xs_sigma)[where]))
        
        mask = np.isfinite(xs_mean / xs_sigma)

        k_1d = k_1d[mask]
        xs_mean = xs_mean[mask]
        xs_sigma = xs_sigma[mask]

        # Plot scatter and error bars
        ax[0].scatter(
            k_1d,
            k_1d * xs_mean,
            s = 80,
        )

        ax[0].errorbar(
            k_1d,
            k_1d * xs_mean,
            k_1d * xs_sigma,
            lw = 3,
            fmt = " ",
        )

        ax[0].scatter(
            k_1d - k_1d * 0.05,
            k_1d * xs_mean,
            s = 80,
            color = "r",
        )

        ax[0].errorbar(
            k_1d - k_1d * 0.05,
            k_1d * xs_mean,
            k_1d * self._error_1d,
            lw = 3,
            fmt = " ",
            color = "r",
        )


        if np.isfinite(lim):
            ax[0].set_ylim(-lim, lim)
        ax[0].set_ylabel(
            r"$k\tilde{C}(k)$ [$\mu$K${}^2$ (Mpc)${}^3$]",
            fontsize = 16,
        )
        
        chi2_sum = np.nansum((xs_mean / xs_sigma) ** 2)
        
        chi2_cdf = chi2.cdf(chi2_sum, df = np.sum(np.isfinite((xs_mean))))

        PTE = chi2.sf(chi2_sum, df = np.sum(np.isfinite((xs_mean))))

        number_accepted_cross_spectra = np.sum(np.abs(chi2_grid) < self.params.psx_chi2_cut_limit)

        ax[0].set_title(rf"# accepted $\chi^2 < {self.params.psx_chi2_cut_limit}$: {number_accepted_cross_spectra} / {chi2_grid.size}" + " " * 5 + rf"$\chi^2 = \sum_i (d_i/\sigma_i)^2$: {chi2_sum:.3f}" + " " * 5 + f"dof: {np.sum(np.isfinite((xs_mean)))}" + " " * 5 + rf"$\chi^2$ cdf: {chi2_cdf:.3f}" + " " * 5 + rf"PTE: {PTE:.3f}", fontsize = 16)

        # Plot scatter and error bar of significance plot
        ax[1].scatter(
            k_1d,
            xs_mean / xs_sigma,
            s = 80,
        )

        ax[1].errorbar(
            k_1d,
            xs_mean / xs_sigma,
            1,                  # NOTE: that in significance units error bar always has length 1!
            lw = 3,
            fmt = " ",
        )

        ax[1].scatter(
            k_1d - k_1d * 0.05,
            xs_mean / self._error_1d,
            s = 80,
            color = "r",
        )
        
        ax[1].errorbar(
            k_1d - k_1d * 0.05,
            xs_mean / self._error_1d,
            1,
            lw = 3,
            fmt = " ",
            color = "r",
        )
        
        # Add some zero line
        ax[0].axhline(0, color = "k", alpha = 0.5)
        ax[1].axhline(0, color = "k", alpha = 0.5)

        if np.isfinite(lim_significance):
            ax[1].set_ylim(-lim_significance, lim_significance)
    
        ax[1].set_ylabel(
            r"$\tilde{C} / \sigma_\tilde{C}(k)$",
            fontsize = 16,
        )

        # Logarithmic x-axis
        ax[0].set_xscale("log")
        ax[1].set_xscale("log")

        # Define ticks and ticklabels
        klabels = [0.05, 0.1, 0.2, 0.5, 1.0]

        ax[0].set_xticks(klabels)
        ax[0].set_xticklabels(klabels, fontsize = 16)
        ax[1].set_xticks(klabels)
        ax[1].set_xticklabels(klabels, fontsize = 16)

        ax[0].set_xlim(0.06, 1.0)
        ax[1].set_xlim(0.06, 1.0)

        # ax[0].set_xlim(0.1, 1.0)
        # ax[1].set_xlim(0.1, 1.0)

        ax[1].set_xlabel(r"$k [\mathrm{Mpc}^{-1}]$", fontsize = 16)

        # Enable grid in plot
        ax[0].grid(True)
        ax[1].grid(True)

        ###############################
        # chi2 of white noise spectra #
        ###############################
        x_lim = [0, 40]
        x = np.linspace(x_lim[0], x_lim[1], 1000)
        ax[2].hist(
            chi2_wn_list[0], 
            histtype = "step", 
            bins = int(np.round(chi2_wn_list[0].size * 0.05)), 
            density = True, 
            lw = 3,
            label = r"$\sum_i (d_i/\sigma_i)^2$",
        )
        
        ax[2].hist(
            chi2_wn_list[1],
            histtype = "step",
            bins = int(np.round(chi2_wn_list[1].size * 0.05)),
            density = True,
            lw = 3,
            label = r"$\mathbf{d}^T\mathbf{N}^{-1}\mathbf{d}$",
            )

        chi2_analytical = chi2.pdf(x, df = np.sum(np.isfinite(xs_mean / xs_sigma)))
        ax[2].plot(
            x,
            chi2_analytical,
            color = "r",
            linestyle = "dashed",
            lw = 3,
            label = rf"$\chi^2(dof = {np.sum(np.isfinite(xs_mean / xs_sigma))})$",
        )
        
        if chi2_sum >= x_lim[0] and chi2_sum <= x_lim[1]:
            ax[2].axvline(
                chi2_sum, 
                label = r"$\chi^2_\mathrm{data}$",
                linestyle = "dashed",
                lw = 3,
                color = "k",
            )
        elif chi2_sum < x_lim[0]:
            ax[2].arrow(
                x_lim[0] + 5,
                chi2_analytical.max() / 2,
                -4.5,
                0,
                width = 0.002,
                head_length = 0.5,
                label = r"$\chi^2_\mathrm{data}$",
                color = "k",
            )
        elif chi2_sum > x_lim[1]:
            ax[2].arrow(
                x_lim[1] - 5,
                chi2_analytical.max() / 2,
                4.5,
                0,
                width = 0.002,
                head_length = 0.5,
                label = r"$\chi^2_\mathrm{data}$",
                color = "k",
            )

        ax[2].legend(fontsize = 16, ncol = 4)

        ax[2].set_xlabel(r"$\chi^2$", fontsize = 16)
        ax[2].set_ylabel(r"$P(\chi^2)$", fontsize = 16)
        ax[2].set_xlim(0, 40)

        fig.savefig(outname, bbox_inches = "tight")

        fig, ax = plt.subplots(figsize = (10, 8))
        img = ax.imshow(
            cov_wn / np.sqrt(np.outer(cov_wn.diagonal(), cov_wn.diagonal(),)),
            interpolation = "none",
            vmin = -1,
            vmax = 1,
            cmap = "bwr",
        )
        cbar = fig.colorbar(img, ax=ax, fraction=0.046, pad=0.018)
        cbar.set_label(
            r"bin correlation",
            size=16,
        )
        
        ax.set_xlabel("bin number", fontsize = 16)
        ax.set_ylabel("bin number", fontsize = 16)
        
        fig.savefig(outname_corr, bbox_inches = "tight")

    def plot_chi2_grid(self, chi2: npt.NDArray, chi2_mask: npt.NDArray, splits: tuple, outname: str):
        """Method that plots feed-split grid of FPXS chi-squared values

        Args:
            chi2 (npt.NDArray): Array of chi2 values for all feed and split combinations
            chi2 (npt.NDArray): Array of chi2 mask values for all feed and split combinations
            splits (tuple): Split names that were crossed
            outname (str): Plot file output path
        """

        # Define default ticks label sizes
        matplotlib.rcParams["xtick.labelsize"] = 10
        matplotlib.rcParams["ytick.labelsize"] = 10
        matplotlib.rcParams["hatch.color"] = "gray"
        matplotlib.rcParams["hatch.linewidth"] = 0.3
        # Define the two split names that were cross-correlated
        if self.params.psx_null_diffmap:
            split1 = splits[0][0].split("map_")[-1][5:]
            split2 = splits[1][0].split("map_")[-1][5:]
        else:
            split1 = splits[0].split("map_")[-1]
            split2 = splits[1].split("map_")[-1]

        # Add output name to output path
        outname = os.path.join(
            outname, 
            f"xs_chi2_grid_{split1}_X_{split2}.png"
            )
        
        # Number of detectors used in cross-correlation combinations
        N_feed = len(self.included_feeds)

        # Define symetric collormap
        cmap = matplotlib.cm.RdBu.reversed()

        # Bad values, i.e. NaN and Inf, are set to black
        cmap.set_bad("k", 1)
        

        lim = np.nanmin((100, np.nanmax(np.abs(chi2))))
        norm = matplotlib.colors.Normalize(vmin=-1.2 * lim, vmax=1.2 * lim)
        norm_chi2_cut = matplotlib.colors.Normalize(vmin=-self.params.psx_chi2_cut_limit, vmax=self.params.psx_chi2_cut_limit)

        # Plot chi2 value grid
        fig, ax = plt.subplots(1, 2, figsize = (13, 5))

        img = ax[0].imshow(
            chi2,
            interpolation = "none",
            norm=norm,
            extent = (0.5, N_feed + 0.5, N_feed + 0.5, 0.5),
            cmap = cmap,
            rasterized = True,
            zorder = 1,
        )

        ax[0].fill_between(
            [0.5, N_feed + 0.5], 
            0.5, 
            N_feed + 0.5, 
            hatch='xxxx', 
            alpha = 0, 
            zorder = 2
        )
        

        # chi2_masked = np.ma.masked_where(~(np.abs(chi2) < self.params.psx_chi2_cut_limit), chi2)

        chi2_masked = np.ma.masked_where(~chi2_mask.astype(bool), chi2)

        ax[0].imshow(
            chi2_masked,
            interpolation="none",
            extent=(0.5, N_feed + 0.5, N_feed + 0.5, 0.5),
            cmap="RdBu_r",
            norm=norm,
            rasterized=True,
            zorder = 3,
        )


        new_tick_locations = np.array(range(N_feed)) + 1
        ax[0].set_xticks(new_tick_locations)
        ax[0].set_yticks(new_tick_locations)
        
        ax[0].set_xlabel(f"Feed of {split1}")
        ax[0].set_ylabel(f"Feed of {split2}")
        
        cbar = plt.colorbar(img, ax = ax[0])
        
        cbar.set_label(r"$|\chi^2| \times$ sign($\chi^3$)")

        ####################################################

        ax[1].set_title(r"Colorbar zoomed in on $\chi^2$ cut limit")
        img2 = ax[1].imshow(
            chi2,
            interpolation = "none",
            norm=norm_chi2_cut,
            extent = (0.5, N_feed + 0.5, N_feed + 0.5, 0.5),
            cmap = cmap,
            rasterized = True,
            zorder = 1,
        )

        ax[1].fill_between(
            [0.5, N_feed + 0.5], 
            0.5, 
            N_feed + 0.5, 
            hatch='xxxx', 
            alpha = 0, 
            zorder = 2
        )

        # chi2_masked = np.ma.masked_where(~(np.abs(chi2) < self.params.psx_chi2_cut_limit), chi2)

        chi2_masked = np.ma.masked_where(~chi2_mask.astype(bool), chi2)
        
        ax[1].imshow(
            chi2_masked,
            interpolation="none",
            extent=(0.5, N_feed + 0.5, N_feed + 0.5, 0.5),
            cmap="RdBu_r",
            norm=norm_chi2_cut,
            rasterized=True,
            zorder = 3,
        )


        new_tick_locations = np.array(range(N_feed)) + 1
        ax[1].set_xticks(new_tick_locations)
        ax[1].set_yticks(new_tick_locations)
        
        ax[1].set_xlabel(f"Feed of {split1}")
        ax[1].set_ylabel(f"Feed of {split2}")
        
        cbar = plt.colorbar(img2, ax = ax[1])
        
        cbar.set_label(r"$|\chi^2| \times$ sign($\chi^3$)")
        
        fig.savefig(outname, bbox_inches="tight")

    def read_params(self):
        """Method reading and parsing the parameters from file or command line.

        Raises:
            ValueError: If no power spectrum output directory is provided
            ValueError: If no COMAP mapmaker map name is provided
            ValueError: If no COMAP mapmaker map directory is provided 
            ValueError: If no jackknife "split" definition file, that defines what is the primary split, secondary split and cross-spectrum variable.
        """
        from l2gen_argparser import parser

        # Read parameter file/command line arguments
        self.params = parser.parse_args()

        # Define often used parameters as class attributes

        # Power spectrum output directory
        self.power_spectrum_dir = self.params.power_spectrum_dir
        
        # Mapmaker map name and directory
        self.map_name = self.params.map_name + self.params.psx_map_name_postfix
        self.map_dir = self.params.map_dir

        # Jackknive "split" definition file path, defining which splits to use 
        self.jk_def_file = self.params.jk_def_file

        # Raising errors if required parameters are missing
        if self.power_spectrum_dir is None:
            raise ValueError(
                "A power spectrum data directory must be specified in parameter file or terminal."
            )
        if self.map_name is None:
            raise ValueError(
                "A map file name must be specified in parameter file or terminal."
            )
        if self.map_dir is None:
            raise ValueError(
                "A map data directory must be specified in parameter file or terminal."
            )
        if self.jk_def_file is None:
            raise ValueError(
                "Please specify a jk_def_file in parameter file or terminal."
            )
            

    def read_cosmology(self):
        """
        Method that reads in the standard cosmology to use form pickled astropy object.
        """
        
        cosmology_path = os.path.join(self.params.phy_cosmology_dir, self.params.phy_cosmology_name)

        with open(cosmology_path, mode="rb") as file:
            self.cosmology = pickle.load(file)

    def read_jackknife_definition_file(self):
        """Method that reads the jackknife/split definition file and outputs definition of split variables.
        """

        with open(self.params.jk_def_file, "r") as jk_file:
            all_lines = jk_file.readlines()

        # skip the first two lines (number of different jk and accr)
        all_lines = all_lines[
            2:
        ]  
        
        # Variables marked with "2" in jk_def_file. 
        # The first split in heirarchy/split-tree
        primary_variables = []  

        # Variables marked with "3" in jk_def_file.
        # The secondary successive splits in herarchy/split-tree
        secondary_variables = []  

        # The variable over which to perform cross-spectra. 
        # Marked with "1" in jk_def_file
        cross_variables = []  # extra 1

        # List of all split variables
        all_variables = []
        
        # Read and sort split variables to respective lists
        for line in all_lines:
            line = line.split("#")[0].strip()
            line = line.split("$")[0].strip()
            split_line = line.split()
            variable = split_line[0]
            number = split_line[1]

            extra = split_line[-1]

            all_variables.append(variable)

            if number == "3":
                secondary_variables.append(variable)

            if number == "2":
                primary_variables.append(variable)

            if extra == "1":
                cross_variables.append(variable)

        # Find all feed-feed variables that are also primary variables or secondary variables
        cross_and_primary = []
        cross_and_secondary = []

        for variable in all_variables:
            if variable in primary_variables and variable in cross_variables:
                cross_and_primary.append(variable)
            if variable in cross_variables and variable in secondary_variables:
                cross_and_secondary.append(variable)

        
        # Define as class attributes
        self.primary_variables  = primary_variables 
        self.secondary_variables  = secondary_variables 
        self.cross_variables  = cross_variables 
        self.all_variables  = all_variables 
        self.cross_and_primary  = cross_and_primary 
        self.cross_and_secondary  = cross_and_secondary 
    
    def generate_split_map_names(self):  
        primary_variables = self.primary_variables
        secondary_variables = self.secondary_variables
        
        cross_and_primary = self.cross_and_primary
        cross_and_secondary = self.cross_and_secondary
        
        split_map_combinations = []

        # If some primary variables are simultaneously feed-feed variables
        if (len(cross_and_primary) != 0):  
            
            number_of_secondary_variables = len(secondary_variables)  
            
            # Generate indices for all split combinations
            combinations = list(itertools.product(range(self.params.split_base_number), repeat = number_of_secondary_variables))  
            for primary_variable in cross_and_primary:

                # Generating names of split combinations
                for combo in combinations:
                    name = ""
                    for i, bin_number in enumerate(combo):
                        name = name + f"{secondary_variables[i]}{bin_number}" 

                    split_map_combinations.append(
                        (f"multisplits/{primary_variable}/map_{primary_variable}{0}{name}",
                         f"multisplits/{primary_variable}/map_{primary_variable}{1}{name}",)
                    )
        else:
            import re
            number_of_secondary_variables = len(secondary_variables)  
            
            # Generate indices for all split combinations

            combinations = list(itertools.combinations(range(self.params.split_base_number), r = 2))  
            secondary_combinations = list(itertools.combinations(range(self.params.split_base_number), r = 2 * len(cross_and_secondary)))  

            secondary_permutations = list(itertools.product(range(self.params.split_base_number), repeat = len(secondary_variables)))
            
            secondary_cross_combos = []
            for permutation in secondary_permutations:
                secondary_name = ""
                for i, secondary_variable in enumerate(secondary_variables):
                    secondary_name += f"{secondary_variable}{permutation[i]}"

                name1 = secondary_name
                name2 = secondary_name
                for combo in secondary_combinations:
                    for cross_variable in cross_and_secondary:

                        name1 = re.sub(rf"{cross_variable}\d", f"{cross_variable}{combo[0]}", name1)
                        name2 = re.sub(rf"{cross_variable}\d", f"{cross_variable}{combo[1]}", name2)
                    if not ((name1, name2) in secondary_cross_combos):
                        secondary_cross_combos.append((name1, name2))

            for primary_variable in primary_variables:
                # Generating names of split combinations
                for primary_bin_number in range(self.params.split_base_number):
                    if self.params.psx_null_diffmap:
                        ####################################
                        # GENERALIZE TO MORE THAN 2 SPLIT BINS LATER
                        ####################################
                        
                        null_name1 = f"multisplits/{primary_variable}/map_{primary_variable}{0}"
                        null_name2 = f"multisplits/{primary_variable}/map_{primary_variable}{1}"

                        for secondary_cross_combo in secondary_cross_combos:
                         
                            secondary_name1, secondary_name2 = secondary_cross_combo
                            
                            split_name11 =  f"{null_name1}{secondary_name1}"
                            split_name22 =  f"{null_name2}{secondary_name2}"
                            split_name12 =  f"{null_name1}{secondary_name2}"
                            split_name21 =  f"{null_name2}{secondary_name1}"

                            split_names = (
                                (split_name11, split_name21),
                                (split_name12, split_name22),
                            )
                            if split_names not in split_map_combinations:
                                split_map_combinations.append(
                                    split_names
                                )
                            
                    else:
                        primary_name = f"{primary_variable}{primary_bin_number}"
                        primary_name = f"multisplits/{primary_variable}/map_{primary_name}"
                    
                        for secondary_cross_combo in secondary_cross_combos:
                            secondary_name1, secondary_name2 = secondary_cross_combo

                            split_name1 =  f"{primary_name}{secondary_name1}"
                            split_name2 =  f"{primary_name}{secondary_name2}"
                            split_map_combinations.append((
                                split_name1,
                                split_name2,
                            ))


    
        self.split_map_combinations = split_map_combinations

    def log2lin(self, x: npt.NDArray, k_edges: npt.NDArray) -> npt.NDArray:
        """Method that converts from logarithmic to linear spaced k-bin edges.

        Args:
            x (npt.NDArray): Input array of k-ticks
            k_edges (npt.NDArray): k-bin edges

        Returns:
            npt.NDArray: Linear spaced bin center k-ticks
        """
        
        loglen = np.log10(k_edges[-1]) - np.log10(k_edges[0])
        logx = np.log10(x) - np.log10(k_edges[0])
        return logx / loglen
    
    def define_transfer_function(self):
        """Method to define full transfer function to be used. Reads in all specified
        transfer functions, multiplies them and defines an interpolation function.
        """

        transfer_function_dir = os.path.join(current, "transfer_functions")
        
        # By if now other transfer function is specified use an empty one, i.e. filled with ones
        empty_transfer_function_path = os.path.join(transfer_function_dir, "tf_all_ones.h5")

        if not os.path.exists(empty_transfer_function_path):
            raise FileExistsError("Cannot find empty (filled with ones) transfer function file.")
        
        # Define transfer function class instance for full transfer function product
        full_transfer_function = TransferFunction()
        full_transfer_function.read(empty_transfer_function_path)

        if len(self.params.psx_transfer_function_names) == 0:
            print("WARNING: You are running without any transfer function!")

        # If other transfer funtions are to be included multiply these into the full (empty) transfer function
        for filename in self.params.psx_transfer_function_names:
            
            if self.params.transfer_function_dir is None:
                self.params.transfer_function_dir = transfer_function_dir

            path = os.path.join(
                self.params.transfer_function_dir,
                filename
            )

            transfer_function = TransferFunction()
            transfer_function.read(path)

            # Multiplying together transfer functions
            full_transfer_function.transfer_function_2D *=  transfer_function.transfer_function_2D

        # Spline transfer function to make it applicable on non-standard grids
        self.full_transfer_function_interp = interpolate.RectBivariateSpline(
            full_transfer_function.k_bin_centers_perp_2D,
            full_transfer_function.k_bin_centers_par_2D,
            full_transfer_function.transfer_function_2D, 
            s = 0, # No smoothing when splining
            kx = 3, # Use bi-cubic spline in x-direction
            ky = 3, # Use bi-cubic spline in x-direction
            )
        
        if self.params.debug:
            # Unit test to check whether interpolation reproduces input when evaluated at original grid
            approx = self.full_transfer_function_interp(
                full_transfer_function.k_bin_centers_perp_2D,
                full_transfer_function.k_bin_centers_par_2D,
            )
            np.testing.assert_allclose(approx, full_transfer_function.transfer_function_2D)

        # White noise transfer function used to deconvole the effect of the pipeline filteres
        path_wn_transfer_function = os.path.join(
                self.params.transfer_function_dir,
                self.params.psx_white_noise_transfer_function_name
            )
        transfer_function_wn = TransferFunction()
        transfer_function_wn.read(path_wn_transfer_function) 

        self.transfer_function_wn_interp = interpolate.RectBivariateSpline(
            transfer_function_wn.k_bin_centers_perp_2D,
            transfer_function_wn.k_bin_centers_par_2D,
            transfer_function_wn.transfer_function_2D, 
            s = 0, # No smoothing when splining
            kx = 3, # Use bi-cubic spline in x-direction
            ky = 3, # Use bi-cubic spline in x-direction
            )

    def generate_new_monte_carlo_seed(self):
            """Method that generates global Monte Carlo seed from current time.time() used in white noise simulations.
            """
            # If no seed is provided make base seed from current time
            if self.rank == 0:
                t = time.perf_counter()
                # seed = int(t - 1e6 * (t // 1e6))
                seed = int(t)
            else:
                seed = None

            seed = self.comm.bcast(seed, root = 0)
            self.params.psx_white_noise_sim_seed = seed

if __name__ == "__main__":
    
    comap2fpxs = COMAP2FPXS()
    run_wn_sim = comap2fpxs.params.psx_generate_white_noise_sim    
    
    if run_wn_sim:
        comap2fpxs.params.psx_generate_white_noise_sim = False
    
    comap2fpxs.params.psx_white_noise_sim_seed = None
    comap2fpxs.run()

    if run_wn_sim:
        comap2fpxs.params.psx_generate_white_noise_sim = True


        # basepath = "/mn/stornext/d22/cmbco/comap/protodir/power_spectrum/test/average_spectra/co2_python_poly_debug_null_w_bug/"
        # import glob
        # filelist = glob.glob(f"*white_noise_seed*/**/*.h5", root_dir = basepath, recursive = True)
        # seedlist = [int(file.split("seed")[-1].split("/")[0]) for file in filelist]
        seed_list = []
        seed_list_path = os.path.join(comap2fpxs.params.power_spectrum_dir, comap2fpxs.params.psx_seed_list)
        if comap2fpxs.params.psx_use_seed_list:
            seeds_to_run = np.loadtxt(seed_list_path)
        else:
            seeds_to_run = range(comap2fpxs.params.psx_monte_carlo_sim_number)

        global_verbose = comap2fpxs.verbose

        if comap2fpxs.params.psx_use_seed_list:
            print("#" * 80)
            print(f"Running with provided seed list: {seed_list_path}")
            print(f"Seed list contains {len(seeds_to_run)} Monte Carlo simulations")
            print("#" * 80)
        elif global_verbose and comap2fpxs.rank == 0:
            print("#" * 80)
            print(f"Running {comap2fpxs.params.psx_monte_carlo_sim_number} white noise Monte Carlo simulations:")
            print("#" * 80)


        for i, seed in enumerate(seeds_to_run):
            # try:
            # Turning this to None will make new seed from time.time() each iteration
            
            if comap2fpxs.params.psx_use_seed_list:
                comap2fpxs.params.psx_white_noise_sim_seed = seed
            else:
                comap2fpxs.generate_new_monte_carlo_seed()
                
                seed_list.append(comap2fpxs.params.psx_white_noise_sim_seed)

            
            comap2fpxs.verbose = False 

            if global_verbose and comap2fpxs.rank == 0:
                print("-"*40)
                print(f"\033[91m Simulation # {i + 1} / {comap2fpxs.params.psx_monte_carlo_sim_number}: \033[00m \033[93m Seed = {comap2fpxs.params.psx_white_noise_sim_seed} \033[00m")

            t0 = time.perf_counter()
            comap2fpxs.run()
        
            if global_verbose and comap2fpxs.rank == 0:
                print(f"\033[92m Run time: {time.perf_counter() - t0} sec \033[00m")
                print("-"*40)
            # except:
            #     print("SKIP")
            #     continue
        
        comap2fpxs.comm.Barrier()
        if not comap2fpxs.params.psx_use_seed_list and comap2fpxs.rank == 0:
            seed_list = np.array(seed_list)

            if comap2fpxs.verbose:
                print(f"Saving Monte Carlo seed list to: {seed_list_path}")

            np.savetxt(seed_list_path, seed_list.astype(int))
