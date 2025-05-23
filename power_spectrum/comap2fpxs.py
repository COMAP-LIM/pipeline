

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
from matplotlib.gridspec import GridSpec
import matplotlib.patheffects as path_effects

matplotlib.use('Agg')

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time 
import scipy.interpolate as interpolate
import scipy.stats
import scipy.optimize 

from astropy import units as u
import astropy.cosmology

import tqdm
import re

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
                
        self.load_models()
        
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
            if self.params.jk_rnd_split_seed is not None: 
                rnd_string = f"_rnd{self.params.jk_rnd_split_seed}" 
            else:
                rnd_string = ""
            
            field_combinations = [(f"{field_name}_{self.params.map_name}{rnd_string}{self.params.psx_map_name_postfix}.h5", f"{field_name}_{self.params.map_name}{rnd_string}{self.params.psx_map_name_postfix}.h5")
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
            
        
        if self.params.psx_rnd_run:
            all_spectra = np.zeros(
                (       
                len(self.primary_variables),
                2, 
                2,
                len(self.included_feeds), 
                len(self.included_feeds), 
                self.params.psx_number_of_k_bins, 
                self.params.psx_number_of_k_bins,
                )
            )
            all_overlap = np.zeros(
                (   
                len(self.primary_variables),
                2, 
                2,
                len(self.included_feeds), 
                len(self.included_feeds), 
                )

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
                
                current_primary_split = split1[0].split("/")[-1].split("map_")[-1][:4]
                
                if self.params.psx_rnd_run:
                    split_idx = np.where(np.array(self.primary_variables) == current_primary_split)[0]
                
                if self.params.psx_null_diffmap or self.params.psx_rnd_run:
                    cross_var1 = int(split1[0].split("/")[-1][-1])
                    cross_var2 = int(split2[0].split("/")[-1][-1])
                else:
                    cross_var1 = int(split1.split("/")[-1][-1])
                    cross_var2 = int(split2.split("/")[-1][-1])
                    
                if cross_var1 == cross_var2 and feed2 < feed1:
                    # print(split1, split2)
                    # print("Skipping:", cross_var1, cross_var2, feed2, feed1)
                    continue
                
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
                
                try:
                    # Generate cross-spectrum instance from split keys and feeds of current FPXS combo 
                    cross_spectrum = xs_class.CrossSpectrum_nmaps(
                            self.params, 
                            splits, 
                            feed1, 
                            feed2
                        )
                except KeyError:
                    # If some split dataset is not found skip and continue to next
                    print(f"\033[95m WARNING: Split {split1} or {split2} not found in map file\033[00m")
                    continue                
                
                # If map difference null test is computed the output is saved in separate directory
                if self.params.psx_null_diffmap:
                    outdir = os.path.join(outdir, f"null_diffmap/{cross_spectrum.null_variable}")
                
                spectrum_subdir = "spectra_2D"
                if self.params.psx_mode == "saddlebag":
                    spectrum_subdir += "_saddlebag"
                
                # Skip loop iteration if cross-spectrum of current combination already exists
                if os.path.exists(os.path.join(self.params.power_spectrum_dir, spectrum_subdir, outdir, cross_spectrum.outname)):
                    if self.params.psx_rnd_run:
                        inname = os.path.join(self.params.power_spectrum_dir, spectrum_subdir, outdir, cross_spectrum.outname)
                        cross_spectrum.read_and_append_attribute(["xs_2D", "weighted_overlap"], None, inname)                        
                        all_spectra[split_idx, cross_var1, cross_var2, feed1 - 1, feed2 - 1, ...] = cross_spectrum.xs_2D
                        all_overlap[split_idx, cross_var1, cross_var2, feed1 - 1, feed2 - 1] = cross_spectrum.weighted_overlap
                    continue
                else:
                    # Read maps from generated map paths, and provide cosmology to translate from
                    # observational units to cosmological ones.
                    try:
                        cross_spectrum.read_map(
                            mappaths, 
                            self.cosmology, 
                        )
                    except KeyError:                         
                        print(f"\033[95m Map for {feed1}x{feed2} {splits} not found. Moving to next map cross!\033[00m")
                        continue
                    
                    # Compute cross-spectrum for current FPXS combination
                    cross_spectrum.calculate_xs_2d(
                        no_of_k_bins=self.params.psx_number_of_k_bins + 1,
                    )
                    
                    if np.all(cross_spectrum.xs == 0):
                        print(self.rank, "\033[95m WARNING: All XS 2D zero:\033[00m", mapnames, splits, feeds)
                    if np.any(cross_spectrum.xs == 0):
                        print(self.rank, "\033[95m WARNING: Any XS 2D zero:\033[00m", mapnames, splits, feeds)

                    if not self.params.psx_generate_white_noise_sim:
                        # Run noise simulations to generate FPXS errorbar
                        
                        seed = self.params.psx_error_bar_seed
                        if seed is None:
                            t = time.time()
                            seed = int(np.round((t - np.floor(t)) * 1e4))


                        # Computing overlap quantities
                        inv_rms0 = 1.0 / cross_spectrum.maps[0].rms.copy()
                        inv_rms1 = 1.0 / cross_spectrum.maps[1].rms.copy()
                        
                        IoU = np.sum(np.logical_and(np.isfinite(inv_rms0), np.isfinite(inv_rms1)), axis = (0, 1)).astype(float)
                        IoU /= np.sum(np.logical_or(np.isfinite(inv_rms0), np.isfinite(inv_rms1)), axis = (0, 1)).astype(float)
                        cross_spectrum.IoU = np.nanmean(IoU)
                        
                        inv_rms0[~np.isfinite(inv_rms0)] = np.nan
                        inv_rms1[~np.isfinite(inv_rms1)] = np.nan
                        
                        cross_spectrum.weighted_overlap = (
                            (inv_rms0 * inv_rms1) 
                            / np.sqrt(np.nansum(inv_rms0 ** 2, axis = (0, 1)) * np.nansum(inv_rms1 ** 2, axis = (0, 1)))[None, None, :]
                        )
                        
                        cross_spectrum.weighted_overlap = np.nansum(cross_spectrum.weighted_overlap, axis = (0, 1))
                        
                        cross_spectrum.weighted_overlap = np.nanmean(cross_spectrum.weighted_overlap[cross_spectrum.weighted_overlap != 0])
                        
                        if self.params.psx_rnd_run:
                            all_spectra[split_idx, cross_var1, cross_var2, feed1 - 1, feed2 - 1, ...] = cross_spectrum.xs
                            all_overlap[split_idx, cross_var1, cross_var2, feed1 - 1, feed2 - 1] = cross_spectrum.weighted_overlap 
                        
                        elif not self.params.psx_rnd_run and self.params.psx_noise_sim_number > 0:
                            cross_spectrum.run_noise_sims_2d(
                                self.params.psx_noise_sim_number,
                                no_of_k_bins=self.params.psx_number_of_k_bins + 1,
                                seed = seed + (i + 1),
                            )

                    else:
                        cross_spectrum.xs *= transfer_function_wn
                        cross_spectrum.read_and_append_attribute(["rms_xs_mean_2D", "rms_xs_std_2D", "white_noise_covariance", "white_noise_simulation"], outdir_data)
                                        
                    # Save resulting FPXS from current combination to file
                    cross_spectrum.make_h5_2d(outdir)
        
        # MPI barrier to prevent thread 0 from computing average FPXS before all individual combinations are finished.
        self.comm.Barrier()
        if self.rank == 0:
            pbar.close()
            
        if self.rank == 0 and not self.params.psx_rnd_run:
            # Compute average FPXS and finished data product plots
            print("\nComputing averages:")
            self.compute_averages()
            self.grid_combo_plot()
        
        elif self.params.psx_rnd_run:
            all_spectra_buffer = np.zeros_like(all_spectra)
            all_overlap_buffer = np.zeros_like(all_overlap)
            self.comm.Reduce(
                [all_spectra, MPI.DOUBLE],
                [all_spectra_buffer, MPI.DOUBLE],
                op=MPI.SUM,
                root=0,
            )
            
            self.comm.Reduce(
                [all_overlap, MPI.DOUBLE],
                [all_overlap_buffer, MPI.DOUBLE],
                op=MPI.SUM,
                root=0,
            )
            
            if self.rank == 0:
                all_spectra = all_spectra_buffer
                all_overlap = all_overlap_buffer 
                rndsubdir = "rnd_split_files"
                if self.params.psx_mode == "saddlebag":
                    rndsubdir += "_saddlebag"
                rndpath = os.path.join(self.params.power_spectrum_dir, rndsubdir)
                rndpath = os.path.join(rndpath, f"{self.params.fields[0]}_{self.params.map_name}_rnd{self.params.jk_rnd_split_seed}{self.params.psx_map_name_postfix}")
                if not os.path.exists(rndpath):
                    os.makedirs(rndpath, exist_ok = True)
                
                rndfile = os.path.join(rndpath, f"rndfile_seed{self.params.jk_rnd_split_seed}.h5")
                if not os.path.exists(rndfile):
                    with h5py.File(rndfile, "w") as outfile:
                        outfile["all_spectra"] = all_spectra
                        outfile["all_overlap"] = all_overlap
        
    
    def compute_averages(self):
        
        beam_tf = self.beam_transfer_function() 
        
        if self.params.psx_mode == "saddlebag":
            average_spectrum_dir = os.path.join(self.power_spectrum_dir, "average_spectra_saddlebag")
        else:
            average_spectrum_dir = os.path.join(self.power_spectrum_dir, "average_spectra")

        fig_dir = os.path.join(self.power_spectrum_dir, "figs")

        if not os.path.exists(average_spectrum_dir):
            os.mkdir(average_spectrum_dir)
    
        # fig_dir = os.path.join(fig_dir, self.params.psx_subdir)
        if not os.path.exists(fig_dir):
            os.mkdir(fig_dir)

        if self.params.psx_mode == "saddlebag":
            if not os.path.exists(os.path.join(fig_dir, "chi2_grid_saddlebag")):
                os.mkdir(os.path.join(fig_dir, "chi2_grid_saddlebag"))

            if not os.path.exists(os.path.join(fig_dir, "average_spectra_saddlebag")):
                os.mkdir(os.path.join(fig_dir, "average_spectra_saddlebag"))
            if self.params.psx_null_diffmap:
                if not os.path.exists(os.path.join(fig_dir, "null_pte_saddlebag")):
                    os.mkdir(os.path.join(fig_dir, "null_pte_saddlebag"))
        else:
            if not os.path.exists(os.path.join(fig_dir, "chi2_grid")):
                os.mkdir(os.path.join(fig_dir, "chi2_grid"))

            if not os.path.exists(os.path.join(fig_dir, "average_spectra")):
                os.mkdir(os.path.join(fig_dir, "average_spectra"))
                
            if self.params.psx_null_diffmap:
                if not os.path.exists(os.path.join(fig_dir, "null_pte")):
                    os.mkdir(os.path.join(fig_dir, "null_pte"))
        
        N_feed = len(self.included_feeds)
        N_splits = len(self.split_map_combinations) // 3
        N_k = self.params.psx_number_of_k_bins

        if self.params.psx_noise_sim_number <= 0 and len(self.params.psx_rnd_file_list) == 0:
            raise ValueError("Cannot compute average power spectrum without noise simulations or an RND ensemble.")
        elif len(self.params.psx_rnd_file_list) > 0:
            import glob
            rndsubdir = "rnd_split_files"
            if self.params.psx_mode == "saddlebag":
                rndsubdir += "_saddlebag"
                
            if len(self.params.psx_rnd_path) == 0 or not os.path.exists(self.params.psx_rnd_path):            
                rndpath = os.path.join(self.params.power_spectrum_dir, rndsubdir)
            else:
                rndpath = os.path.join(self.params.psx_rnd_path, rndsubdir)
                
            for num_rnd, rndfile_name in enumerate(self.params.psx_rnd_file_list):
                rndfile = os.path.join(rndpath, f"{self.params.fields[0]}_{rndfile_name}")
                print("Loading RND-file from: ", rndfile)
                rndfile = glob.glob(os.path.join(rndfile, f"*.h5"))[0]
                
                with h5py.File(rndfile, "r") as infile:
                    rnd_spectra = infile["all_spectra"][()] 
                    rnd_overlap = infile["all_overlap"][()] 
                    print(np.any(rnd_spectra == 0), *np.where(rnd_spectra == 0), np.unique(np.where(rnd_spectra == 0)[1]), (np.sum(rnd_spectra == 0)) / rnd_spectra.size * 100, rnd_spectra.shape)
                if num_rnd == 0:
                    all_rnd_spectra = rnd_spectra
                    all_rnd_overlap = rnd_overlap
                else:
                    all_rnd_spectra = np.concatenate((all_rnd_spectra, rnd_spectra), axis = 0)
                    all_rnd_overlap = np.concatenate((all_rnd_overlap, rnd_overlap), axis = 0)
            
                
            all_rnd_overlap = np.nanmedian(all_rnd_overlap, axis = 0)
            
            if np.all(all_rnd_spectra == 0) or np.any(~np.isfinite(all_rnd_spectra)):
                raise ValueError("All loaded RND spectra are either zero or infinite/NaN!")
            # if np.any(all_rnd_spectra == 0) or np.any(~np.isfinite(all_rnd_spectra)):
            #     print(np.sum(all_rnd_spectra == 0), all_rnd_spectra.size, np.any(all_rnd_spectra == 0), np.any(~np.isfinite(all_rnd_spectra)))
            #     # for s in range(len(all_rnd_spectra.shape)):
                #     print(np.sum(np.all(all_rnd_spectra == 0, axis = s)), all_rnd_spectra.shape, all_rnd_spectra.shape[s])
                # raise ValueError("Some loaded RND spectra are either zero or infinite/NaN!")
            
                        
            ### all in one ###
            # all_rnd_std = np.nanstd(all_rnd_spectra, axis = 0, ddof = 1)
            
            all_rnd_spectra[all_rnd_spectra == 0] = np.nan
            
            ### 60-120 ###
            all_rnd_std = np.nanstd(all_rnd_spectra[:all_rnd_spectra.shape[0] // 4], axis = 0, ddof = 1)
            all_rnd_spectra = all_rnd_spectra[all_rnd_spectra.shape[0] // 4:]

            ### 90-90 ###
            # all_rnd_std = np.nanstd(all_rnd_spectra[:all_rnd_spectra.shape[0] // 2], axis = 0, ddof = 1)
            # all_rnd_spectra = all_rnd_spectra[all_rnd_spectra.shape[0] // 2:]
            
            ### 120-60 ###
            # all_rnd_std = np.nanstd(all_rnd_spectra[all_rnd_spectra.shape[0] // 3:], axis = 0, ddof = 1)
            # all_rnd_spectra = all_rnd_spectra[:all_rnd_spectra.shape[0] // 2]
            
            self.params.psx_noise_sim_number = all_rnd_spectra.shape[0]
            
        
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

            xs_all = np.zeros((N_splits, N_feed, N_feed, N_k, N_k))
            xs_error_all = np.zeros((N_splits, N_feed, N_feed, N_k, N_k))

            xs_mean = np.zeros((N_splits, N_k, N_k))
            xs_error = np.zeros((N_splits, N_k, N_k))
            xs_covariance = np.zeros((N_splits, N_k ** 2, N_k ** 2))
            xs_full_operator = np.zeros((N_splits, N_k ** 2, N_k ** 2))
            
            xs_wn_covariance = np.zeros((N_splits, N_k ** 2, N_k ** 2))
            xs_wn_mean = np.zeros((N_splits, self.params.psx_noise_sim_number, N_k, N_k))
            chi2_wn = np.zeros((N_splits, self.params.psx_noise_sim_number))
            chi2_wn_cov = np.zeros((N_splits, self.params.psx_noise_sim_number))
            chi2_data = np.zeros(N_splits)
            chi2_data_cov = np.zeros(N_splits)
            chi2_data_coadd = np.zeros(N_splits)
            PTE_data_cov = np.zeros(N_splits)
            PTE_data = np.zeros(N_splits)
            PTE_data_coadd = np.zeros(N_splits)
            PTE_data_coadd_numeric = np.zeros(N_splits)
            
            xs_mean_1d = np.zeros((N_splits, N_k))
            xs_error_1d = np.zeros((N_splits, N_k))
            xs_wn_mean_1d = np.zeros((N_splits, self.params.psx_noise_sim_number, N_k))
            xs_wn_covariance_1d = np.zeros((N_splits, N_k, N_k))
            chi2_wn_1d = np.zeros((N_splits, self.params.psx_noise_sim_number))
            chi2_wn_1d_models = np.zeros((N_splits, len(self.models), self.params.psx_noise_sim_number))
            chi2_wn_cov_1d = np.zeros((N_splits, self.params.psx_noise_sim_number))
            chi2_data_1d = np.zeros(N_splits)
            chi2_data_cov_1d = np.zeros(N_splits)
            chi2_data_coadd_1d = np.zeros(N_splits)
            chi2_data_coadd_1d_models = np.zeros((N_splits, len(self.models)))
            PTE_data_1d = np.zeros(N_splits)
            PTE_data_cov_1d = np.zeros(N_splits)
            PTE_data_coadd_1d = np.zeros(N_splits)
            PTE_data_coadd_numeric_1d = np.zeros(N_splits)
            
            chi2_grids = np.zeros((N_splits, N_feed, N_feed))
            
            if self.params.psx_null_diffmap:
                loaded_chi2_grids = np.zeros((N_splits, N_feed, N_feed))
            
            cross_variable_names = [] 

            
            if self.params.psx_null_diffmap:
                if self.params.psx_mode == "saddlebag":
                    self.params.psx_chi2_import_path = re.sub(r"average_spectra", "average_spectra_saddlebag", self.params.psx_chi2_import_path)

                self.params.psx_chi2_import_path = re.sub(r"co\d_", f"{self.params.fields[0]}_", self.params.psx_chi2_import_path)
                print(self.params.psx_chi2_import_path)
                if len(self.params.psx_chi2_import_path) <= 0 or not os.path.exists(self.params.psx_chi2_import_path):
                    raise ValueError("No chi2 import file provided to perform null test chi2 cuts!")

                with h5py.File(self.params.psx_chi2_import_path, "r") as infile:
                    loaded_names = infile["cross_variable_names"][()].astype(str)
                    loaded_chi2_grid = infile["chi2_grid"][()]

            split_counter = 0
            # for i, splits in enumerate(self.split_map_combinations):
            for i in range(N_splits):
                splits = self.split_map_combinations[split_counter]
                split_counter += 3

                if not self.params.psx_null_diffmap:
                    cross_variable1 = int(splits[0].split("/")[-1][-1])
                    cross_variable2 = int(splits[1].split("/")[-1][-1])
                else:
                    cross_variable1 = int(splits[0][0].split("/")[-1][-1])
                    cross_variable2 = int(splits[1][0].split("/")[-1][-1])
                    
                if cross_variable1 == cross_variable2:
                    # Exclude auto-cross variable combos
                    continue
                
                
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

                mean_IoUs = np.zeros((N_feed, N_feed)) * np.nan
                mean_weighted_overlaps = np.zeros((N_feed, N_feed)) * np.nan
                
                
                for feed1 in range(N_feed):
                    for feed2 in range(N_feed):
                        cross_spectrum = xs_class.CrossSpectrum_nmaps(
                            self.params, 
                            splits, 
                            self.included_feeds[feed1], 
                            self.included_feeds[feed2],
                        )   
                        
                        try:
                            cross_spectrum.read_spectrum(indir)
                        except (FileNotFoundError, KeyError):
                            print(f"\033[95m WARNING: Split {splits[0]} or {splits[1]} not found in map file. Skipping split in averaging.\033[00m")
                            continue            
                        
                        
                        cross_spectrum.read_and_append_attribute(["white_noise_simulation", "IoU", "weighted_overlap", "dx", "dy", "dz"], indir)
                        
                        if len(self.params.psx_rnd_file_list) > 0:
                            xs_wn = all_rnd_spectra[:, cross_variable1, cross_variable2, feed1, feed2, ...]
                            xs_sigma = all_rnd_std[cross_variable1, cross_variable2, feed1, feed2, ...]
                        else:
                            xs_wn = cross_spectrum.white_noise_simulation
                            xs_sigma = cross_spectrum.rms_xs_std_2D
                            # Applying white noise transfer function
                            transfer_function_wn = self.transfer_function_wn_interp(k_bin_centers_perp, k_bin_centers_par)
                            
                            xs_sigma *= transfer_function_wn 
                            xs_wn *= transfer_function_wn[None, ...]
                            
                        xs = cross_spectrum.xs_2D

                        xs_all[i, feed1, feed2, ...] = xs
                        xs_error_all[i, feed1, feed2, ...] = xs_sigma
                        
                        k_bin_centers_perp, k_bin_centers_par  = cross_spectrum.k
                        
                        k_bin_edges_par = cross_spectrum.k_bin_edges_par
                        
                        k_bin_edges_perp = cross_spectrum.k_bin_edges_perp

                        kgrid = np.sqrt(sum(ki**2 for ki in np.meshgrid(k_bin_centers_perp, k_bin_centers_par, indexing="ij")))

                        # Filter transfer function
                        transfer_function = np.abs(self.full_transfer_function_interp(k_bin_centers_perp, k_bin_centers_par))
                        
                        # Beam and voxel window transfer functions
                        px_window = self.pix_window(k_bin_centers_perp, cross_spectrum.dx)
                        freq_window = self.pix_window(k_bin_centers_par, cross_spectrum.dz)
                        transfer_function *= beam_tf(k_bin_centers_perp)[:, None] * px_window[:, None] 
                        transfer_function *= freq_window[None, :] 
                        
                        
                        # Transfer function mask
                        tf_cutoff = self.params.psx_tf_cutoff * np.nanmax(transfer_function[1:-1, 1:-1])
                        transfer_function_mask = transfer_function > tf_cutoff 
                        
                        k_PAR, k_PERP = np.meshgrid(k_bin_centers_par, k_bin_centers_perp)
                        
                        
                        
                        # Only edges of k-space to be masked
                        if self.params.psx_mask_k_perp_max < self.params.psx_mask_k_perp_min:
                            raise ValueError("Value self.params.psx_mask_k_perp_max smaller than self.params.psx_mask_k_perp_min")
                        if self.params.psx_mask_k_par_max < self.params.psx_mask_k_par_min:
                            raise ValueError("Value self.params.psx_mask_k_par_max smaller than self.params.psx_mask_k_par_min")
                        if self.params.psx_mask_k_max < self.params.psx_mask_k_min:
                            raise ValueError("Value self.params.psx_mask_k_max smaller than self.params.psx_mask_k_min")
                        
                        
                        # Mask bins at edges of k-space
                        k_edge_mask = np.ones_like(transfer_function_mask, dtype = bool)
                        k_edge_mask[k_PERP > self.params.psx_mask_k_perp_max] = False
                        k_edge_mask[k_PERP < self.params.psx_mask_k_perp_min] = False
                        k_edge_mask[k_PAR > self.params.psx_mask_k_par_max] = False
                        k_edge_mask[k_PAR < self.params.psx_mask_k_par_min] = False
                        
                        # Mask k-rings in k-space
                        k_ring_mask = np.ones_like(k_edge_mask, dtype = bool)
                        k_ring_mask[kgrid < self.params.psx_mask_k_min] = False
                        k_ring_mask[kgrid > self.params.psx_mask_k_max] = False
                        
                        # Merge edge and ring masks
                        k_space_mask = np.logical_and(k_edge_mask, k_ring_mask)
                        
                        transfer_function_mask = np.logical_and(transfer_function_mask, k_space_mask)
                        
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
                            # print("hallo", np.where(loaded_names == cross_variable,)[0], loaded_names, cross_variable)
                            current_cross_variable = np.where(loaded_names == cross_variable)[0][0]
                            accept_chi2 = np.abs(loaded_chi2_grid[current_cross_variable, feed1, feed2]) < self.params.psx_chi2_cut_limit
                            # accept_chi2 = np.abs(chi2[feed1, feed2]) < self.params.psx_chi2_cut_limit
                        else:
                            accept_chi2 = np.abs(chi2[feed1, feed2]) < self.params.psx_chi2_cut_limit
                        

                        accepted_chi2[feed1, feed2] = accept_chi2
                        mean_weighted_overlaps[feed1, feed2] = cross_spectrum.weighted_overlap
                        mean_IoUs[feed1, feed2] = cross_spectrum.IoU
                
                        if (np.isfinite(chi2[feed1, feed2]) and chi2[feed1, feed2] != 0) and feed1 != feed2:
                            if cross_spectrum.weighted_overlap > self.params.psx_overlap_limit:
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
                    _chi2 = loaded_chi2_grid[current_cross_variable, ...].copy()
                    for ii in range(_chi2.shape[0]):
                        _chi2[ii, ii] = np.inf
                    print(f"{indir} {splits} \n# |chi^2| < {self.params.psx_chi2_cut_limit}:", np.sum(np.logical_and(np.abs(_chi2) < self.params.psx_chi2_cut_limit, mean_weighted_overlaps > self.params.psx_overlap_limit)))
                    # print(f"{indir} {splits} \n# |chi^2| < {self.params.psx_chi2_cut_limit}:", np.sum(np.abs(chi2) < self.params.psx_chi2_cut_limit))
                else:
                    _chi2 = chi2.copy()
                    for ii in range(_chi2.shape[0]):
                        _chi2[ii, ii] = np.inf
                    print(f"{indir} {splits} \n# |chi^2| < {self.params.psx_chi2_cut_limit}:", np.sum(np.logical_and(np.abs(_chi2) < self.params.psx_chi2_cut_limit, mean_weighted_overlaps > self.params.psx_overlap_limit)))

                
                xs_mean[i, ...] = xs_sum / xs_inv_var
                xs_error[i, ...] = 1.0 / np.sqrt(xs_inv_var)

                xs_wn_mean[i, ...] = wn_xs_sum / xs_inv_var[None, ...]
                
                chi2_wn_data = xs_wn_mean[i, :, transfer_function_mask].T
                chi2_wn[i, :] = np.nansum((chi2_wn_data  / xs_error[i, None, transfer_function_mask].T) ** 2, axis = 1)
                # chi2_wn[i, :] = np.nansum((chi2_wn_data  / _error) ** 2, axis = 1)
                
                
                flattened_mask = transfer_function_mask.flatten()
                flattened_chi2_wn_data = xs_wn_mean[i, :, ...].reshape(self.params.psx_noise_sim_number, N_k * N_k)                    
                flattened_chi2_wn_data = flattened_chi2_wn_data[:, flattened_mask]
                
                flattened_data = xs_mean[i, ...].reshape(N_k * N_k)
                flattened_error = xs_error[i, ...].reshape(N_k * N_k)
                flattened_data = flattened_data[flattened_mask]
                flattened_error = flattened_error[flattened_mask]
                
                
                # cov_2d = xs_wn_covariance[i, :, :].copy()
                
                
                # new_cov_2d = np.zeros_like(xs_wn_covariance[i, ...])
                # for ii, d in enumerate(range(13, 14 * 14, 14)):
                #     if d > cov_2d.diagonal().size:
                #         break
                #     new_cov_2d += np.diag(cov_2d.diagonal(d), d)
                #     new_cov_2d += np.diag(cov_2d.diagonal(-(d)), -(d))
                # new_cov_2d += np.diag(cov_2d.diagonal(0), 0)
                
                # cov_2d = new_cov_2d

                
                # cov_2d = cov_2d[flattened_mask, :]
                # cov_2d = cov_2d[:, flattened_mask]
                
                # inv_xs_wn_covariance = np.linalg.inv(cov_2d)

                # for k in range(self.params.psx_noise_sim_number):
                #     chi2_wn_cov[i, k] = flattened_chi2_wn_data[k, :].T @ inv_xs_wn_covariance @ flattened_chi2_wn_data[k, :]

                # chi2_data_cov[i] = flattened_data.T @ inv_xs_wn_covariance @ flattened_data
                
                # chi2_data[i] = np.nansum((masked_data  / xs_error[i, flattened_mask]) ** 2)
                # chi2_data[i] = np.nansum((flattened_data  / np.sqrt(cov_2d.diagonal())) ** 2)
                chi2_data_coadd[i] = np.nansum((flattened_data  / flattened_error) ** 2)
                
                # PTE_data_cov[i] = scipy.stats.chi2.sf(chi2_data_cov[i], df = np.sum(transfer_function_mask))
                # PTE_data[i] = scipy.stats.chi2.sf(chi2_data[i], df = np.sum(transfer_function_mask))
                # PTE_data_coadd[i] = scipy.stats.chi2.sf(chi2_data_coadd[i], df = np.sum(transfer_function_mask))
                # PTE_data_coadd_numeric[i] = 1 - np.sum(chi2_wn[i, :] <= chi2_data_coadd[i]) / self.params.psx_noise_sim_number
                
                
                ecdf = scipy.stats.ecdf(chi2_wn[i, :])
                PTE_data_coadd_numeric[i] = 1 - ecdf.cdf.evaluate(chi2_data_coadd[i])
                
                chi2_cdf_x = np.linspace(0, 1e3, int(1e4))
                chi2_cdf_y = ecdf.cdf.evaluate(chi2_cdf_x)
                
                df_opt, _ = scipy.optimize.curve_fit(lambda x, df: scipy.stats.chi2.cdf(x, df), chi2_cdf_x, chi2_cdf_y, p0 = np.sum(transfer_function_mask)) 
                # df_opt, _ = scipy.optimize.curve_fit(lambda x, df: scipy.stats.chi2.cdf(x + df, np.sum(transfer_function_mask)), chi2_cdf_x, chi2_cdf_y, p0 = np.sum(transfer_function_mask)) 
                
                self.df_2d_best_fit = df_opt[0]
                print("2D chi2-dist. best-fit shift:", *df_opt)
                # PTE_data_coadd[i] = scipy.stats.chi2.sf(chi2_data_coadd[i], df = np.sum(transfer_function_mask))
                PTE_data_coadd[i] = scipy.stats.chi2.sf(chi2_data_coadd[i], df = df_opt[0])
                # PTE_data_coadd[i] = scipy.stats.chi2.sf(chi2_data_coadd[i] + df_opt[0], df = np.sum(transfer_function_mask))

                print("2D PTE:", PTE_data_coadd[i], "2D PTE numeric:", PTE_data_coadd_numeric[i], "2D chi2: ", chi2_data_coadd[i])
                                    
                if np.all(~np.isfinite(chi2)) or np.all(chi2 == 0):
                    print("All chi2 == NaN or chi2 == 0. Continuing loop!")
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

                print(Ck_wn_1d.shape)
                rms_1d[k_1d < self.params.psx_mask_k_1d_min] = np.nan
                Ck_1d[k_1d < self.params.psx_mask_k_1d_min] = np.nan
                Ck_wn_1d[:, k_1d < self.params.psx_mask_k_1d_min] = np.nan
                
                rms_1d[k_1d > self.params.psx_mask_k_1d_max] = np.nan
                Ck_1d[k_1d > self.params.psx_mask_k_1d_max] = np.nan
                Ck_wn_1d[:, k_1d > self.params.psx_mask_k_1d_max] = np.nan
                
                xs_error_1d[i, ...] = rms_1d

                xs_mean_1d[i, ...] = Ck_1d
                xs_wn_mean_1d[i, ...] = Ck_wn_1d
                # xs_wn_covariance_1d[i, ...] = np.cov(Ck_wn_1d.T)

                mask = np.where(np.isfinite(Ck_1d / rms_1d))[0]
                # cov_1d = xs_wn_covariance_1d[i, mask, :]
                # cov_1d = cov_1d[:, mask]
                
                chi2_wn_data_1d = xs_wn_mean_1d[i, :, mask].T
                data_1d = xs_mean_1d[i, mask]
                
                # _error_1d = np.sqrt(cov_1d.diagonal())
                # self._error_1d = _error_1d.copy()

                df_1d = np.isfinite(Ck_1d / rms_1d).sum()
                # chi2_wn_1d[i, :] = np.nansum((chi2_wn_data_1d  / rms_1d[mask]) ** 2, axis = 1)
                # chi2_wn_1d[i, :] = np.nansum((chi2_wn_data_1d  / _error_1d) ** 2, axis = 1)
                chi2_wn_1d[i, :] = np.nansum((chi2_wn_data_1d  / rms_1d[None, mask]) ** 2, axis = 1)

                
                # print(np.any(chi2_wn_1d[i, :] == 0), chi2_wn_1d[i, :], chi2_wn_1d[i, :].shape, chi2_wn_data_1d.shape, np.where(chi2_wn_data_1d == 0))
                # print(rms_1d[None, mask].shape, chi2_wn_data_1d.shape, chi2_wn_data_1d)

                # inv_xs_wn_covariance_1d = np.linalg.inv(cov_1d)

                # for k in range(self.params.psx_noise_sim_number):
                #     chi2_wn_cov_1d[i, k] = chi2_wn_data_1d[k, :].T @ inv_xs_wn_covariance_1d @ chi2_wn_data_1d[k, :]

                # chi2_data_cov_1d[i] = data_1d.T @ inv_xs_wn_covariance_1d @ data_1d
                # chi2_data_1d[i] = np.nansum((data_1d  / _error_1d) ** 2)
                chi2_data_coadd_1d[i] = np.nansum((data_1d  / xs_error_1d[i, mask]) ** 2)

                modelnames = []
                for m, (name, model) in enumerate(self.models.items()):
                    chi2_wn_1d_models[i, m, :] = np.nansum(((chi2_wn_data_1d  - model.interpolation(k_1d[mask])) / rms_1d[None, mask]) ** 2, axis = 1)
                    chi2_data_coadd_1d_models[i, m] = np.nansum(((data_1d - model.interpolation(k_1d[mask]))  / xs_error_1d[i, mask]) ** 2)
                    modelnames.append(name)

                # PTE_data_cov_1d[i] = scipy.stats.chi2.sf(chi2_data_cov_1d[i], df = df_1d)
                # PTE_data_1d[i] = scipy.stats.chi2.sf(chi2_data_1d[i], df = df_1d)
                # PTE_data_coadd_1d[i] = scipy.stats.chi2.sf(chi2_data_coadd_1d[i], df = df_1d)
                #PTE_data_coadd_numeric_1d[i] = 1 - np.sum(chi2_wn_1d[i, :] <= chi2_data_coadd_1d[i]) / self.params.psx_noise_sim_number
                ecdf = scipy.stats.ecdf(chi2_wn_1d[i, :])
                PTE_data_coadd_numeric_1d[i] = 1 - ecdf.cdf.evaluate(chi2_data_coadd_1d[i])

                
                chi2_cdf_x = np.linspace(0, 1e2, int(1e4))
                chi2_cdf_y = ecdf.cdf.evaluate(chi2_cdf_x)
                
                df_opt, _ = scipy.optimize.curve_fit(lambda x, df: scipy.stats.chi2.cdf(x, df), chi2_cdf_x, chi2_cdf_y, p0 = df_1d) 
                # df_opt, _ = scipy.optimize.curve_fit(lambda x, df: scipy.stats.chi2.cdf(x + df, df_1d), chi2_cdf_x, chi2_cdf_y, p0 = df_1d) 
                
                self.df_1d_best_fit = df_opt[0]
                print("1D chi2-dist. best-fit shift:", *df_opt)
                
                # PTE_data_coadd_1d[i] = scipy.stats.chi2.sf(chi2_data_coadd_1d[i], df = df_1d)
                PTE_data_coadd_1d[i] = scipy.stats.chi2.sf(chi2_data_coadd_1d[i], df = df_opt[0])
                # PTE_data_coadd_1d[i] = scipy.stats.chi2.sf(chi2_data_coadd_1d[i] + df_opt[0], df = df_1d)
                print("1D PTE:", PTE_data_coadd_1d[i], "1D PTE numeric:", PTE_data_coadd_numeric_1d[i], "1D chi2: ", chi2_data_coadd_1d[i])
                        
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

                    self.plot_chi2_grid(chi2, np.logical_and(accepted_chi2, mean_weighted_overlaps > self.params.psx_overlap_limit), splits, chi2_name)
                    self.plot_overlap_stats(mean_weighted_overlaps / all_rnd_overlap[cross_variable1, cross_variable2], mean_weighted_overlaps, np.logical_and(accepted_chi2, mean_weighted_overlaps > self.params.psx_overlap_limit), splits, chi2_name)
                    
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
                        average_name = os.path.join(fig_dir, "average_spectra_saddlebag")
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

                    
                    
                self.angle2Mpc = cross_spectrum.angle2Mpc * u.Mpc / u.arcmin

                self.map_dx = cross_spectrum.dx
                self.map_dy = cross_spectrum.dy
                self.map_dz = cross_spectrum.dz


                #if not self.params.psx_generate_white_noise_sim:
                self.plot_2D_mean(
                    k_bin_edges_par,
                    k_bin_edges_perp,
                    k_bin_centers_par,
                    k_bin_centers_perp,
                    xs_mean[i, ...] / transfer_function,
                    xs_error[i, ...] / transfer_function,
                    xs_wn_covariance[i, ...],
                    transfer_function_mask,
                    (chi2_wn[i, ...], chi2_wn_cov[i, ...]),
                    (chi2_data_coadd[i], PTE_data_coadd[i], PTE_data_coadd_numeric[i]),
                    splits,
                    (mapname1, mapname2),
                    average_name,
                )

                # self.plot_feed_grid(
                #     k_bin_edges_par,
                #     k_bin_edges_perp,
                #     k_bin_centers_par,
                #     k_bin_centers_perp,
                #     xs_all[i, ...] / transfer_function,
                #     xs_error_all[i, ...] / transfer_function,
                #     transfer_function_mask,
                #     splits,
                #     average_name,
                # )
                
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
                    mean_weighted_overlaps,
                    (chi2_wn_1d[i, ...], chi2_wn_cov_1d[i, ...]),
                    (chi2_data_coadd_1d[i], PTE_data_coadd_1d[i], PTE_data_coadd_numeric_1d[i]),
                    splits,
                    (mapname1, mapname2),
                    average_name,
                )
            
            if not self.params.psx_null_diffmap:
                cross_variable_names = np.array(cross_variable_names, dtype = "S")
            else:
                cross_variable_names = np.array([*self.cross_variables], dtype = "S")

            if not os.path.exists(os.path.join(outdir, self.params.psx_subdir)):
                os.mkdir(os.path.join(outdir, self.params.psx_subdir))
            
            with h5py.File(os.path.join(os.path.join(outdir, self.params.psx_subdir), mapname + "_average_fpxs.h5"), "w") as outfile:
                outfile.create_dataset("k_1d", data = k_1d)      
                outfile.create_dataset("k_edges_1d", data = k_bin_edges)             
                outfile.create_dataset("k_centers_par", data = k_bin_centers_par)
                outfile.create_dataset("k_centers_perp", data = k_bin_centers_perp)
                outfile.create_dataset("k_edges_par", data = k_bin_edges_par)      
                outfile.create_dataset("k_edges_perp", data = k_bin_edges_perp)     
                outfile.create_dataset("xs_mean_1d", data = xs_mean_1d)       
                outfile.create_dataset("xs_mean_2d", data = xs_mean / transfer_function[None, ...])
                outfile.create_dataset("xs_sigma_1d", data = xs_error_1d)      
                outfile.create_dataset("xs_sigma_2d", data = xs_error / transfer_function[None, ...])
                outfile.create_dataset("cross_variable_names", data = cross_variable_names)                
                outfile.create_dataset("white_noise_covariance", data = xs_covariance)
                outfile.create_dataset("transfer_function_mask", data = transfer_function_mask)
                outfile.create_dataset("transfer_function", data = transfer_function)
                outfile.create_dataset("xs_covariance_2d", data = xs_wn_covariance)
                outfile.create_dataset("xs_covariance_1d", data = xs_wn_covariance_1d)
                
                # Normalized chi2 value for feed-feed split cross-spectra
                outfile.create_dataset("chi2_grid", data = chi2_grids)
                
                # White noise simulation chi2 values
                outfile.create_dataset("chi2_white_noise_sim_2d", data = chi2_wn)
                outfile.create_dataset("chi2_white_noise_sim_cov_2d", data = chi2_wn_cov)
                outfile.create_dataset("chi2_white_noise_sim_1d", data = chi2_wn_1d)
                outfile.create_dataset("chi2_white_noise_sim_cov_1d", data = chi2_wn_cov_1d)
                
                # Data chi2 values
                outfile.create_dataset("chi2_data_cov_2d", data = chi2_data_cov)
                outfile.create_dataset("chi2_data_coadd_2d", data = chi2_data_coadd)
                outfile.create_dataset("chi2_data_2d", data = chi2_data)
                outfile.create_dataset("chi2_data_cov_1d", data = chi2_data_cov_1d)
                outfile.create_dataset("chi2_data_coadd_1d", data = chi2_data_coadd_1d)
                outfile.create_dataset("chi2_data_1d", data = chi2_data_1d)
                
                outfile.create_dataset("modelnames", data = np.array(modelnames, dtype = "S"))
                outfile.create_dataset("chi2_data_coadd_1d_models", data = chi2_data_coadd_1d_models)
                outfile.create_dataset("chi2_white_noise_sim_1d_models", data = chi2_wn_1d_models)
                
                
                # Data PTE values
                outfile.create_dataset("PTE_data_cov_2d", data = PTE_data_cov)
                outfile.create_dataset("PTE_data_coadd_2d", data = PTE_data_coadd)
                outfile.create_dataset("PTE_data_coadd_numeric_2d", data = PTE_data_coadd_numeric)
                outfile.create_dataset("PTE_data_2d", data = PTE_data)
                outfile.create_dataset("PTE_data_cov_1d", data = PTE_data_cov_1d)
                outfile.create_dataset("PTE_data_coadd_1d", data = PTE_data_coadd_1d)
                outfile.create_dataset("PTE_data_coadd_numeric_1d", data = PTE_data_coadd_numeric_1d)
                outfile.create_dataset("PTE_data_1d", data = PTE_data_1d)
                
                
                if not self.params.psx_null_diffmap:                
                    outfile.create_dataset("num_chi2_below_cutoff", data = np.sum(np.abs(chi2_grids) < self.params.psx_chi2_cut_limit, axis = (1, 2)))
                else:
                    outfile.create_dataset("null_variable_names", data = self.primary_variables)
                    outfile.create_dataset("num_chi2_below_cutoff", data = np.sum(np.abs(loaded_chi2_grids) < self.params.psx_chi2_cut_limit, axis = (1, 2)))
                    outfile.create_dataset("loaded_chi2_grid", data = loaded_chi2_grids)

                    if self.params.psx_mode == "saddlebag":
                        null_pte_name = os.path.join(fig_dir, "null_pte_saddlebag")
                    else:
                        null_pte_name = os.path.join(fig_dir, "null_pte")

                    null_pte_name = os.path.join(null_pte_name, indir)
                    null_pte_name = null_pte_name + f"_{self.params.psx_plot_name_suffix}"
                    # null_pte_name = os.path.join(null_pte_name, self.params.psx_subdir)
                    
                    if not os.path.exists(null_pte_name):
                        os.makedirs(null_pte_name, exist_ok = True)
                    
                    self.tabulate_null_PTEs(
                        PTE_data_coadd_numeric_1d, 
                        PTE_data_coadd_numeric,
                        PTE_data_coadd_1d, 
                        PTE_data_coadd,
                        null_pte_name,
                    )

                outfile.create_dataset("angle2Mpc", data = self.angle2Mpc)
                outfile.create_dataset("dx", data = self.map_dx)
                outfile.create_dataset("dy", data = self.map_dy)
                outfile.create_dataset("dz", data = self.map_dz)

                outfile["angle2Mpc"].attrs["unit"] = "Mpc/arcmin"
                outfile["dx"].attrs["unit"] = "Mpc"
                outfile["dy"].attrs["unit"] = "Mpc"
                outfile["dz"].attrs["unit"] = "Mpc"

                if self.params.psx_white_noise_sim_seed is not None:
                    outfile.create_dataset("white_noise_seed", data = self.params.psx_white_noise_sim_seed)
                
                for key in vars(
                    self.params
                ):  # Writing entire parameter file to separate hdf5 group.
                    if (
                        getattr(self.params, key) == None
                    ):  # hdf5 didn't like the None type.
                        outfile[f"params/{key}"] = "None"
                    else:
                        outfile[f"params/{key}"] = getattr(self.params, key)
    
    def grid_combo_plot(self):
        
        beam_tf = self.beam_transfer_function() 
        
        if self.params.psx_mode == "saddlebag":
            average_spectrum_dir = os.path.join(self.power_spectrum_dir, "average_spectra_saddlebag")
        else:
            average_spectrum_dir = os.path.join(self.power_spectrum_dir, "average_spectra")

        fig_dir = os.path.join(self.power_spectrum_dir, "figs")

        if not os.path.exists(average_spectrum_dir):
            os.mkdir(average_spectrum_dir)
    
        # fig_dir = os.path.join(fig_dir, self.params.psx_subdir)
        if not os.path.exists(fig_dir):
            os.mkdir(fig_dir)

        if self.params.psx_mode == "saddlebag":
            if not os.path.exists(os.path.join(fig_dir, "chi2_grid_saddlebag")):
                os.mkdir(os.path.join(fig_dir, "chi2_grid_saddlebag"))

            if not os.path.exists(os.path.join(fig_dir, "average_spectra_saddlebag")):
                os.mkdir(os.path.join(fig_dir, "average_spectra_saddlebag"))
            if self.params.psx_null_diffmap:
                if not os.path.exists(os.path.join(fig_dir, "null_pte_saddlebag")):
                    os.mkdir(os.path.join(fig_dir, "null_pte_saddlebag"))
        else:
            if not os.path.exists(os.path.join(fig_dir, "chi2_grid")):
                os.mkdir(os.path.join(fig_dir, "chi2_grid"))

            if not os.path.exists(os.path.join(fig_dir, "average_spectra")):
                os.mkdir(os.path.join(fig_dir, "average_spectra"))
                
            if self.params.psx_null_diffmap:
                if not os.path.exists(os.path.join(fig_dir, "null_pte")):
                    os.mkdir(os.path.join(fig_dir, "null_pte"))
        
        N_feed = len(self.included_feeds)
        N_splits = len(self.split_map_combinations) // 3
        N_k = self.params.psx_number_of_k_bins

        if self.params.psx_noise_sim_number <= 0 and len(self.params.psx_rnd_file_list) == 0:
            raise ValueError("Cannot compute average power spectrum without noise simulations or an RND ensemble.")
        elif len(self.params.psx_rnd_file_list) > 0:
            import glob
            rndsubdir = "rnd_split_files"
            if self.params.psx_mode == "saddlebag":
                rndsubdir += "_saddlebag"
            rndpath = os.path.join(self.params.power_spectrum_dir, rndsubdir)
            
            for num_rnd, rndfile_name in enumerate(self.params.psx_rnd_file_list):
                rndfile = os.path.join(rndpath, f"{self.params.fields[0]}_{rndfile_name}")
                print("Loading RND-file from: ", rndfile)
                rndfile = glob.glob(os.path.join(rndfile, f"*.h5"))[0]
                
                with h5py.File(rndfile, "r") as infile:
                    rnd_spectra = infile["all_spectra"][()] 
                    rnd_overlap = infile["all_overlap"][()] 
                    print(np.any(rnd_spectra == 0), *np.where(rnd_spectra == 0), np.unique(np.where(rnd_spectra == 0)[1]), (np.sum(rnd_spectra == 0)) / rnd_spectra.size * 100, rnd_spectra.shape)
                if num_rnd == 0:
                    all_rnd_spectra = rnd_spectra
                    all_rnd_overlap = rnd_overlap
                else:
                    all_rnd_spectra = np.concatenate((all_rnd_spectra, rnd_spectra), axis = 0)
                    all_rnd_overlap = np.concatenate((all_rnd_overlap, rnd_overlap), axis = 0)
            
                
            all_rnd_overlap = np.nanmedian(all_rnd_overlap, axis = 0)
            
            if np.all(all_rnd_spectra == 0) or np.any(~np.isfinite(all_rnd_spectra)):
                raise ValueError("All loaded RND spectra are either zero or infinite/NaN!")
            if np.any(~np.isfinite(all_rnd_spectra)):
                raise ValueError("All loaded RND spectra contain infinite/NaN!")
                
            all_rnd_spectra[all_rnd_spectra == 0] = np.nan
            
            ### 60-120 ###
            all_rnd_std = np.nanstd(all_rnd_spectra[:all_rnd_spectra.shape[0] // 4], axis = 0, ddof = 1)
            all_rnd_mean = np.nanmean(all_rnd_spectra[:all_rnd_spectra.shape[0] // 4], axis = 0)
            all_rnd_spectra = all_rnd_spectra[all_rnd_spectra.shape[0] // 4:]
            
            self.params.psx_noise_sim_number = all_rnd_spectra.shape[0]
            
        
        for map1, map2 in self.field_combinations:
            # Generate name of outpute data directory
            mapname1 = map1.split("/")[-1]
            mapname2 = map2.split("/")[-1]
            
            if self.params.psx_null_cross_field:
                indir = f"{mapname1[:-3]}_X_{mapname2[:-3]}"
            else:
                indir = f"{mapname1[:-3]}"
            
            outdir = os.path.join(average_spectrum_dir, indir)
            if self.params.psx_generate_white_noise_sim:
                #outdir_data = f"{outdir}"
                outdir = f"{outdir}/white_noise_seed{self.params.psx_white_noise_sim_seed}"
                        
            if self.params.psx_generate_white_noise_sim:
                #indir_data = f"{indir}"
                indir = f"{indir}/white_noise_seed{self.params.psx_white_noise_sim_seed}"

            if not os.path.exists(outdir):
                os.mkdir(outdir)

            xs_all = np.zeros((N_splits, 2, N_feed, 2, N_feed, N_k, N_k))
            xs_error_all = np.zeros((N_splits, 2, N_feed, 2, N_feed, N_k, N_k))

            cross_variable_names = [] 

            
            # if self.params.psx_null_diffmap:
            #     if self.params.psx_mode == "saddlebag":
            #         self.params.psx_chi2_import_path = re.sub(r"average_spectra", "average_spectra_saddlebag", self.params.psx_chi2_import_path)

            #     self.params.psx_chi2_import_path = re.sub(r"co\d_", f"{self.params.fields[0]}_", self.params.psx_chi2_import_path)
            #     if len(self.params.psx_chi2_import_path) <= 0 or not os.path.exists(self.params.psx_chi2_import_path):
            #         raise ValueError("No chi2 import file provided to perform null test chi2 cuts!")
                    
            split_counter = 0
            # for i, splits in enumerate(self.split_map_combinations):
            for i in range(N_splits):
                splits = self.split_map_combinations[split_counter]
                split_counter += 3

                for cross_variable1, cross_variable2 in [[0, 1], [0, 0], [1, 1]]:

                    if not self.params.psx_null_diffmap:
                        cross_variable = splits[0].split("/")[1]                    
                        cross_variable_names.append(cross_variable)
                        splits = list(splits)
                        splits = tuple([re.sub(rf"{cross_variable}\d", rf"{cross_variable}{cross_variable1}", splits[s]) for s in range(2)])
                        splits = (
                            re.sub(rf"{cross_variable}\d", rf"{cross_variable}{cross_variable1}", splits[0]), 
                            re.sub(rf"{cross_variable}\d", rf"{cross_variable}{cross_variable2}", splits[1])
                            )
                        splits = tuple(splits)
                        
                    else:
                        cross_variable = splits[0][0].split("/")[-1]
                        cross_variable = cross_variable[-5:-1]
                        splits = list(splits)
                        splits[0] = tuple([re.sub(rf"{cross_variable}\d", rf"{cross_variable}{cross_variable1}", splits[0][s]) for s in range(2)])
                        splits[1] = tuple([re.sub(rf"{cross_variable}\d", rf"{cross_variable}{cross_variable2}", splits[0][s]) for s in range(2)])
                        splits = tuple(splits)

                    for feed1 in range(N_feed):
                        for feed2 in range(N_feed):
                            if cross_variable2 < cross_variable1:
                                continue
                            elif cross_variable2 == cross_variable1 and feed2 < feed1:
                                continue

                            cross_spectrum = xs_class.CrossSpectrum_nmaps(
                                self.params, 
                                splits, 
                                self.included_feeds[feed1], 
                                self.included_feeds[feed2],
                            )   

                            try:
                                cross_spectrum.read_spectrum(indir)
                            except (FileNotFoundError, KeyError):
                                print(f"\033[95m WARNING: Split {splits[0]} or {splits[1]} not found in map file. Skipping split in averaging.\033[00m")
                                continue            
                            
                            cross_spectrum.read_and_append_attribute(["white_noise_simulation", "IoU", "weighted_overlap", "dx", "dy", "dz"], indir)
                            
                            if len(self.params.psx_rnd_file_list) > 0:
                                xs_wn = all_rnd_spectra[:, cross_variable1, cross_variable2, feed1, feed2, ...]
                                xs_sigma = all_rnd_std[cross_variable1, cross_variable2, feed1, feed2, ...]
                            else:
                                xs_wn = cross_spectrum.white_noise_simulation
                                xs_sigma = cross_spectrum.rms_xs_std_2D
                                # Applying white noise transfer function
                                transfer_function_wn = self.transfer_function_wn_interp(k_bin_centers_perp, k_bin_centers_par)
                                
                                xs_sigma *= transfer_function_wn 
                                xs_wn *= transfer_function_wn[None, ...]
                            
                            xs = cross_spectrum.xs_2D

                            if cross_variable1 == cross_variable2 and feed1 == feed2:
                                xs -= all_rnd_mean[cross_variable1, cross_variable2, feed1, feed2, ...]
                            
                            xs_all[i, cross_variable1, feed1, cross_variable2, feed2, ...] = xs
                            xs_error_all[i, cross_variable1, feed1, cross_variable2, feed2, ...] = xs_sigma
                            
                            k_bin_centers_perp, k_bin_centers_par  = cross_spectrum.k
                            
                            k_bin_edges_par = cross_spectrum.k_bin_edges_par
                            
                            k_bin_edges_perp = cross_spectrum.k_bin_edges_perp
                
                # Filter transfer function
                transfer_function = np.abs(self.full_transfer_function_interp(k_bin_centers_perp, k_bin_centers_par))
                
                # Beam and voxel window transfer functions
                px_window = self.pix_window(k_bin_centers_perp, cross_spectrum.dx)
                freq_window = self.pix_window(k_bin_centers_par, cross_spectrum.dz)
                transfer_function *= beam_tf(k_bin_centers_perp)[:, None] * px_window[:, None] 
                transfer_function *= freq_window[None, :] 
                        
                self.angle2Mpc = cross_spectrum.angle2Mpc * u.Mpc / u.arcmin

                self.map_dx = cross_spectrum.dx
                self.map_dy = cross_spectrum.dy
                self.map_dz = cross_spectrum.dz

                if not self.params.psx_generate_white_noise_sim:

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
                        average_name = os.path.join(fig_dir, "average_spectra_saddlebag")
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
            
                self.plot_feed_grid_all(
                    k_bin_edges_par,
                    k_bin_edges_perp,
                    k_bin_centers_par,
                    k_bin_centers_perp,
                    xs_all[i, ...] / transfer_function,
                    xs_error_all[i, ...] / transfer_function,
                    splits,
                    average_name,
                )
                    
    
    def tabulate_null_PTEs(self, 
                    PTE_numeric_1d: npt.NDArray,
                    PTE_numeric_2d: npt.NDArray,
                    PTE_fit_1d: npt.NDArray,
                    PTE_fit_2d: npt.NDArray,
                    filename: str,
                    ):
        
        from astropy.table import Table
        from astropy.io import ascii
        import shutil
        
        names = [
            r"Null variables", 
            r"1D numeric", 
            r"1D from fit", 
            r"2D numeric", 
            r"2D from fit",
            ]

        null_names = np.array(self.primary_variables).astype(bytes)
        cross_variable = self.secondary_variables[0]
        
        outname = os.path.join(
            filename, 
            self.params.psx_subdir,
        )

        if not os.path.exists(outname):
            os.makedirs(outname, exist_ok = True)

        outname = os.path.join(
            outname,
            f"null_pte",
        )
        
        print("hei pte", PTE_numeric_1d.shape, PTE_numeric_1d.shape)
        
        table = Table(
            [
                null_names,
                np.round(100 * PTE_numeric_1d, 2).astype(str),
                (100 * PTE_fit_1d).astype(str),
                np.round(100 * PTE_numeric_2d, 2).astype(str),
                (100 * PTE_fit_2d).astype(str),

            ],
            names = names
        )
        
        # table.pprint_all()
        
        with open(f"{outname}_table.tex", mode = "w") as outfile:
            outfile.write("PTEs:\n")
            outfile.write(f"cross_variable: {cross_variable}\n")
            
            outfile.write(f"\n")
            table.write(outfile, overwrite = True, format = "latex", col_align='c|cccc', latexdict=ascii.latex.latexdicts['AA'], formats = {})
            string = f"KS PTE & {np.round(100 * scipy.stats.ks_1samp(PTE_numeric_1d[np.isfinite(PTE_numeric_1d)], scipy.stats.uniform.cdf).pvalue, 1)} & & {np.round(100 * scipy.stats.ks_1samp(PTE_numeric_2d[np.isfinite(PTE_numeric_2d)], scipy.stats.uniform.cdf).pvalue, 1)} & \n"
            outfile.write(string)
            
            
            
            
        html_table = table.copy() 
        html_base = """
        <!DOCTYPE html>
        <html>
            <head>
                <style>
                    red {
                    color: red;
                    font-size: 100%;
                    }
                    orange {
                    color: orange;
                    font-size: 100%;
                    }
                </style>
                <meta charset="utf-8"/>
                <meta content="text/html;charset=UTF-8" http-equiv="Content-type"/>
            </head>
            <body>
        """

        html_base_end = """
                </table>
            </body>
        </html>
        """
        html_base += "PTEs:\n" + f"cross_variable: {cross_variable}\n"
        html_base += "<table><thead><tr><th>" + "</th><th>".join(names) + "</th></tr></thead>\n"
        for j, row in enumerate(html_table):
            html_base += "<tr>"
            for i, _ in enumerate(row):
                val = html_table[j][i]
                if np.any([s.isalpha() for s in val]):
                    html_base += fr'<td><b>{val}</b></td>'
                elif float(val) <= 1.0:
                    html_base += fr'<td><red>{float(val):.5g}</red></td>'
                elif float(val) >= 99.5:
                    html_base += fr'<td><orange>{float(val):.5g}</orange></td>'
                else:
                    html_base += fr'<td>{float(val):.5g}</td>'
                    
            html_base += "</tr>\n"

        ks_1d = np.round(100 * scipy.stats.ks_1samp(PTE_numeric_1d[np.isfinite(PTE_numeric_1d)], scipy.stats.uniform.cdf).pvalue, 1)
        ks_2d = np.round(100 * scipy.stats.ks_1samp(PTE_numeric_2d[np.isfinite(PTE_numeric_2d)], scipy.stats.uniform.cdf).pvalue, 1)
        
        if float(ks_1d) <= 1.0:
            ks_1d = fr'<td><red>{float(ks_1d):.5g}</red></td>'
        elif float(ks_1d) >= 99.5:
            ks_1d = fr'<td><orange>{float(ks_1d):.5g}</orange></td>'
        else:
            ks_1d = fr'<td>{ks_1d}</td>'
        
        if float(ks_2d) <= 1.0:
            ks_2d = fr'<td><red>{float(ks_2d):.5g}</red></td>'
        elif float(ks_2d) >= 99.5:
            ks_2d = fr'<td><orange>{float(ks_2d):.5g}</orange></td>'
        else:
            ks_2d = fr'<td>{ks_2d}</td>'
        
        html_base += ("<tr>"  
                    + f"<td>KS PTE</td>"
                    + f"{ks_1d}"  
                    + f"<td>N/A</td>"
                    + f"{ks_2d}"
                    + f"<td>N/A</td>"
                    + "</tr>")
        
        html_base += html_base_end
        html_name = f"{outname}_table.html"
        with open(html_name, mode = "w") as outfile:
            outfile.write(html_base)
        
        www_tsih3_path = "/mn/stornext/d16/www_cmb/comap/power_spectrum"
        move_to = os.path.join(www_tsih3_path, html_name.split(os.path.join(self.power_spectrum_dir, "figs/"))[-1])
        move_to_dir = os.path.dirname(move_to) 

        if not os.path.exists(move_to_dir):
            os.makedirs(move_to_dir, exist_ok = True)
            
        shutil.copyfile(
            html_name, 
            move_to,
        )

        
        #######################
        #### PLOT PTE HIST ####
        #######################
        fig, ax = plt.subplots(figsize = (9, 5))

        bins = np.linspace(0, 1, 7)
        offset = np.linspace(-0.01, 0.01, 4)

        ax.hist(PTE_numeric_1d, bins = bins + offset[0], histtype = "step", density = True, label = "1D", lw = 3, alpha = 1, color = "k")
        ax.hist(PTE_numeric_2d, bins = bins + offset[1], histtype = "step", density = True, label = "2D", lw = 3, alpha = 1, color = "r")

        ax.set_ylabel(r"$P(\mathrm{PTE})$", fontsize = 14)
        ax.set_xlabel(r"$\mathrm{PTE}$", fontsize = 14)

        ax.set_xticks(np.round(np.linspace(0, 1.0, 6), 1))
        ax.set_xticklabels(ax.get_xticks(), fontsize = 14, rotation = 0)


        ax.legend(frameon = False, ncols = 2, fontsize = 14, loc = "upper right")#, bbox_to_anchor = (0.65, -0.01))
        ax.set_xlim(-0.015, 1.015)

            
        #plt.figlegend(lns[:4], labs[:4], loc = 'lower center', ncol=2, fontsize = 14, bbox_to_anchor = (0.51, 0.615), bbox_transform = fig.transFigure, frameon = False)
        #plt.figlegend(lns[4:-1], labs[4:-1], loc = 'lower center', ncol=2, fontsize = 14, bbox_to_anchor = (0.51, 0.365), bbox_transform = fig.transFigure, frameon = False)
        fig.savefig(f"{outname}_histogram.pdf", facecolor = "white", bbox_inches = "tight")

    def plot_feed_grid_all(self,
                    k_bin_edges_par: npt.NDArray,
                    k_bin_edges_perp: npt.NDArray,
                    k_bin_centers_par: npt.NDArray,
                    k_bin_centers_perp: npt.NDArray,
                    xs: npt.NDArray,
                    xs_sigma: npt.NDArray,
                    splits: Sequence[str],
                    outname: str,
                    ):
        
        
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

        outname = os.path.join(
            outname, 
            self.params.psx_subdir,
            f"xs_full_grid_2d_{split1[:-1]}_X_{split2[:-1]}.png"
        )

        if not os.path.exists(os.path.dirname(outname)):
            os.makedirs(os.path.dirname(outname), exist_ok = True)
            
        N_feed = len(self.included_feeds)
        
        _, N_feed, _, N_feed, N_kx, N_ky = xs.shape
        
        xs = xs.reshape(2 * N_feed, 2 * N_feed, N_kx, N_ky)
        xs = xs.transpose(1, 0, 2, 3)
        
        xs_sigma = xs_sigma.reshape(2 * N_feed, 2 * N_feed, N_kx, N_ky)
        xs_sigma = xs_sigma.transpose(1, 0, 2, 3)
        
        
        auto_feeds = []
        c = 0
        for k in range(2):
            for i in range(N_feed):
                for l in range(2):        
                    for j in range(N_feed):
                        if (i == j and k != l) and k>l: 
                            auto_feeds.append(c)
                        c += 1        
        
        fig = plt.figure(figsize = (16, 15))
        gs = GridSpec(N_feed * 2, N_feed * 2, figure=fig, wspace = 0.08, hspace = 0.08)
        axes = []

        suptitle = f"Map file: {self.params.fields[0]}_{self.params.map_name}{self.params.psx_map_name_postfix}.h5\n"
        suptitle += f"Cross: {split1[:-1]}"
        
        if self.params.psx_null_diffmap:
            if len(self.params.psx_subdir) > 0:
                nullvar = outname.split("/")[-3]
            else:
                nullvar = outname.split("/")[-2]
                
            suptitle += f" | Null: {nullvar}"
        
        if len(self.params.psx_subdir) > 0: 
            suptitle += f" | Plot subdir: {self.params.psx_subdir}"
        
        fig.suptitle(suptitle, fontsize = 20, y = 0.92)
        
        # lim1 = []
        # lim2 = []
        # for i in range(2 * N_feed):
        #     lim2.append(0.2 * np.nanmax(np.abs(xs[i, i] / xs_sigma[i, i])))
        #     for j in range(2 * N_feed):
        #         if i != j:
        #             lim1.append(0.2 * np.nanmax(np.abs(xs[i, j] / xs_sigma[i, j])))
        
        # lim1.append(3)
        # lim2.append(8)

        # lim1 = np.nanmax(lim1)
        # lim2 = np.nanmax(lim2)
        
        # norm_significance = matplotlib.colors.Normalize(vmin=-lim1, vmax=lim1)
        # norm_significance2 = matplotlib.colors.Normalize(vmin=-lim2, vmax=lim2)
        
        
        pbar1 = tqdm.tqdm(
                total = N_feed * 2, 
                colour = "green", 
                ncols = 80,
                desc = f"Total",
                position = 0,
            )
        pbar2 = tqdm.tqdm(
                total = N_feed * 2, 
                colour = "blue", 
                ncols = 80,
                desc = f"Total",
                position = 1,
                # leave = False,
            )
        

        axes = []
        axes_auto = []
        axes_cross = []
        imgs = []
        
        # cmap = "RdGy_r"
        cmap = "RdBu_r"
        
        X_perp, X_par = np.meshgrid(k_bin_centers_perp, k_bin_centers_par)                    

        for i in range(N_feed * 2):
            pbar2.reset()
            for j in range(N_feed * 2):
                
                if i >= j:
                    

                    ax = fig.add_subplot(gs[i, j])
                    X_perp, X_par = np.meshgrid(k_bin_centers_perp, k_bin_centers_par)
                    
                    # print(i, j, (xs / xs_sigma)[i, j])
                    if i != j:
                        axes_cross.append(ax)
                        lim = np.nanmax((3, 2 * np.nanstd(np.abs(xs[i, j] / xs_sigma[i, j]))))
                    else:
                        lim = np.nanmax((8, 2 * np.nanstd(np.abs(xs[i, i] / xs_sigma[i, i]))))
                        axes_auto.append(ax)
                    
                    norm =  matplotlib.colors.Normalize(vmin=-lim, vmax=lim)
                    
                    
                    if (i >= N_feed and j < N_feed) and (i * (2 * N_feed) + j not in auto_feeds):
                        ax.spines[:].set_color("k")
                        ax.spines[:].set_linewidth(5)
                    if i * (2 * N_feed) + j in auto_feeds:
                        ax.spines[:].set_color("fuchsia")
                        ax.spines[:].set_linewidth(3)
                    elif i == j:
                        ax.spines[:].set_color("m")
                        ax.spines[:].set_linewidth(3)
                        
                    img = ax.pcolormesh(
                        X_perp, 
                        X_par,
                        (xs / xs_sigma)[i, j].T,
                        cmap=cmap,
                        norm = norm,
                        zorder = 1,
                        rasterized = True,
                    )
                    imgs.append(img)
                    ax.set_yscale("log")
                    ax.set_xscale("log")

                    ticks = [0.03, 0.1, 0.3, 1]
                    ticklabels = ["0.03", "0.1", "0.3", "1"]

                    ylabels = ticklabels
                    xlabels = ticklabels
                    
                    # divider = make_axes_locatable(ax)
                    # cax = divider.append_axes("right", size="5%", pad=0.05)
                    # cbar = plt.colorbar(img, cax = cax)
                    
                    # if i == j: 
                    #     cbar.set_label(
                    #         r"$\tilde{C}/\sigma\left(k_{\bot},k_{\parallel}\right)$",
                    #         size=11,
                    #     )
                    # cticks = np.linspace(-np.round(lim * 0.75), np.round(lim * 0.75), 3)
                    # cbar.set_ticks(cticks)
                    # cbar.set_ticklabels(cticks, rotation = 90, fontsize = 9)
                    
                    # ax.text(0.03, 0.87, fr"$\pm${np.round(lim)}$\sigma$", transform = ax.transAxes, color = "k")
                    text = ax.text(0.95, 0.05, fr"$\pm${int(np.round(lim))}$\sigma$", transform = ax.transAxes, color = "k", fontsize = 20, horizontalalignment='right')
                    #ax.text(0.5, 1.1, fr"$\pm${np.round(lim)}$\sigma$", transform = ax.transAxes, color = "k")
                    text.set_path_effects([path_effects.Stroke(linewidth=3, foreground = 'white'), path_effects.Normal()])
                    
                    if j > 0:
                        ylabels = []
                    else:
                        ax.set_ylabel(f"Feed {i % N_feed + 1}")
                    
                    if i < (N_feed * 2) - 1:
                        xlabels = []
                    else:
                        ax.set_xlabel(f"Feed {j % N_feed + 1}", rotation = 90)

                    ax.set_xticks(ticks)
                    ax.set_xticklabels(xlabels, fontsize = 10, rotation = 90)
                    ax.set_yticks(ticks)
                    ax.set_yticklabels(ylabels, fontsize = 10)

                    ax.set_ylim(k_bin_edges_par[0], k_bin_edges_par[-1])
                    ax.set_xlim(k_bin_edges_perp[0], k_bin_edges_perp[-1])

                    axes.append(ax)
                else:
                    continue
                
                pbar2.update(1)
            pbar1.update(1)

        pbar1.close()
        pbar2.close()

        fig.text(0.05, 0.65, f"Feeds, low {split1[:-1]}", fontsize = 15, rotation = 90, transform = fig.transFigure)
        fig.text(0.05, 0.25, f"Feeds, high {split2[:-1]}", fontsize = 15, rotation = 90, transform = fig.transFigure)

        fig.text(0.65, 0.03, f"Feeds, high {split2[:-1]}", fontsize = 15, rotation = 0, transform = fig.transFigure)
        fig.text(0.25, 0.03, f"Feeds, low {split1[:-1]}", fontsize = 15, rotation = 0, transform = fig.transFigure)


        fig.text(0.01, 0.4, r"$k_\parallel$ [Mpc${}^{-1}$]", rotation = 90, fontsize=16, transform = fig.transFigure)
        fig.text(0.4, 0.01, r"$k_\bot$ [Mpc${}^{-1}$]", fontsize=16, transform = fig.transFigure)
        
        # axins1 = inset_axes(
        #     axes[-1],
        #     width="2.5%",  # width: 5% of parent_bbox width
        #     height="40%",  # height: 50%
        #     loc="center left",
        #     bbox_to_anchor=(0.83, 0., 1, 1),
        #     bbox_transform=fig.transFigure,
        #     borderpad=0,
        # )            
        
        # axins2 = inset_axes(
        #     axes[-2],
        #     width="2.5%",  # width: 5% of parent_bbox width
        #     height="40%",  # height: 50%
        #     loc="center left",
        #     bbox_to_anchor=(0.86, 0., 1, 1),
        #     bbox_transform=fig.transFigure,
        #     borderpad=0,
        # )            
        
        # cbar1 = fig.colorbar(imgs[-1], cax=axins1, ticklocation = 'left')
        # cbar2 = fig.colorbar(imgs[-2], cax=axins2, ticklocation = 'right')
        # cbar1.set_label(
        #     #r"$\tilde{C}\left(k_{\bot},k_{\parallel}\right)$ [$\mu$K${}^2$ (Mpc)${}^3$]",
        #     r"$\tilde{C}/\sigma\left(k_{\bot},k_{\parallel}\right)$ diagonal",
        #     size=16,
        # )
        # cbar2.set_label(
        #     #r"$\tilde{C}\left(k_{\bot},k_{\parallel}\right)$ [$\mu$K${}^2$ (Mpc)${}^3$]",
        #     r"$\tilde{C}/\sigma\left(k_{\bot},k_{\parallel}\right)$ off-diagonal",
        #     size=16,
        # )
        fig.savefig(outname, bbox_inches = "tight", dpi = 200)
        
    def plot_feed_grid(self,
                    k_bin_edges_par: npt.NDArray,
                    k_bin_edges_perp: npt.NDArray,
                    k_bin_centers_par: npt.NDArray,
                    k_bin_centers_perp: npt.NDArray,
                    xs: npt.NDArray,
                    xs_sigma: npt.NDArray,
                    transfer_function_mask: npt.NDArray,
                    splits: Sequence[str],
                    outname: str,
                    ):
        
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

        outname = os.path.join(
            outname, 
            self.params.psx_subdir,
            f"xs_grid_2d_{split1}_X_{split2}.jpg"
        )
        
        if not os.path.exists(os.path.dirname(outname)):
            os.makedirs(os.path.dirname(outname), exist_ok = True)
            
        N_feed = len(self.included_feeds)
        fig = plt.figure(figsize = (16, 16))
        # fig_significance = plt.figure(figsize = (single_width, 9))
        # fig_sigma = plt.figure(figsize = (single_width, 9))
        gs = GridSpec(N_feed, N_feed, figure=fig, wspace = 0.08, hspace = 0.08)
        # gs_significance = GridSpec(N_feed, N_feed, figure=fig, wspace = 0.00, hspace = 0.00)
        # gs_sigma = GridSpec(N_feed, N_feed, figure=fig, wspace = 0.00, hspace = 0.00)
        axes = []

        limit_idx = int(np.round(3 / 14 * xs.shape[0])) 

        if limit_idx != 0:
            lim = np.nanmax(np.abs(xs[..., limit_idx:-limit_idx, limit_idx:-limit_idx]))
            lim_error = np.nanmax(xs_sigma[..., limit_idx:-limit_idx, limit_idx:-limit_idx])
            lim_significance = np.nanmax(np.abs((xs / xs_sigma)[..., limit_idx:-limit_idx, limit_idx:-limit_idx]))
        else:
            lim = np.percentile(np.abs(xs), 80)
            # lim = np.nanmax(np.abs(xs))
            lim_error = np.nanmax(xs_sigma)
            lim_significance = np.nanmax(np.abs((xs / xs_sigma)))

        norm = matplotlib.colors.Normalize(vmin=-lim, vmax=lim)
        # norm = matplotlib.colors.Normalize(vmin=-3e4, vmax=3e4)
        lim_error = matplotlib.colors.Normalize(vmin=0, vmax=lim_error)
        norm_significance = matplotlib.colors.Normalize(vmin=-3, vmax=3)
        # lim_significance = matplotlib.colors.Normalize(vmin=-lim_significance, vmax=lim_significance)
        
        
        pbar1 = tqdm.tqdm(
                total = N_feed, 
                colour = "green", 
                ncols = 80,
                desc = f"Total",
                position = 0,
            )
        pbar2 = tqdm.tqdm(
                total = N_feed, 
                colour = "blue", 
                ncols = 80,
                desc = f"Total",
                position = 1,
                # leave = False,
            )
        
        for i in range(N_feed):
            pbar2.reset()
            for j in range(N_feed):
                ax = fig.add_subplot(gs[i, j])
                # ax.text(0.1, 0.5, f"i={i},j={j},\nidx={i * 8 + j}", transform = ax.transAxes)
                        
                X_perp, X_par = np.meshgrid(k_bin_centers_perp, k_bin_centers_par)
        
                img = ax.pcolormesh(
                    X_perp, 
                    X_par,
                    (xs / xs_sigma)[i, j].T,
                    #transfer_function.T,
                    cmap="RdBu_r",
                    norm = norm_significance,
                    zorder = 1,
                    rasterized = True,
                )
                
                ax.set_yscale("log")
                ax.set_xscale("log")

                ticks = [0.03, 0.1, 0.3, 1]
                ticklabels = ["0.03", "0.1", "0.3", "1"]

                ylabels = ticklabels
                xlabels = ticklabels
                
                
                if j > 0:
                    ylabels = []
                else:
                    ax.set_ylabel(f"Feed {i + 1}")
                
                if i < N_feed - 1:
                    xlabels = []
                else:
                    ax.set_xlabel(f"Feed {j + 1}", rotation = 90)

                ax.set_xticks(ticks)
                ax.set_xticklabels(xlabels, fontsize = 10, rotation = 90)
                ax.set_yticks(ticks)
                ax.set_yticklabels(ylabels, fontsize = 10)

                ax.set_ylim(k_bin_edges_par[0], k_bin_edges_par[-1])
                ax.set_xlim(k_bin_edges_perp[0], k_bin_edges_perp[-1])

                axes.append(ax)
                pbar2.update(1)
            pbar1.update(1)

        pbar1.close()
        pbar2.close()

        fig.text(0.01, 0.5, r"$k_\parallel$ [Mpc${}^{-1}$]", rotation = 90, fontsize=16, transform = fig.transFigure)
        fig.text(0.4, 0.01, r"$k_\bot$ [Mpc${}^{-1}$]", fontsize=16, transform = fig.transFigure)
                    
        cbar = fig.colorbar(img, ax=axes)
        cbar.set_label(
            #r"$\tilde{C}\left(k_{\bot},k_{\parallel}\right)$ [$\mu$K${}^2$ (Mpc)${}^3$]",
            r"$\tilde{C}/\sigma\left(k_{\bot},k_{\parallel}\right)$",
            size=16,
        )
        fig.savefig(outname, bbox_inches = "tight")

        
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
                    summary_stats_list: list[float, float, float],
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
        matplotlib.rcParams["hatch.color"] = "green"
        matplotlib.rcParams["hatch.linewidth"] = 0.5
        
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
            self.params.psx_subdir,
            f"corr_2d_{split1}_X_{split2}.png"
            )
        
        outname = os.path.join(
            outname, 
            self.params.psx_subdir,
            f"xs_mean_2d_{split1}_X_{split2}.png"
        )
        
        if not os.path.exists(os.path.dirname(outname_corr)):
            os.makedirs(os.path.dirname(outname_corr), exist_ok = True)
            
        if not os.path.exists(os.path.dirname(outname)):
            os.makedirs(os.path.dirname(outname), exist_ok = True)
            
        # fig, ax = plt.subplots(1, 3, figsize=(16, 5.6))
        fig, ax = plt.subplots(2, 3, figsize=(16, 13))

        fig.suptitle(f"Fields: {fields[0]} X {fields[1]} | {split1} X {split2}", fontsize=16)
        
        limit_idx = int(np.round(3 / 14 * xs_mean.shape[0])) 

        if limit_idx != 0:
            lim = np.nanmax(np.abs(xs_mean[limit_idx:-limit_idx, limit_idx:-limit_idx]))
            lim_error = np.nanmax(xs_sigma[limit_idx:-limit_idx, limit_idx:-limit_idx])
            lim_significance = np.nanmax(np.abs((xs_mean / xs_sigma)[limit_idx:-limit_idx, limit_idx:-limit_idx]))
        else:
            lim = np.percentile(np.abs(xs_mean), 80)
            # lim = np.nanmax(np.abs(xs_mean))
            lim_error = np.nanmax(xs_sigma)
            lim_significance = np.nanmax(np.abs((xs_mean / xs_sigma)))

        norm = matplotlib.colors.Normalize(vmin=-lim, vmax=lim)
        # norm = matplotlib.colors.Normalize(vmin=-3e4, vmax=3e4)
        lim_error = matplotlib.colors.Normalize(vmin=0, vmax=lim_error)
        lim_significance = matplotlib.colors.Normalize(vmin=-3, vmax=3)
        # lim_significance = matplotlib.colors.Normalize(vmin=-lim_significance, vmax=lim_significance)
        
        k_contour_levels = np.logspace(-2.0, np.log10(1.5), len(k_bin_centers_perp) + 1)
        k_1d = np.logspace(-2.0, np.log10(1.5), 1000)
        K_x, K_y = np.meshgrid(k_1d, k_1d)
        for i in range(3):
            ax[0, i].contour(K_x, K_y, np.sqrt(K_x ** 2 + K_y ** 2), levels = k_contour_levels, colors = "gray", alpha = 0.4, zorder = 4)
        
        X_perp, X_par = np.meshgrid(k_bin_centers_perp, k_bin_centers_par)

        img1 = ax[0, 0].pcolormesh(
            X_perp, 
            X_par,
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
            hatch='xx', 
            transform = ax[0, 0].transAxes, 
            alpha = 0, 
            zorder = 2
        )

        xs_mean_masked = np.ma.masked_where(~transfer_function_mask, xs_mean)
        ax[0, 0].pcolormesh(
            X_perp,
            X_par,
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
            X_perp, 
            X_par,
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
            hatch='xx', 
            transform = ax[0, 1].transAxes, 
            alpha = 0, 
            zorder = 2
        )

        xs_sigma_masked = np.ma.masked_where(~transfer_function_mask, xs_sigma)
        ax[0, 1].pcolormesh(
            X_perp,
            X_par,
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
            X_perp, 
            X_par,
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
            hatch='xx', 
            transform = ax[0, 2].transAxes, 
            alpha = 0, 
            zorder = 2
        )

        ax[0, 2].pcolormesh(
            X_perp,
            X_par,
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

            ax[0, i].set_ylim(k_bin_edges_par[0], k_bin_edges_par[-1])
            ax[0, i].set_xlim(k_bin_edges_perp[0], k_bin_edges_perp[-1])


            ax[0, i].set_xlabel(r"$k_\bot$ [Mpc${}^{-1}$]", fontsize=16)
        
        ax[0, 0].set_ylabel(r"$k_\parallel$ [Mpc${}^{-1}$]", fontsize=16)


        majorticks = [0.03, 0.1, 0.3, 1]

        for i in range(3):
            ax2 = ax[0, i].twiny()
            ax2.set_xscale("log")

            ax2.set_xticks(majorticks)
            ax2.set_xticklabels(majorticks)
            
            ax2.set_xlim(k_bin_edges_perp[0], k_bin_edges_perp[-1])
            ax2.set_xticklabels(np.round(2 * np.pi / (np.array(majorticks) * self.angle2Mpc.value), 2).astype(str))
            ax2.set_xlabel(r"angular scale [$\mathrm{arcmin}$]", fontsize = 16)

        ###############################
        # chi2 of white noise spectra #
        ###############################
        ax[1, 0].axis("off")
        ax[1, 1].axis("off")
        ax[1, 2].axis("off")

        ax1 = fig.add_subplot(212)

        x_lim = [0, 200]
        x = np.linspace(x_lim[0], x_lim[1], 1000)
        
        if not np.all(~np.isfinite(chi2_wn_list[0])):
            ax1.hist(
                chi2_wn_list[0], 
                histtype = "step", 
                bins = int(np.round(chi2_wn_list[0].size * 0.2)), 
                density = True, 
                lw = 3,
                label = r"$\sum_i (d_i/\sigma_i)^2$",
            )
        # if not np.all(~np.isfinite(chi2_wn_list[1])):
        #     ax1.hist(
        #         chi2_wn_list[1],
        #         histtype = "step",
        #         bins = int(np.round(chi2_wn_list[1].size * 0.2)),
        #         density = True,
        #         lw = 3,
        #         label = r"$\mathbf{d}^T\mathbf{N}^{-1}\mathbf{d}$",
        #         )
            
        # chi2_analytical = chi2.pdf(x, df = np.sum(transfer_function_mask))
        chi2_analytical = chi2.pdf(x, df = self.df_2d_best_fit)
        # chi2_analytical = chi2.pdf(x + self.df_2d_best_fit, df = np.sum(transfer_function_mask))
        
        ax1.plot(
            x,
            chi2_analytical,
            color = "r",
            linestyle = "dashed",
            lw = 3,
            # label = rf"$\chi^2(dof = {np.sum(transfer_function_mask)})$",
            label = rf"$\chi^2(dof = {self.df_2d_best_fit:.2f})$",
        )
        
        chi2_sum, PTE, PTE_numeric = summary_stats_list
        # chi2_sum = np.nansum((xs_mean_masked / xs_sigma_masked) ** 2)
        # PTE = scipy.stats.chi2.sf(chi2_sum, df = np.sum(transfer_function_mask))
        
        if chi2_sum >= x_lim[0] and chi2_sum <= x_lim[1]:
            ax1.axvline(
                chi2_sum, 
                # label = rf"$\chi^2_\mathrm{{data}} = \sum_i (d_i/\sigma_i)^2$ = {chi2_sum:.3f},  " + f"dof: {np.sum(transfer_function_mask)}, " + rf"PTE: {PTE:.3f}, PTE numeric {PTE_numeric:.3f}",
                label = rf"$\chi^2_\mathrm{{data}} = \sum_i (d_i/\sigma_i)^2$ = {chi2_sum:.3f},  " + f"dof: {self.df_2d_best_fit:.2f}, " + rf"PTE: {PTE:.3f}, PTE numeric {PTE_numeric:.3f}",
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
                # label = rf"$\chi^2_\mathrm{{data}} = \sum_i (d_i/\sigma_i)^2$ = {chi2_sum:.3f},  " + f"dof: {np.sum(transfer_function_mask)}, " + rf"PTE: {PTE:.3f}",
                label = rf"$\chi^2_\mathrm{{data}} = \sum_i (d_i/\sigma_i)^2$ = {chi2_sum:.3f},  " + f"dof: {self.df_2d_best_fit:.2f}, " + rf"PTE: {PTE:.3f}, PTE numeric {PTE_numeric:.3f}",
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
                # label = rf"$\chi^2_\mathrm{{data}} = \sum_i (d_i/\sigma_i)^2$ = {chi2_sum:.3f},  " + f"dof: {np.sum(transfer_function_mask)}, " + rf"PTE: {PTE:.3f}",
                label = rf"$\chi^2_\mathrm{{data}} = \sum_i (d_i/\sigma_i)^2$ = {chi2_sum:.3f},  " + f"dof: {self.df_2d_best_fit:.2f}, " + rf"PTE: {PTE:.3f}, PTE numeric {PTE_numeric:.3f}",
                color = "k",
            )

        ax1.legend(fontsize = 16, ncol = 4, loc = "upper center")

        ax1.set_xlabel(r"$\chi^2$", fontsize = 16)
        ax1.set_ylabel(r"$P(\chi^2)$", fontsize = 16)
        ax1.set_xlim(x_lim)
        # ax1.set_yscale("log")

        fig.tight_layout()
        fig.savefig(outname, bbox_inches = "tight")
        

        fig, ax = plt.subplots(1, 1, figsize = (12, 8))
        plotdata = cov_wn / np.sqrt(np.outer(cov_wn.diagonal(), cov_wn.diagonal(),))
        plotdata[~transfer_function_mask.flatten(), :] = np.nan
        plotdata[:, ~transfer_function_mask.flatten()] = np.nan
        img = ax.imshow(
            plotdata,
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
                    overlap_grid: npt.NDArray,
                    chi2_wn_list: list[npt.NDArray, npt.NDArray],
                    summary_stats_list: list[float, float, float],
                    splits: Sequence[str],
                    fields: Sequence[str],
                    outname: str
                    ):
        
        """Method that plots 1D, i.e. spherically averaged mean FPXS.

        Args:
            k_1d (npt.NDArray): Array of k-bin centers in 1/Mpc
            xs_mean (npt.NDArray): Array of mean spherically averaged FPXS 
            xs_sigma (npt.NDArray): Array of errors of mean spherically averaged FPXS
            chi2_grid (npt.NDArray): Array of normalized chi2 values for each feed-split combo
            overlap_grid (npt.NDArray): Array weighted overlap values for each feed-split combo
            splits (Sequence[str]): Sequence of strings with split names used for as cross-spectrum variable
            fields (Sequence[str]): Sequence of strings with field names used in cross correlation
            outname (str): String with output directory for plot
        """


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
            self.params.psx_subdir,
            f"corr_1d_{split1}_X_{split2}.png"
        )
            
        # Add output name to output path
        outname = os.path.join(
            outname, 
            self.params.psx_subdir,
            f"xs_mean_1d_{split1}_X_{split2}.png"
            )
        
        if not os.path.exists(os.path.dirname(outname_corr)):
            os.makedirs(os.path.dirname(outname_corr), exist_ok = True)    
        if not os.path.exists(os.path.dirname(outname)):
            os.makedirs(os.path.dirname(outname), exist_ok = True)
        
        # Plot spherically averaged mean FPXS
        # Only want to use points between 0.04 and 1.0 /Mpc        
        # where = np.logical_and(k_1d > 0.1, k_1d < 1.0)
        where = np.logical_and(k_1d > 0.1, k_1d < 0.8)
        
        # Plot y-limits
        lim = np.nanmax(np.abs((k_1d * (xs_mean + xs_sigma))[where]))
        lim = 1.1 * np.nanmax((np.nanmax(np.abs((k_1d * (xs_mean - xs_sigma))[where])), lim))
        
        # lim = 2e4

        lim_significance = 1.1 * np.nanmax(np.abs((xs_mean / xs_sigma)[where]))
        
        mask = np.isfinite(xs_mean / xs_sigma)

        k_1d = k_1d[mask]
        xs_mean = xs_mean[mask]
        xs_sigma = xs_sigma[mask]
        
        if np.all(~mask):
            return 
                
        fig, ax = plt.subplots(3, 1, figsize=(16, 15))

        # Figure title
        fig.suptitle(f"Fields: {fields[0]} X {fields[1]} | {split1} X {split2}", fontsize=16)
        
        
        # Plot scatter and error bars
        ax[0].scatter(
            k_1d,
            k_1d * xs_mean,
            s = 80,
            label = r"$\sigma$ from coaddition",
        )

        ax[0].errorbar(
            k_1d,
            k_1d * xs_mean,
            k_1d * xs_sigma,
            lw = 3,
            fmt = " ",
        )

        # ax[0].scatter(
        #     k_1d - k_1d * 0.05,
        #     k_1d * xs_mean,
        #     s = 80,
        #     color = "r",
        #     label = r"$\sigma$ from $\mathrm{diag}(N)$",
        # )

        # ax[0].errorbar(
        #     k_1d - k_1d * 0.05,
        #     k_1d * xs_mean,
        #     k_1d * self._error_1d,
        #     lw = 3,
        #     fmt = " ",
        #     color = "r",
        # )

        if np.isfinite(lim):
            ax[0].set_ylim(-lim, lim)
        ax[0].set_ylabel(
            r"$k\tilde{C}(k)$ [$\mu$K${}^2$ (Mpc)${}^3$]",
            fontsize = 16,
        )
        
        
        chi2_sum, PTE, PTE_numeric = summary_stats_list
        
        # chi2_sum = np.nansum((xs_mean / xs_sigma) ** 2)
        
        # chi2_cdf = scipy.stats.chi2.cdf(chi2_sum, df = np.sum(np.isfinite((xs_mean))))

        # PTE = scipy.stats.chi2.sf(chi2_sum, df = np.sum(np.isfinite((xs_mean))))

        _chi2_grid = chi2_grid.copy()
        for i in range(_chi2_grid.shape[0]):
            _chi2_grid[i, i] = np.inf
        
        number_accepted_cross_spectra = np.abs(_chi2_grid) < self.params.psx_chi2_cut_limit
        number_accepted_cross_spectra = np.sum(np.logical_and(number_accepted_cross_spectra, overlap_grid > self.params.psx_overlap_limit))

        # ax[0].set_title(rf"# accepted $\chi^2 < {self.params.psx_chi2_cut_limit}$: {number_accepted_cross_spectra} / {_chi2_grid.size}" + " " * 5 + rf"$\chi^2 = \sum_i (d_i/\sigma_i)^2$: {chi2_sum:.3f}" + " " * 5 + f"dof: {np.sum(np.isfinite((xs_mean)))}" + " " * 5 + rf"PTE: {PTE:.3f} PTE numeric {PTE_numeric:.3f}", fontsize = 16)
        ax[0].set_title(rf"# accepted $\chi^2 < {self.params.psx_chi2_cut_limit}$: {number_accepted_cross_spectra} / {_chi2_grid.size}" + " " * 5 + rf"$\chi^2 = \sum_i (d_i/\sigma_i)^2$: {chi2_sum:.3f}" + " " * 5 + f"dof: {self.df_1d_best_fit:.2f}" + " " * 5 + rf"PTE: {PTE:.3f} PTE numeric {PTE_numeric:.3f}", fontsize = 16)
        
        
        if not self.params.psx_null_diffmap:
            # Upper limit from S1:
            # _k_1d = np.linspace(0.05, 1, 100)
            # ax[0].plot(_k_1d, 5.4e3 / 0.24 * _k_1d, linestyle = "dashed", alpha = 0.8, color = "k", lw = 3, label = r"95% $\mathrm{UL_{S1}}$")
            # ax[0].axhline(5.4e3, linestyle = "dashed", alpha = 0.8, color = "k", lw = 3, label = r"95% $\mathrm{UL_{S1}}$")
            
            all_pt_coadded = np.sum((xs_mean) / (xs_sigma) ** 2)
            all_pt_coadded /= np.sum(1 / xs_sigma ** 2)
            all_pt_error = np.sqrt(1 / np.sum(1 / (xs_sigma) ** 2))
            
            ax[0].errorbar(
                0.24 + 0.24 * 0.05,
                0.24 * all_pt_coadded,
                0.24 * all_pt_error,
                lw = 1,
                fmt = " ",
                color = "g",
            )

            ax[0].scatter(
                0.24 + 0.24 * 0.05,
                0.24 * all_pt_coadded,
                s = 8,
                color = "g",
                marker = "x",
                label = "coadded points"
            )
            
            ax[1].errorbar(
                0.24 + 0.24 * 0.05,
                all_pt_coadded / all_pt_error,
                1,
                lw = 3,
                fmt = " ",
                color = "g",
            )
            
            
            ax[1].scatter(
                0.24 + 0.24 * 0.05,
                all_pt_coadded / all_pt_error,
                s = 80,
                color = "g",
                marker = "x",
                label = "coadded points"
                )
        
        for name, model in self.models.items():
            ax[0].plot(model.k, model.interpolation(model.k) * model.k, label = name)
            
        ax[0].legend(ncols = 8)
        
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

        # ax[1].scatter(
        #     k_1d - k_1d * 0.05,
        #     xs_mean / self._error_1d,
        #     s = 80,
        #     color = "r",
        # )
        
        # ax[1].errorbar(
        #     k_1d - k_1d * 0.05,
        #     xs_mean / self._error_1d,
        #     1,
        #     lw = 3,
        #     fmt = " ",
        #     color = "r",
        # )
        
    
        
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
        klabels = [0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0]
        # klabels = [0.05, 0.1, 0.2, 0.5, 1.0]

        ax[0].set_xticks(klabels)
        ax[0].set_xticklabels(klabels, fontsize = 16)
        ax[1].set_xticks(klabels)
        ax[1].set_xticklabels(klabels, fontsize = 16)

        
        ax[0].set_xlim(k_1d.min() * 0.8, 1.0)
        ax[1].set_xlim(k_1d.min() * 0.8, 1.0)

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
        
        if not np.all(~np.isfinite(chi2_wn_list[0])):
            ax[2].hist(
                chi2_wn_list[0], 
                histtype = "step", 
                bins = int(np.round(chi2_wn_list[0].size * 0.2)), 
                density = True, 
                lw = 3,
                label = r"$\sum_i (d_i/\sigma_i)^2$",
            )

        chi2_analytical = scipy.stats.chi2.pdf(x, df = self.df_1d_best_fit)
        # chi2_analytical = scipy.stats.chi2.pdf(x + self.df_1d_best_fit, df = np.sum(np.isfinite(xs_mean / xs_sigma)))
        # chi2_analytical = scipy.stats.chi2.pdf(x, df = np.sum(np.isfinite(xs_mean / xs_sigma)))
        ax[2].plot(
            x,
            chi2_analytical,
            color = "r",
            linestyle = "dashed",
            lw = 3,
            # label = rf"$\chi^2(dof = {np.sum(np.isfinite(xs_mean / xs_sigma))})$",
            label = rf"$\chi^2(dof = {self.df_1d_best_fit:.2f})$",
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

        ax[2].legend(fontsize = 16, ncol = 4, loc = "upper center")

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

    def plot_overlap_stats(self, overlap_vs_rnd: npt.NDArray, weighted_overlap: npt.NDArray, chi2_mask: npt.NDArray, splits: tuple, outname: str):
        """Method that plots feed-split grid of FPXS chi-squared values

        Args:
            overlap_vs_rnd (npt.NDArray): Array of noise weighted map overlap quantities for all feed and split combinations vs that of RND splits (i.e. near perfect overlap)
            weighted_overlap (npt.NDArray): Array of noise weighted map overlap quantities for all feed and split combinations
            chi2_mask (npt.NDArray): Array of chi2 mask values for all feed and split combinations
            splits (tuple): Split names that were crossed
            outname (str): Plot file output path
        """

        # Define default ticks label sizes
        matplotlib.rcParams["xtick.labelsize"] = 10
        matplotlib.rcParams["ytick.labelsize"] = 10
        matplotlib.rcParams["hatch.color"] = "green"
        matplotlib.rcParams["hatch.linewidth"] = 0.5

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
            self.params.psx_subdir,
            f"overlap_grid_{split1}_X_{split2}.png"
            )
        
        if not os.path.exists(os.path.dirname(outname)):
            os.makedirs(os.path.dirname(outname), exist_ok = True)
            
        # Number of detectors used in cross-correlation combinations
        N_feed = len(self.included_feeds)

        # Define symetric collormap
        cmap = matplotlib.cm.CMRmap

        # Bad values, i.e. NaN and Inf, are set to black
        cmap.set_bad("lime", 1)

        overlap_vs_rnd_masked = np.ma.masked_where(~chi2_mask.astype(bool), overlap_vs_rnd)
        
        # Plot overlap value grid
        fig, ax = plt.subplots(2, 2, figsize = (13, 10))

        img = ax[0, 0].imshow(
            overlap_vs_rnd,
            interpolation = "none",
            extent = (0.5, N_feed + 0.5, N_feed + 0.5, 0.5),
            cmap = cmap,
            vmin = np.nanmin(overlap_vs_rnd) - 0.5 * np.nanstd(overlap_vs_rnd),
            vmax = np.nanmax(overlap_vs_rnd) + 0.2 * np.nanstd(overlap_vs_rnd),
            rasterized = True,
            zorder = 1,
        )

        ax[0, 0].fill_between(
            [0.5, N_feed + 0.5], 
            0.5, 
            N_feed + 0.5, 
            hatch='xx', 
            alpha = 0, 
            zorder = 2
        )
        
        for i in range(chi2_mask.shape[0]):
            chi2_mask[i, i] = False
        
        # chi2_masked = np.ma.masked_where(~(np.abs(chi2) < self.params.psx_chi2_cut_limit), chi2)

        ax[0, 0].imshow(
            overlap_vs_rnd_masked,
            interpolation="none",
            extent=(0.5, N_feed + 0.5, N_feed + 0.5, 0.5),
            cmap = "CMRmap",
            vmin = np.nanmin(overlap_vs_rnd) - 0.5 * np.nanstd(overlap_vs_rnd),
            vmax = np.nanmax(overlap_vs_rnd) + 0.2 * np.nanstd(overlap_vs_rnd),
            rasterized=True,
            zorder = 3,
        )

        new_tick_locations = np.array(range(N_feed)) + 1
        ax[0, 0].set_xticks(new_tick_locations)
        ax[0, 0].set_yticks(new_tick_locations)
        
        ax[0, 0].set_xlabel(f"Feed of {split1}")
        ax[0, 0].set_ylabel(f"Feed of {split2}")
        
        cbar = plt.colorbar(img, ax = ax[0, 0])
        
        cbar.set_label(r"Relative Noise Weigted Overlap")

        ####################################################

        img2 = ax[0, 1].imshow(
            weighted_overlap,
            interpolation = "none",
            extent = (0.5, N_feed + 0.5, N_feed + 0.5, 0.5),
            cmap = cmap,
            vmin = 0,
            vmax = 1,
            rasterized = True,
            zorder = 1,
        )

        ax[0, 1].fill_between(
            [0.5, N_feed + 0.5], 
            0.5, 
            N_feed + 0.5, 
            hatch='xx', 
            alpha = 0, 
            zorder = 2
        )
        
        weighted_overlap_masked = np.ma.masked_where(~chi2_mask.astype(bool), weighted_overlap)
        
        ax[0, 1].imshow(
            weighted_overlap_masked,
            interpolation="none",
            extent=(0.5, N_feed + 0.5, N_feed + 0.5, 0.5),
            cmap="CMRmap",
            vmin = 0,
            vmax = 1,
            rasterized=True,
            zorder = 3,
        )


        new_tick_locations = np.array(range(N_feed)) + 1
        ax[0, 1].set_xticks(new_tick_locations)
        ax[0, 1].set_yticks(new_tick_locations)
        
        ax[0, 1].set_xlabel(f"Feed of {split1}")
        ax[0, 1].set_ylabel(f"Feed of {split2}")
        
        cbar = plt.colorbar(img2, ax = ax[0, 1])
        
        cbar.set_label(r"Noise Weighed Overlap")
        
        ####################################################
        
        ax[1, 0].axis("off")
        ax[1, 1].axis("off")

        ax1 = fig.add_subplot(212)
        if np.all(~np.isfinite(weighted_overlap.flatten())).dtype == bool and np.all(~np.isfinite(overlap_vs_rnd.flatten())).dtype == bool:
        
            ax1.hist(
                weighted_overlap.flatten(), 
                histtype = "step",
                bins = 15,
                lw = 3,
                color = "b",
                label = "Noise Weigted Overlap",
            )
            
            ax1.hist(
                overlap_vs_rnd.flatten(), 
                histtype = "step",
                bins = 15,
                lw = 3,
                color = "r",
                label = "Relative Noise Weigted Overlap",
            )
            # ax1.axvline(self.params.psx_relative_overlap_limit, linestyle = "dashed", color = "r", label = f"Cut limit :{self.params.psx_relative_overlap_limit:.2f}")
            ax1.axvline(self.params.psx_overlap_limit, linestyle = "dashed", color = "b", label = f"Cut limit :{self.params.psx_overlap_limit:.2f}")
        
        ax1.legend()
        ax1.set_ylabel("Number Count")
        ax1.set_xlabel("Overlap Statistic")
        ax1.set_xlim(0, 1.1)
        
        fig.savefig(outname, bbox_inches="tight")

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
        matplotlib.rcParams["hatch.color"] = "green"
        matplotlib.rcParams["hatch.linewidth"] = 0.5
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
            self.params.psx_subdir,
            f"xs_chi2_grid_{split1}_X_{split2}.png"
            )
                
        if not os.path.exists(os.path.dirname(outname)):
            os.makedirs(os.path.dirname(outname), exist_ok = True)
            
        # Number of detectors used in cross-correlation combinations
        N_feed = len(self.included_feeds)

        # Define symetric collormap
        cmap = matplotlib.cm.RdBu.reversed()

        # Bad values, i.e. NaN and Inf, are set to black
        cmap.set_bad("k", 1)
        chi2[chi2 == 0] = np.nan
        chi2[~np.isfinite(chi2)] = np.nan

        lim = np.nanmin((100, np.nanmax(np.abs(chi2))))
        norm = matplotlib.colors.Normalize(vmin=-1.2 * lim, vmax=1.2 * lim)
        norm_chi2_cut = matplotlib.colors.Normalize(vmin=-self.params.psx_chi2_cut_limit, vmax=self.params.psx_chi2_cut_limit)

        # Plot chi2 value grid
        fig, ax = plt.subplots(2, 2, figsize = (13, 10))

        img = ax[0, 0].imshow(
            chi2,
            interpolation = "none",
            norm=norm,
            extent = (0.5, N_feed + 0.5, N_feed + 0.5, 0.5),
            cmap = cmap,
            rasterized = True,
            zorder = 1,
        )

        ax[0, 0].fill_between(
            [0.5, N_feed + 0.5], 
            0.5, 
            N_feed + 0.5, 
            hatch='xx', 
            alpha = 0, 
            zorder = 2
        )
        
        for i in range(chi2_mask.shape[0]):
            chi2_mask[i, i] = False
        
        # chi2_masked = np.ma.masked_where(~(np.abs(chi2) < self.params.psx_chi2_cut_limit), chi2)

        chi2_masked = np.ma.masked_where(~chi2_mask.astype(bool), chi2)

        ax[0, 0].imshow(
            chi2_masked,
            interpolation="none",
            extent=(0.5, N_feed + 0.5, N_feed + 0.5, 0.5),
            cmap="RdBu_r",
            norm=norm,
            rasterized=True,
            zorder = 3,
        )


        new_tick_locations = np.array(range(N_feed)) + 1
        ax[0, 0].set_xticks(new_tick_locations)
        ax[0, 0].set_yticks(new_tick_locations)
        
        ax[0, 0].set_xlabel(f"Feed of {split1}")
        ax[0, 0].set_ylabel(f"Feed of {split2}")
        
        cbar = plt.colorbar(img, ax = ax[0, 0])
        
        cbar.set_label(r"$|\chi^2| \times$ sign($\chi^3$)")

        ####################################################

        ax[0, 1].set_title(r"Colorbar zoomed in on $\chi^2$ cut limit")
        img2 = ax[0, 1].imshow(
            chi2,
            interpolation = "none",
            norm=norm_chi2_cut,
            extent = (0.5, N_feed + 0.5, N_feed + 0.5, 0.5),
            cmap = cmap,
            rasterized = True,
            zorder = 1,
        )

        ax[0, 1].fill_between(
            [0.5, N_feed + 0.5], 
            0.5, 
            N_feed + 0.5, 
            hatch='xx', 
            alpha = 0, 
            zorder = 2
        )

        # chi2_masked = np.ma.masked_where(~(np.abs(chi2) < self.params.psx_chi2_cut_limit), chi2)

        chi2_masked = np.ma.masked_where(~chi2_mask.astype(bool), chi2)
        
        ax[0, 1].imshow(
            chi2_masked,
            interpolation="none",
            extent=(0.5, N_feed + 0.5, N_feed + 0.5, 0.5),
            cmap="RdBu_r",
            norm=norm_chi2_cut,
            rasterized=True,
            zorder = 3,
        )


        new_tick_locations = np.array(range(N_feed)) + 1
        ax[0, 1].set_xticks(new_tick_locations)
        ax[0, 1].set_yticks(new_tick_locations)
        
        ax[0, 1].set_xlabel(f"Feed of {split1}")
        ax[0, 1].set_ylabel(f"Feed of {split2}")
        
        cbar = plt.colorbar(img2, ax = ax[0, 1])
        
        cbar.set_label(r"$|\chi^2| \times$ sign($\chi^3$)")
        
        
        ####################################################
        
        
        ax[1, 0].axis("off")
        ax[1, 1].axis("off")
        ax1 = fig.add_subplot(212)

        if np.all(~np.isfinite(chi2_masked.flatten())).dtype == bool:
            ax1.hist(
                chi2_masked.flatten(), 
                histtype = "step",
                bins = 25,
                lw = 3,
                color = "b",
            )
        
        ax1.set_ylabel("Number Count")
        ax1.set_xlabel(r"Normalized $\chi^2$")
        
        
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
        self.map_name = self.params.map_name + f"_rnd{self.params.jk_rnd_split_seed}" + self.params.psx_map_name_postfix
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
        
        self.angle2Mpc = self.cosmology.kpc_comoving_per_arcmin(
            self.params.phy_center_redshift
        ).to(u.Mpc / u.arcmin)

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
                    
                    for j in range(2):
                        split_map_combinations.append(
                            (f"multisplits/{primary_variable}/map_{primary_variable}{j}{name}",
                            f"multisplits/{primary_variable}/map_{primary_variable}{j}{name}",)
                        )
                    
                    
                    
        else:
            number_of_secondary_variables = len(secondary_variables)  
            
            # Generate indices for all split combinations

            combinations = list(itertools.combinations(range(self.params.split_base_number), r = 2))  
            secondary_combinations = list(itertools.combinations(range(self.params.split_base_number), r = 2 * len(cross_and_secondary)))  
            
            secondary_combinations += [(0, 0)] + [(1, 1)]
            
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
    
    def pix_window(self, k, dx):
        """Method for returning pixel window transfer function for target grid.

        Args:
            k (npt.NDArray): Input array of wave number k-bin centers in cosmological units
            dx (float): Grid resolution in cosmological units

        Returns:
            npt.NDArray: Gaussian beam transfer function in perpendicular space.
        """
        # NOTE: 1 / np.pi to get rid of pi in np.sinc = np.sin(pi*x) / (pi*x)
        return np.sinc(0.5 * dx * k / np.pi) ** 2 

    def gaussian_beam_transfer_function(self, k, FWHM):
        """Method for returning beam transfer function for Gaussian beam.

        Args:
            k (npt.NDArray): Input array of wave number k-bin centers in cosmological units
            FWHM (npt.NDArray): Full-width at half-maximum of the Gaussian beam in cosmological units

        Returns:
            npt.NDArray: Gaussian beam transfer function in perpendicular space.
        """
        return np.exp(-(FWHM * k) ** 2 / (8 * np.log(2)))

    def beam_transfer_function(self):
        """Method for returning beam transfer function for realistic beam model.

        Args:
            k (npt.NDArray): Input array of wave number k-bin centers in cosmological units
        Returns:
            npt.NDArray: COMAP beam transfer function in perpendicular space.
        """
        transfer_function_dir = os.path.join(current, "transfer_functions")

        # Loading real space beam model from txt file
        beam_r = np.loadtxt(
            os.path.join(transfer_function_dir, "beam_r.txt")
        )   # degrees
        beam = np.loadtxt(
            os.path.join(transfer_function_dir, "beam.txt")
        ) # beam as function of radius in degrees

        # Interpolating beam model as function of radius in degrees
        beam_interp = interpolate.CubicSpline(beam_r, beam)

        # Defining 2D grid to use to compute beam transfer function
        dx = 1e-3 * u.deg
        
        dx_cosmo = (dx * self.angle2Mpc).decompose().to(u.Mpc)
        x_kernel = np.arange(-1, 1, dx.value)
        y_kernel = np.arange(-1, 1, dx.value)
        X, Y = np.meshgrid(x_kernel, y_kernel)
        R = np.sqrt(X ** 2 + Y ** 2)

        beam_2D = beam_interp(R)

        # Nulling beam at 1 degree scale 
        beam_2D[R > 1] *= 0 
        
        # Normalising beam to integrate to 1
        beam_2D /= np.sum(beam_2D)
        
        # Real COMAP beam is known to have 72% of total power within 6.4 arcmin
        normalisation = 0.72 / np.sum(beam_2D[R * 60 <= 6.4])
        beam_2D *= normalisation

        # Find beam transfer function by fourier transform
        tf_beam_fourier = np.abs(np.fft.fftn(beam_2D)) ** 2 
        kx = 2 * np.pi * np.fft.fftfreq(beam_2D.shape[0], dx_cosmo)
        ky = kx.copy()

        # Sorting mirrored values to correct symmetric order
        tf_beam_fourier = tf_beam_fourier[np.argsort(ky), :]
        tf_beam_fourier = tf_beam_fourier[:, np.argsort(kx)]

        tf_beam_fourier = tf_beam_fourier[np.argsort(ky), :]
        tf_beam_fourier = tf_beam_fourier[:, np.argsort(kx)]
        
        kx = kx[np.argsort(kx)]
        ky = ky[np.argsort(ky)]
        
        NX = kx.size
        NY = ky.size
        
        kx = kx[NX // 2:]
        ky = ky[NY // 2:]
        
        tf_beam_fourier = tf_beam_fourier[:NX // 2, :NY // 2]
        
        # # Returning a interpolation of beam transfer unction
        # tf_beam_interp = interpolate.RectBivariateSpline(
        #     kx,
        #     ky,
        #     tf_beam_fourier, 
        #     s = 0, # No smoothing when splining
        #     kx = 3, # Use bi-cubic spline in x-direction
        #     ky = 3, # Use bi-cubic spline in x-direction
        #     )

        tf_beam_interp = interpolate.CubicSpline(
            kx, 
            tf_beam_fourier[0, :]
        ) 

        return tf_beam_interp
    
    def load_models(self):
        """Method that loads theoretical model power spectral, and defines 
        interpolation thereof.
        """
        from models import Model
        
        self.models = {}
        for model_name in os.listdir(self.params.psx_model_path):
            model = Model(os.path.join(self.params.psx_model_path, model_name))
            model.read_model()
            model.interpolate_model()
            setattr(model, "name", os.path.splitext(model_name)[0].split("_")[-1])        
            self.models[model.name] = model
    

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
    comap2fpxs.params.psx_mode = "saddlebag"
    comap2fpxs.run()

    # comap2fpxs.params.psx_mode = "feed"
    # comap2fpxs.run()


    
    # if run_wn_sim:
    #     comap2fpxs.params.psx_generate_white_noise_sim = True


    #     # basepath = "/mn/stornext/d22/cmbco/comap/protodir/power_spectrum/test/average_spectra/co2_python_poly_debug_null_w_bug/"
    #     # import glob
    #     # filelist = glob.glob(f"*white_noise_seed*/**/*.h5", root_dir = basepath, recursive = True)
    #     # seedlist = [int(file.split("seed")[-1].split("/")[0]) for file in filelist]
    #     seed_list = []
    #     seed_list_path = os.path.join(comap2fpxs.params.power_spectrum_dir, comap2fpxs.params.psx_seed_list)
    #     if comap2fpxs.params.psx_use_seed_list:
    #         seeds_to_run = np.loadtxt(seed_list_path)
    #     else:
    #         seeds_to_run = range(comap2fpxs.params.psx_monte_carlo_sim_number)

    #     global_verbose = comap2fpxs.verbose

    #     if comap2fpxs.params.psx_use_seed_list:
    #         print("#" * 80)
    #         print(f"Running with provided seed list: {seed_list_path}")
    #         print(f"Seed list contains {len(seeds_to_run)} Monte Carlo simulations")
    #         print("#" * 80)
    #     elif global_verbose and comap2fpxs.rank == 0:
    #         print("#" * 80)
    #         print(f"Running {comap2fpxs.params.psx_monte_carlo_sim_number} white noise Monte Carlo simulations:")
    #         print("#" * 80)


    #     for i, seed in enumerate(seeds_to_run):
    #         # try:
    #         # Turning this to None will make new seed from time.time() each iteration
            
    #         if comap2fpxs.params.psx_use_seed_list:
    #             comap2fpxs.params.psx_white_noise_sim_seed = seed
    #         else:
    #             comap2fpxs.generate_new_monte_carlo_seed()
                
    #             seed_list.append(comap2fpxs.params.psx_white_noise_sim_seed)

            
    #         comap2fpxs.verbose = False 

    #         if global_verbose and comap2fpxs.rank == 0:
    #             print("-"*40)
    #             print(f"\033[91m Simulation # {i + 1} / {comap2fpxs.params.psx_monte_carlo_sim_number}: \033[00m \033[93m Seed = {comap2fpxs.params.psx_white_noise_sim_seed} \033[00m")

    #         t0 = time.perf_counter()
    #         comap2fpxs.run()
        
    #         if global_verbose and comap2fpxs.rank == 0:
    #             print(f"\033[92m Run time: {time.perf_counter() - t0} sec \033[00m")
    #             print("-"*40)
    #         # except:
    #         #     print("SKIP")
    #         #     continue
        
    #     comap2fpxs.comm.Barrier()
    #     if not comap2fpxs.params.psx_use_seed_list and comap2fpxs.rank == 0:
    #         seed_list = np.array(seed_list)

    #         if comap2fpxs.verbose:
    #             print(f"Saving Monte Carlo seed list to: {seed_list_path}")

    #         np.savetxt(seed_list_path, seed_list.astype(int))
