from __future__ import annotations
from typing import Optional
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
import time 
import scipy.interpolate as interpolate

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

    def run(self):
        """Main run method to perform computation of cross-spectra
        """

        # Distributed start will slow down some MPI threads in beginning to not overwhelm memory.
        if self.params.distributed_starting:
            time.sleep((self.rank%16) * 30.0)
        time.sleep(self.rank * 0.01)

        # Only want to compute FPXS for included feeds (detectors)
        self.included_feeds = self.params.included_feeds

        # Define all combinations between detectors
        feed_combinations = list(itertools.product(self.included_feeds, self.included_feeds))
        self.feed_combinations = feed_combinations

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
                mapnames = [f"{field_name}_{self.params.map_name}.h5" for field_name in fields]
            
            # Map file name combinations
            field_combinations = list(itertools.product(mapnames, mapnames))
            
        elif len(mapnames) == 0 and not self.params.psx_null_cross_field:
            # If now custom map file name list is provided, and no cross-field spectra are computed,
            # assume that map file name follows mapmaker file name pattern for all fields.

            fields = self.params.fields
            field_combinations = [(f"{field_name}_{self.params.map_name}.h5", f"{field_name}_{self.params.map_name}.h5")
                        for field_name in fields]            
        else:
            # Else use file names from map file name list (when to compute FPXS of custom map list)
            field_combinations = [(name, name) for name in mapnames]
        
        self.field_combinations = field_combinations        

        # Generate tuple of (field/map filename, split map key, feed combination) to use for computing FPXS 
        all_combinations = list(itertools.product(field_combinations, self.split_map_combinations, feed_combinations))
        Number_of_combinations = len(all_combinations)
        

        if self.verbose and self.rank == 0:
            print("#" * 70)
            print(f"Primary splits: {self.primary_variables}")
            print(f"Secondary splits: {self.secondary_variables}")
            print(f"Computing cross-spectra for {Number_of_combinations} combinations with {self.Nranks} MPI processes:")
            print("#" * 70)

        self.comm.Barrier()

        # MPI parallel run over all FPXS combinations
        for i in range(Number_of_combinations):
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
                outdir = f"{mapname1[:-3]}_X_{mapname2[:-3]}"

                if self.verbose:
                    # Print some usefull information while computing FPXS
                    if self.params.psx_null_diffmap:
                        print(f"\033[91m Rank {self.rank} ({i + 1} / {Number_of_combinations}): \033[00m \033[94m {mapname1.split('_')[0]} X {mapname2.split('_')[0]} \033[00m \033[00m \033[92m ({split1[0].split('/map_')[-1]} - {split1[1].split('/map_')[-1]}) X ({split2[0].split('/map_')[-1]} - {split2[1].split('/map_')[-1]}) \033[00m \033[93m Feed {feed1} X Feed {feed2} \033[00m")
                    else:
                        print(f"\033[91m Rank {self.rank} ({i + 1} / {Number_of_combinations}): \033[00m \033[94m {mapname1.split('_')[0]} X {mapname2.split('_')[0]} \033[00m \033[00m \033[92m {split1.split('/map_')[-1]} X {split2.split('/map_')[-1]} \033[00m \033[93m Feed {feed1} X Feed {feed2} \033[00m")  
                
                # Generate cross-spectrum instance from split keys and feeds of current FPXS combo 
                cross_spectrum = xs_class.CrossSpectrum_nmaps(
                        self.params, 
                        splits, 
                        feed1, 
                        feed2,
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

                    # Compute cross-spectrum for current FPXS combination
                    cross_spectrum.calculate_xs_2d(
                        no_of_k_bins=self.params.psx_number_of_k_bins + 1,
                    )

                    # Run noise simulations to generate FPXS errorbar
                    cross_spectrum.run_noise_sims_2d(
                        self.params.psx_noise_sim_number,
                        no_of_k_bins=self.params.psx_number_of_k_bins + 1,
                    )

                    # Save resulting FPXS from current combination to file
                    cross_spectrum.make_h5_2d(outdir)

        # MPI barrier to prevent thread 0 from computing average FPXS before all individual combinations are finished.
        self.comm.Barrier()
        
        if self.rank == 0:
            if self.verbose:
                print("Computing averages:")
                

            self.compute_averages()
        

    def compute_averages(self):
        average_spectrum_dir = os.path.join(self.power_spectrum_dir, "average_spectra")

        fig_dir = os.path.join(self.power_spectrum_dir, "figs")

        if not os.path.exists(average_spectrum_dir):
            os.mkdir(average_spectrum_dir)
    
        if not os.path.exists(fig_dir):
            os.mkdir(fig_dir)

        if not os.path.exists(os.path.join(fig_dir, "chi2_grid")):
            os.mkdir(os.path.join(fig_dir, "chi2_grid"))

        if not os.path.exists(os.path.join(fig_dir, "average_spectra")):
            os.mkdir(os.path.join(fig_dir, "average_spectra"))
    
        N_feed = len(self.included_feeds)
        N_splits = len(self.split_map_combinations)
        N_k = self.params.psx_number_of_k_bins

        for map1, map2 in self.field_combinations:
            mapname1 = map1.split("/")[-1]
            mapname2 = map2.split("/")[-1]
            indir = f"{mapname1[:-3]}_X_{mapname2[:-3]}"

            outdir = os.path.join(average_spectrum_dir, indir)
            
            if not os.path.exists(outdir):
                os.mkdir(outdir)

            xs_mean = np.zeros((N_splits, N_k, N_k))
            xs_error = np.zeros((N_splits, N_k, N_k))

            xs_mean_1d = np.zeros((N_splits, N_k))
            xs_error_1d = np.zeros((N_splits, N_k))

            for i, splits in enumerate(self.split_map_combinations):
                

                xs_sum = np.zeros((N_k, N_k))

                xs_inv_var = np.zeros((N_k, N_k))
                
                chi2 = np.zeros((N_feed, N_feed))

                for feed1 in range(N_feed):
                    for feed2 in range(N_feed):
                        cross_spectrum = xs_class.CrossSpectrum_nmaps(
                            self.params, 
                            splits, 
                            self.included_feeds[feed1], 
                            self.included_feeds[feed2],
                        )
                        cross_spectrum.read_spectrum(indir)

                        xs = cross_spectrum.xs_2D
                        xs_sigma = cross_spectrum.rms_xs_std_2D
                        
                        k_bin_centers_perp, k_bin_centers_par  = cross_spectrum.k
                        
                        k_bin_edges_par = cross_spectrum.k_bin_edges_par
                        
                        k_bin_edges_perp = cross_spectrum.k_bin_edges_perp

                        transfer_function = self.full_trasnfer_function_interp(k_bin_centers_perp, k_bin_centers_par)

                        transfer_function_mask = transfer_function > self.params.tf_cutoff

                        chi3 = np.nansum(
                        (xs[transfer_function_mask] / xs_sigma[transfer_function_mask]) ** 3
                        )

                        number_of_samples = np.sum(transfer_function_mask)
            
                        chi2[feed1, feed2] = np.sign(chi3) * abs(
                            (np.nansum((xs[transfer_function_mask]  / xs_sigma[transfer_function_mask] ) ** 2) - number_of_samples)
                            / np.sqrt(2 * number_of_samples)
                        )

                        if (np.isfinite(chi2[feed1, feed2]) and chi2[feed1, feed2] != 0) and feed1 != feed2:
                            if np.abs(chi2[feed1, feed2]) < self.params.psx_chi2_cut_limit:
                                xs_sum += xs / xs_sigma ** 2
                                xs_inv_var += 1 / xs_sigma ** 2
                

                if self.verbose:
                    print(f"{indir} {splits} \n# |chi^2| < {self.params.psx_chi2_cut_limit}:", np.sum(np.abs(chi2) < self.params.psx_chi2_cut_limit))
                    


                xs_mean[i, ...] = xs_sum / xs_inv_var
                xs_error[i, ...] = 1.0 / np.sqrt(xs_inv_var)

                weights = 1 / (xs_error[i, ...] / transfer_function) ** 2

                xs_1d = xs_mean[i, ...].copy()
                xs_1d /= transfer_function
                xs_1d *= weights

                k_bin_edges = np.logspace(-2.0, np.log10(1.5), len(k_bin_centers_perp) + 1)

                kgrid = np.sqrt(sum(ki**2 for ki in np.meshgrid(k_bin_centers_perp, k_bin_centers_par, indexing="ij")))

                Ck_nmodes_1d = np.histogram(
                    kgrid[kgrid > 0], bins=k_bin_edges, weights=xs_1d[kgrid > 0]
                )[0]
                inv_var_nmodes_1d = np.histogram(
                    kgrid[kgrid > 0], bins=k_bin_edges, weights=weights[kgrid > 0]
                )[0]
                nmodes_1d = np.histogram(kgrid[kgrid > 0], bins=k_bin_edges)[0]

                # Ck = Ck_nmodes / nmodes
                k_1d = (k_bin_edges[1:] + k_bin_edges[:-1]) / 2.0
                
                Ck_1d = np.zeros_like(k_1d)
                rms_1d = np.zeros_like(k_1d)
                
                Ck_1d[np.where(nmodes_1d > 0)] = (
                    Ck_nmodes_1d[np.where(nmodes_1d > 0)]
                    / inv_var_nmodes_1d[np.where(nmodes_1d > 0)]
                )
                rms_1d[np.where(nmodes_1d > 0)] = np.sqrt(
                    1 / inv_var_nmodes_1d[np.where(nmodes_1d > 0)]
                )

                xs_mean_1d[i, ...] = Ck_1d
                xs_error_1d[i, ...] = rms_1d


                chi2_name = os.path.join(fig_dir, "chi2_grid")
                chi2_name = os.path.join(chi2_name, indir)

                if not os.path.exists(chi2_name):
                    os.mkdir(chi2_name)
                
                if self.params.psx_null_diffmap:
                    chi2_name = os.path.join(chi2_name, "null_diffmap")
                    if not os.path.exists(chi2_name):
                        os.mkdir(chi2_name)
                    
                    chi2_name = os.path.join(chi2_name, f"{cross_spectrum.null_variable}")
                    if not os.path.exists(chi2_name):
                        os.mkdir(chi2_name)
                
                
                self.plot_chi2_grid(chi2, splits, chi2_name)

                average_name = os.path.join(fig_dir, "average_spectra")
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
                
                

                self.plot_2D_mean(
                    k_bin_edges_par,
                    k_bin_edges_perp,
                    xs_mean[i, ...],
                    xs_error[i, ...],
                    splits,
                    (mapname1, mapname2),
                    average_name,
                )

                self.plot_1D_mean(
                    k_1d,
                    xs_mean_1d[i, ...],
                    xs_error_1d[i, ...],
                    splits,
                    (mapname1, mapname2),
                    average_name,
                )

            with h5py.File(os.path.join(outdir, indir + "_average_fpxs.h5"), "w") as outfile:
                outfile.create_dataset("k_1d", data = k_1d)             
                outfile.create_dataset("k_centers_par", data = k_bin_centers_par)
                outfile.create_dataset("k_centers_perp", data = k_bin_centers_perp)
                outfile.create_dataset("k_edges_par", data = k_bin_edges_par)      
                outfile.create_dataset("k_edges_perp", data = k_bin_edges_perp)     
                outfile.create_dataset("xs_mean_1d", data = xs_mean_1d)       
                outfile.create_dataset("xs_mean_2d", data = xs_mean)
                outfile.create_dataset("xs_sigma_1d", data = xs_error_1d)      
                outfile.create_dataset("xs_sigma_2d", data = xs_error)

    def plot_2D_mean(self,
                    k_bin_edges_par,
                    k_bin_edges_perp,
                    xs_mean,
                    xs_sigma,
                    splits,
                    fields,
                    outname
                    ):

        if self.params.psx_null_diffmap:
            split1 = splits[0][0].split("map_")[-1][5:]
            split2 = splits[1][0].split("map_")[-1][5:]
        else:
            split1 = splits[0].split("map_")[-1]
            split2 = splits[1].split("map_")[-1]

        outname = os.path.join(
            outname, 
            f"xs_mean_2d_{split1}_X_{split2}.png"
            )
        
        fig, ax = plt.subplots(1, 2, figsize=(16, 5.6), sharey=True)

        fig.suptitle(f"Fields: {fields[0]} X {fields[1]} | {split1} X {split2}", fontsize=16)
        
        lim = np.nanmax(np.abs(xs_mean[3:-3, 3:-3]))
        lim_significance = np.nanmax(np.abs((xs_mean / xs_sigma)[3:-3, 3:-3]))

        norm = matplotlib.colors.Normalize(vmin=-1.1 * lim, vmax=1.1 * lim)
        lim_significance = matplotlib.colors.Normalize(vmin=-lim_significance, vmax=lim_significance)

        img1 = ax[0].imshow(
            xs_mean,
            interpolation="none",
            origin="lower",
            extent=[0, 1, 0, 1],
            cmap="PiYG_r",
            norm=norm,
            rasterized=True,
        )
        fig.colorbar(img1, ax=ax[0], fraction=0.046, pad=0.04).set_label(
            r"$\tilde{C}\left(k_{\bot},k_{\parallel}\right)$ [$\mu$K${}^2$ (Mpc)${}^3$]",
            size=16,
        )

        img2 = ax[1].imshow(
            xs_mean / (xs_sigma),
            interpolation="none",
            origin="lower",
            extent=[0, 1, 0, 1],
            cmap="PiYG_r",
            norm=lim_significance,
            rasterized=True,
        )
        fig.colorbar(img2, ax=ax[1], fraction=0.046, pad=0.04).set_label(
            r"$\tilde{C}/\sigma\left(k_{\bot},k_{\parallel}\right)$",
            size=16,
        )

        ticks = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]

        majorticks = [0.03, 0.1, 0.3, 1]
        majorlabels = ["0.03", "0.1", "0.3", "1"]

        xbins = k_bin_edges_par

        ticklist_x = self.log2lin(ticks[:-3], xbins)
        majorlist_x = self.log2lin(majorticks, xbins)

        ybins = k_bin_edges_perp

        ticklist_y = self.log2lin(ticks, ybins)
        majorlist_y = self.log2lin(majorticks, ybins)

        ax[0].set_title(r"$\tilde{C}^{\mathrm{FPXS}}$ ", fontsize=16)
        ax[1].set_title(r"$\tilde{C}^{\mathrm{FPXS}}/\sigma$ ", fontsize=16)

        for i in range(2):
            ax[i].set_xticks(ticklist_x, minor=True)
            ax[i].set_xticks(majorlist_x, minor=False)
            ax[i].set_xticklabels(majorlabels, minor=False, fontsize=16)
            ax[i].set_yticks(ticklist_y, minor=True)
            ax[i].set_yticks(majorlist_y, minor=False)
            ax[i].set_yticklabels(majorlabels, minor=False, fontsize=16)

        ax[0].set_xlabel(r"$k_{\parallel}$ [Mpc${}^{-1}$]", fontsize=16)
        ax[0].set_ylabel(r"$k_{\bot}$ [Mpc${}^{-1}$]", fontsize=16)
        ax[1].set_xlabel(r"$k_{\parallel}$ [Mpc${}^{-1}$]", fontsize=16)

        fig.savefig(outname, bbox_inches = "tight")
        
    def plot_1D_mean(self,
                    k_1d,
                    xs_mean,
                    xs_sigma,
                    splits,
                    fields,
                    outname
                    ):

        matplotlib.rcParams["xtick.labelsize"] = 16
        matplotlib.rcParams["ytick.labelsize"] = 16

        if self.params.psx_null_diffmap:
            split1 = splits[0][0].split("map_")[-1][5:]
            split2 = splits[1][0].split("map_")[-1][5:]
        else:
            split1 = splits[0].split("map_")[-1]
            split2 = splits[1].split("map_")[-1]

        outname = os.path.join(
            outname, 
            f"xs_mean_1d_{split1}_X_{split2}.png"
            )
        
        fig, ax = plt.subplots(2, 1, figsize=(16, 12), sharex=True)

        fig.suptitle(f"Fields: {fields[0]} X {fields[1]} | {split1} X {split2}", fontsize=16)
        
        where = np.logical_and(k_1d > 0.04, k_1d < 1.0)

        lim = np.nanmax(np.abs((k_1d * (xs_mean + xs_sigma))[where]))
        lim = np.nanmax((np.nanmax(np.abs((k_1d * (xs_mean - xs_sigma))[where])), lim))

        lim_significance = 2 * np.nanmax(np.abs((xs_mean / xs_sigma)[where]))
        
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

        if np.isfinite(lim):
            ax[0].set_ylim(-lim, lim)
        ax[0].set_ylabel(
            r"$k\tilde{C}(k)$ [$\mu$K${}^2$ (Mpc)${}^3$]",
            fontsize = 16,
        )

        ax[1].scatter(
            k_1d,
            xs_mean / xs_sigma,
            s = 80,
        )
        ax[1].errorbar(
            k_1d,
            xs_mean / xs_sigma,
            1,
            lw = 3,
            fmt = " ",
        )
        
        ax[0].axhline(0, color = "k", alpha = 0.5)
        ax[1].axhline(0, color = "k", alpha = 0.5)

        if np.isfinite(lim_significance):
            ax[1].set_ylim(-lim_significance, lim_significance)
    
        ax[1].set_ylabel(
            r"$\tilde{C} / \sigma_\tilde{C}(k)$",
            fontsize = 16,
        )

        ax[0].set_xscale("log")
        ax[1].set_xscale("log")

        klabels = [0.05, 0.1, 0.2, 0.5, 1.0]

        ax[0].set_xticks(klabels)
        ax[0].set_xticklabels(klabels, fontsize = 16)
        ax[1].set_xticks(klabels)
        ax[1].set_xticklabels(klabels, fontsize = 16)

        ax[0].set_xlim(0.04, 1.0)
        ax[1].set_xlim(0.04, 1.0)

        ax[1].set_xlabel(r"$k [\mathrm{Mpc}^{-1}]$", fontsize = 16)

        ax[0].grid(True)
        ax[1].grid(True)

        fig.savefig(outname, bbox_inches = "tight")

    def plot_chi2_grid(self, chi2, splits, outname):
        matplotlib.rcParams["xtick.labelsize"] = 10
        matplotlib.rcParams["ytick.labelsize"] = 10

        if self.params.psx_null_diffmap:
            split1 = splits[0][0].split("map_")[-1][5:]
            split2 = splits[1][0].split("map_")[-1][5:]
        else:
            split1 = splits[0].split("map_")[-1]
            split2 = splits[1].split("map_")[-1]



        outname = os.path.join(
            outname, 
            f"xs_chi2_grid_{split1}_X_{split2}.png"
            )
        
        N_feed = len(self.included_feeds)

        cmap = matplotlib.cm.PiYG.reversed()
    
        cmap.set_bad("k", 1)

        # chi2[np.abs(chi2) >= vmax] = np.nan
        #np.fill_diagonal(chi2, np.nan)
        #chi2[np.abs(chi2) > self.params.psx_chi2_cut_limit] = np.nan
        
        fig, ax = plt.subplots()

        img = ax.imshow(
            chi2,
            interpolation="none",
            vmin=-self.params.psx_chi2_cut_limit,
            vmax=self.params.psx_chi2_cut_limit,
            extent=(0.5, N_feed + 0.5, N_feed + 0.5, 0.5),
            cmap=cmap,
            rasterized=True,
        )
        new_tick_locations = np.array(range(N_feed)) + 1
        ax.set_xticks(new_tick_locations)
        ax.set_yticks(new_tick_locations)
        ax.set_xlabel(f"Feed of {split1}")
        ax.set_ylabel(f"Feed of {split2}")
        cbar = plt.colorbar(img, ax = ax)
        cbar.set_label(r"$|\chi^2| \times$ sign($\chi^3$)")
        
        fig.savefig(outname, bbox_inches="tight")

    def read_params(self):
        """Method reading and parsing the parameters from file or command line.

        Raises:
            ValueError: If no power spectrum directory is provided
            ValueError: if no COMAP map name is provided
            ValueError: _description_
            ValueError: _description_
            ValueError: _description_
        """
        from l2gen_argparser import parser

        self.params = parser.parse_args()

        self.power_spectrum_dir = self.params.power_spectrum_dir
        self.map_name = self.params.map_name
        self.map_dir = self.params.map_dir
        self.jk_def_file = self.params.jk_def_file
        self.accept_data_id_string = self.params.accept_data_id_string

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
            
        if self.accept_data_id_string is None:
            raise ValueError(
                "Please specify a accept_data_id_string in parameter file or terminal."
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
            split_line = line.split()
            variable = split_line[0]
            number = split_line[1]

            if len(split_line) > 2:
                extra = split_line[2]
            
            if len(split_line) < 2 or len(split_line) == 2:
                extra = "#"

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

    def log2lin(self, x, k_edges):
        loglen = np.log10(k_edges[-1]) - np.log10(k_edges[0])
        logx = np.log10(x) - np.log10(k_edges[0])
        return logx / loglen
    
    def define_transfer_function(self):
        """Method to define full transfer function to be used. Reads in all specified
        transfer functions, multiplies them and defines an interpolation function.
        """

        transfer_function_dir = self.params.transfer_function_dir = os.path.join(current, "transfer_functions")
        
        # By if now other transfer function is specified use an empty one, i.e. filled with ones
        empty_transfer_function_path = os.path.join(transfer_function_dir, "tf_all_ones.h5")

        if not os.path.exists(empty_transfer_function_path):
            raise FileExistsError("Cannot find empty (filled with ones) transfer function file.")
        
        full_transfer_function = TransferFunction()
        full_transfer_function.read(empty_transfer_function_path)

        if len(self.params.psx_transfer_function_names) == 0:
            print("WARNING: You are running without any transfer function!")

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
        self.full_trasnfer_function_interp = interpolate.RectBivariateSpline(
            full_transfer_function.k_bin_centers_perp_2D,
            full_transfer_function.k_bin_centers_par_2D,
            full_transfer_function.transfer_function_2D, 
            s = 0, # No smoothing when splining
            kx = 3, # Use bi-cubic spline in x-direction
            ky = 3, # Use bi-cubic spline in x-direction
            )
        
        if self.params.debug:
            # Unit test to check whether interpolation reproduces input when evaluated at original grid
            approx = self.full_trasnfer_function_interp(
                full_transfer_function.k_bin_centers_perp_2D,
                full_transfer_function.k_bin_centers_par_2D,
            )
            np.testing.assert_allclose(approx, full_transfer_function.transfer_function_2D)
        
if __name__ == "__main__":
    
    comap2fpxs = COMAP2FPXS()
    
    # print(comap2fpxs.primary_variables)
    # print(comap2fpxs.secondary_variables)
    # print(comap2fpxs.cross_variables)
    # print(comap2fpxs.all_variables)
    # print(comap2fpxs.cross_and_primary)
    # print(comap2fpxs.cross_and_secondary)
    # print(comap2fpxs.split_map_combinations)
    
    comap2fpxs.run()
