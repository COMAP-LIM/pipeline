# to create feed-feed pseudo-cross-spectra between all feed combinations and all split combinations

import numpy as np
import h5py
import tools
import map_cosmo
import itertools as itr
import matplotlib.pyplot as plt

plt.ioff()  # turn of the interactive plotting
import scipy.interpolate
import sys
import os

import warnings

warnings.filterwarnings(
    "ignore", category=RuntimeWarning
)  # ignore warnings caused by weights cut-off


class CrossSpectrum_nmaps:
    def __init__(self, params, split_keys, feed1=None, feed2=None):
        
        self.params = params
        
        split_base_number = self.params.split_base_number

        self.feed1 = feed1
        self.feed2 = feed2
        self.split_keys = split_keys


        self.names = []
        self.maps = []

        n_list = list(range(split_base_number))
        
        all_different_possibilities = list(itr.combinations(n_list, 2))  

        self.how_many_combinations = len(all_different_possibilities)

        if params.psx_null_diffmap:
            combination1 = split_keys[0]
            self.null_variable = combination1[0].split("/map_")[-1][:4]
            combination1 = combination1[0].split("/map_")[-1][5:]

            combination2 = split_keys[1]
            combination2 = combination2[0].split("/map_")[-1][5:]

        else:
            combination1 = split_keys[0]
            try:
                combination1 = combination1.split("/map_")[-1]
            except:
                pass
            combination2 = split_keys[1]
            try:
                combination2 = combination2.split("/map_")[-1]
            except:
                pass
        
        name1 = f"{combination1}_{self.params.psx_mode}{feed1}"
        name2 = f"{combination2}_{self.params.psx_mode}{feed2}"

        self.names.append(name1)
        self.names.append(name2)

        self.outname = (
                "xs_2D_"
                + self.names[0]
                + "_and_"
                + self.names[1]
                + ".h5"
        )


    def read_map(self, mappaths, cosmology):
        self.name_of_map = mappaths  
        if len(self.name_of_map) < 2:
            raise ValueError("Can only compute cross-spectra when two map paths are provided.")
        
        split_map1 = map_cosmo.MapCosmo(
            self.params,
            mappaths[0], 
            cosmology,
            self.feed1 - 1, # feed index to use for loading map is base-0
            self.split_keys[0],
            )

        split_map2 = map_cosmo.MapCosmo(
            self.params,
            mappaths[1], 
            cosmology,
            self.feed2 - 1, # feed index to use for loading map is base-0
            self.split_keys[1],
            )

        self.maps.append(split_map1)
        self.maps.append(split_map2)

    # NORMALIZE WEIGHTS FOR A GIVEN PAIR OF MAPS
    def normalize_weights(self, i, j):
        norm = np.sqrt(np.mean((self.maps[i].w * self.maps[j].w).flatten()))
        self.maps[i].w = self.maps[i].w / norm
        self.maps[j].w = self.maps[j].w / norm

    # REVERSE NORMALIZE_WEIGHTS, TO NORMALIZE EACH PAIR OF MAPS INDEPENDENTLY
    def reverse_normalization(self, i, j):
        norm = np.sqrt(np.mean((self.maps[i].w * self.maps[j].w).flatten()))
        self.maps[i].w = self.maps[i].w * norm
        self.maps[j].w = self.maps[j].w * norm

    # INFORM WHICH XS INDEX CORRESPONDS TO WHICH MAP-PAIR
    def get_information(self):
        indexes_xs = []
        index = -1
        for i in range(0, len(self.maps) - 1, 2):
            j = i + 1
            index += 1
            indexes_xs.append([index, self.names[i], self.names[j]])
        return indexes_xs

    # COMPUTE ALL THE XS
    def calculate_xs(
        self, no_of_k_bins=15
    ):  # here take the number of k-bins as an argument
        n_k = no_of_k_bins
        self.k_bin_edges = np.logspace(-2.0, np.log10(1.5), n_k)
        calculated_xs = self.get_information()

        # store each cross-spectrum and corresponding k and nmodes by appending to these lists:
        self.xs = []
        self.k = []
        self.nmodes = []
        index = -1
        for i in range(0, len(self.maps) - 1, 2):
            j = i + 1
            index += 1
            print(
                "Computing xs between "
                + calculated_xs[index][1]
                + " and "
                + calculated_xs[index][2]
            )
            self.normalize_weights(i, j)  # normalize weights for given xs pair of maps

            wi = self.maps[i].w
            wj = self.maps[j].w
            wh_i = np.where(np.log10(wi) < -0.5)
            wh_j = np.where(np.log10(wj) < -0.5)
            wi[wh_i] = 0.0
            wj[wh_j] = 0.0
            full_weight = np.sqrt(wi * wj) / np.sqrt(np.mean((wi * wj).flatten()))

            my_xs, my_k, my_nmodes = tools.compute_cross_spec3d(
                (self.maps[i].map * full_weight, self.maps[j].map * full_weight),
                self.k_bin_edges,
                dx=self.maps[i].dx,
                dy=self.maps[i].dy,
                dz=self.maps[i].dz,
            )
            self.reverse_normalization(
                i, j
            )  # go back to the previous state to normalize again with a different map-pair

            self.xs.append(my_xs)
            self.k.append(my_k)
            self.nmodes.append(my_nmodes)

        self.xs = np.array(self.xs)
        self.k = np.array(self.k)
        self.nmodes = np.array(self.nmodes)
        return self.xs, self.k, self.nmodes

    # COMPUTE ALL THE XS
    def calculate_xs_with_tf(
        self, no_of_k_bins=15
    ):  # here take the number of k-bins as an argument

        self.get_tf()

        n_k = no_of_k_bins
        self.k_bin_edges = np.logspace(-2.0, np.log10(1.5), n_k)
        self.k_bin_edges_par = np.logspace(-2.0, np.log10(1.0), n_k)
        self.k_bin_edges_perp = np.logspace(-2.0 + np.log10(2), np.log10(1.5), n_k)
        calculated_xs = self.get_information()

        # store each cross-spectrum and corresponding k and nmodes by appending to these lists:
        self.xs = []
        self.k = []
        self.nmodes = []
        self.rms_mean = []
        self.rms_std = []
        index = -1
        for i in range(0, len(self.maps) - 1, 2):
            j = i + 1
            index += 1
            print(
                "Computing 2D xs between "
                + calculated_xs[index][1]
                + " and "
                + calculated_xs[index][2]
            )
            self.normalize_weights(i, j)  # normalize weights for given xs pair of maps

            wi = self.maps[i].w
            wj = self.maps[j].w
            wh_i = np.where(np.log10(wi) < -0.5)
            wh_j = np.where(np.log10(wj) < -0.5)
            wi[wh_i] = 0.0
            wj[wh_j] = 0.0

            (
                my_xs,
                my_k,
                my_nmodes,
                my_rms_mean,
                my_rms_std,
            ) = tools.compute_cross_spec3d_with_tf(
                (
                    self.maps[i].map * np.sqrt(wi * wj),
                    self.maps[j].map * np.sqrt(wi * wj),
                ),
                (self.k_bin_edges_perp, self.k_bin_edges_par),
                self.k_bin_edges,
                self.rms_xs_mean_2D[i],
                self.rms_xs_std_2D[i],
                self.transfer_filt_2D,
                self.transfer_sim_2D,
                dx=self.maps[i].dx,
                dy=self.maps[i].dy,
                dz=self.maps[i].dz,
            )

            self.reverse_normalization(
                i, j
            )  # go back to the previous state to normalize again with a different map-pair

            self.xs.append(my_xs)
            self.k.append(my_k)
            self.rms_mean.append(my_rms_mean)
            self.rms_std.append(my_rms_std)
            self.nmodes.append(my_nmodes)
        self.xs = np.array(self.xs)
        self.k = np.array(self.k)
        self.nmodes = np.array(self.nmodes)
        self.rms_mean = np.array(self.rms_mean)
        self.rms_std = np.array(self.rms_std)
        return self.xs, self.k, self.nmodes, self.rms_mean, self.rms_std

    # COMPUTE ALL THE XS IN 2D
    def calculate_xs_2d(
        self, no_of_k_bins=15
    ):  # here take the number of k-bins as an argument
        n_k = no_of_k_bins
        self.k_bin_edges_par = np.logspace(-2.0, np.log10(1.0), n_k)
        self.k_bin_edges_perp = np.logspace(-2.0 + np.log10(2), np.log10(1.5), n_k)
        calculated_xs = self.get_information()

        # store each cross-spectrum and corresponding k and nmodes by appending to these lists:
        self.xs = []
        self.k = []
        self.nmodes = []
        index = -1
        for i in range(0, len(self.maps) - 1, 2):
            j = i + 1
            index += 1

            self.normalize_weights(i, j)  # normalize weights for given xs pair of maps

            wi = self.maps[i].w.copy()
            wj = self.maps[j].w.copy()
            
            wh_i = np.where((np.log10(wi) < -0.5))
            wh_j = np.where((np.log10(wj) < -0.5))
            wi[wh_i] = 0.0
            wj[wh_j] = 0.0

            full_weight = np.sqrt(wi * wj) #/ np.sqrt(np.mean((wi * wj).flatten()))
            
            full_weight[wi * wj == 0] = 0.0
            full_weight[np.isnan(full_weight)] = 0.0
           
            my_xs, my_k, my_nmodes = tools.compute_cross_spec_perp_vs_par(
                (self.maps[i].map * full_weight, self.maps[j].map * full_weight),
                (self.k_bin_edges_perp, self.k_bin_edges_par),
                dx=self.maps[i].dx,
                dy=self.maps[i].dy,
                dz=self.maps[i].dz,
            )

            self.reverse_normalization(
                i, j
            )  # go back to the previous state to normalize again with a different map-pair

            self.xs.append(my_xs)
            self.k.append(my_k)
            self.nmodes.append(my_nmodes)

            # print(np.allclose(my_xs, np.zeros_like(my_xs)))
            # print(np.any(np.isnan(my_xs)))

            # print("------------------------------------------------")

        self.xs = np.array(self.xs)
        self.k = np.array(self.k)
        self.nmodes = np.array(self.nmodes)

        # print("all close to zero", np.allclose(self.xs, np.zeros_like(self.xs)))
        return self.xs, self.k, self.nmodes

    # COMPUTE ALL THE XS IN 2D
    def calculate_xs_ra_dec_nu(
        self, no_of_k_bins=15
    ):  # here take the number of k-bins as an argument
        n_k = no_of_k_bins
        self.k_bin_edges_par = np.logspace(-2.0, np.log10(1.0), n_k)
        self.k_bin_edges_ra = np.logspace(-2.0 + np.log10(2), np.log10(1.5), n_k)
        self.k_bin_edges_dec = self.k_bin_edges_ra.copy()

        calculated_xs = self.get_information()

        # store each cross-spectrum and corresponding k and nmodes by appending to these lists:
        self.xs = []
        self.k = []
        self.nmodes = []
        index = -1
        for i in range(0, len(self.maps) - 1, 2):
            j = i + 1
            index += 1

            self.normalize_weights(i, j)  # normalize weights for given xs pair of maps

            wi = self.maps[i].w.copy()
            wj = self.maps[j].w.copy()
            
            wh_i = np.where((np.log10(wi) < -0.5))
            wh_j = np.where((np.log10(wj) < -0.5))
            wi[wh_i] = 0.0
            wj[wh_j] = 0.0

            full_weight = np.sqrt(wi * wj) #/ np.sqrt(np.mean((wi * wj).flatten()))
            
            full_weight[wi * wj == 0] = 0.0
            full_weight[np.isnan(full_weight)] = 0.0
           
            my_xs, my_k, my_nmodes = tools.compute_cross_spec_angular2d_vs_par(
                (self.maps[i].map * full_weight, self.maps[j].map * full_weight),
                (self.k_bin_edges_ra, self.k_bin_edges_dec, self.k_bin_edges_par),
                dx=self.maps[i].dx,
                dy=self.maps[i].dy,
                dz=self.maps[i].dz,
            )

            self.reverse_normalization(
                i, j
            )  # go back to the previous state to normalize again with a different map-pair

            self.xs.append(my_xs)
            self.k.append(my_k)
            self.nmodes.append(my_nmodes)

            # print(np.allclose(my_xs, np.zeros_like(my_xs)))
            # print(np.any(np.isnan(my_xs)))

            # print("------------------------------------------------")

        self.xs = np.array(self.xs)
        self.k = np.array(self.k)
        self.nmodes = np.array(self.nmodes)

        # print("all close to zero", np.allclose(self.xs, np.zeros_like(self.xs)))
        return self.xs, self.k, self.nmodes


    # RUN NOISE SIMULATIONS (for all combinations of n maps, to match xs)
    def run_noise_sims(self, n_sims, seed=None):
        self.rms_xs_mean = []
        self.rms_xs_std = []
        for i in range(0, len(self.maps) - 1, 2):
            j = i + 1

            self.normalize_weights(i, j)
            wi = self.maps[i].w
            wj = self.maps[j].w
            wh_i = np.where(np.log10(wi) < -0.5)
            wh_j = np.where(np.log10(wj) < -0.5)
            wi[wh_i] = 0.0
            wj[wh_j] = 0.0
            full_weight = np.sqrt(wi * wj) / np.sqrt(np.mean((wi * wj).flatten()))

            if seed is not None:
                if self.maps[i].feed is not None:
                    feeds = np.array([self.maps[i].feed, self.maps[j].feed])
                else:
                    feeds = np.array([1, 1])

            rms_xs = np.zeros((len(self.k_bin_edges) - 1, n_sims))
            for g in range(n_sims):
                randmap = [
                    np.zeros(self.maps[i].rms.shape),
                    np.zeros(self.maps[i].rms.shape),
                ]
                for l in range(2):
                    if seed is not None:
                        np.random.seed(seed * (g + 1) * (l + 1) * feeds[l])
                    randmap[l] = (
                        np.random.randn(*self.maps[l].rms.shape) * self.maps[l].rms
                    )

                rms_xs[:, g] = tools.compute_cross_spec3d(
                    (randmap[0] * full_weight, randmap[1] * full_weight),
                    self.k_bin_edges,
                    dx=self.maps[i].dx,
                    dy=self.maps[i].dy,
                    dz=self.maps[i].dz,
                )[0]

            self.reverse_normalization(
                i, j
            )  # go back to the previous state to normalize again with a different map-pair

            self.rms_xs_mean.append(np.mean(rms_xs, axis=1))
            self.rms_xs_std.append(np.std(rms_xs, axis=1, ddof = 1))

        return np.array(self.rms_xs_mean), np.array(self.rms_xs_std)

    def run_noise_sims_2d(self, n_sims, seed=None, no_of_k_bins=15):
        n_k = no_of_k_bins

        self.k_bin_edges_par = np.logspace(-2.0, np.log10(1.0), n_k)
        self.k_bin_edges_perp = np.logspace(-2.0 + np.log10(2), np.log10(1.5), n_k)

        self.rms_xs_mean_2D = []
        self.rms_xs_std_2D = []

        # with h5py.File("/mn/stornext/d22/cmbco/comap/nils/pipeline/power_spectrum/transfer_functions/TF_wn_v2.h5", "r") as infile:
        #     k_centers_perp = infile["k_centers_perp"][()]
        #     k_centers_par = infile["k_centers_par"][()]
        #     tf_wn = infile["tf"][()]

        for i in range(0, len(self.maps) - 1, 2):
            j = i + 1

            self.normalize_weights(i, j)
            wi = self.maps[i].w.copy()
            wj = self.maps[j].w.copy()
            
            wh_i = np.where(np.log10(wi) < -0.5)
            wh_j = np.where(np.log10(wj) < -0.5)
            
            wi[wh_i] = 0.0
            wj[wh_j] = 0.0
            
            full_weight = np.sqrt(wi * wj) #/ np.sqrt(np.mean((wi * wj).flatten()))

            full_weight[wi * wj == 0] = 0.0
            full_weight[np.isnan(full_weight)] = 0.0

            if seed is not None:
                if self.maps[i].feed is not None:
                    feeds = np.array([self.maps[i].feed, self.maps[j].feed])
                else:
                    feeds = np.array([1, 1])

            rms_xs = np.zeros(
                (len(self.k_bin_edges_perp) - 1, len(self.k_bin_edges_par) - 1, n_sims)
            )
            for g in range(n_sims):
                randmap = [
                    np.zeros(self.maps[i].rms.shape),
                    np.zeros(self.maps[i].rms.shape),
                ]
                for l in range(2):
                    if seed is not None:
                        np.random.seed(seed * (g + 1) * (l + 1) * feeds[l])
                    randmap[l] = (
                        np.random.randn(*self.maps[l].rms.shape) * self.maps[l].rms
                    )

                rms_xs[:, :, g] = tools.compute_cross_spec_perp_vs_par(
                    (randmap[0] * full_weight, randmap[1] * full_weight),
                    (self.k_bin_edges_perp, self.k_bin_edges_par),
                    dx=self.maps[i].dx,
                    dy=self.maps[i].dy,
                    dz=self.maps[i].dz,
                )[0]

                ## MAY CHANGE IN THE FUTURE!!! THIS IS TO CORRECT WN SIMS TO GET SAME PIPELINE BIAS AS DATA
                # rms_xs[:, :, g] *= tf_wn

            self.reverse_normalization(
                i, j
            )  # go back to the previous state to normalize again with a different map-pair

            self.rms_xs_mean_2D.append(np.mean(rms_xs, axis=2))
            self.rms_xs_std_2D.append(np.std(rms_xs, axis=2, ddof=1))
            
        return np.array(self.rms_xs_mean_2D), np.array(self.rms_xs_std_2D)

    # MAKE SEPARATE H5 FILE FOR EACH 2D XS
    def make_h5_2d(self, outdir, outname=None):
        if outname is None:
            path = os.path.join("spectra_2D/", outdir)
            path = os.path.join(self.params.power_spectrum_dir, path)

            tools.ensure_dir_exists(path)
        
            # if self.params.psx_null_diffmap:
            #     path = os.path.join(path, "null_diffmap")
            #     print(path)
            #     sys.exit()
            #     if not os.path.exists(path):
            #         os.mkdir(path)

            #     path = os.path.join(path, f"{self.null_variable}")
                
            #     if not os.path.exists(path):
            #         os.mkdir(path)

            outname = os.path.join(path, self.outname)


        with h5py.File(outname, "w") as outfile:
            outfile.create_dataset("mappath1", data=self.name_of_map[0])
            outfile.create_dataset("mappath2", data=self.name_of_map[1])
            outfile.create_dataset("xs_2D", data=self.xs[0])
            outfile.create_dataset("k", data=self.k[0])
            outfile.create_dataset("k_bin_edges_perp", data=self.k_bin_edges_perp)
            outfile.create_dataset("k_bin_edges_par", data=self.k_bin_edges_par)
            outfile.create_dataset("nmodes", data=self.nmodes[0])
            
            
            if self.params.psx_white_noise_sim_seed is not None:
                outfile.create_dataset("rms_xs_mean_2D", data=self.rms_xs_mean_2D)
                outfile.create_dataset("rms_xs_std_2D", data=self.rms_xs_std_2D)
                outfile.create_dataset("white_noise_seed", data = self.params.psx_white_noise_sim_seed)
            else:
                outfile.create_dataset("rms_xs_mean_2D", data=self.rms_xs_mean_2D[0])
                outfile.create_dataset("rms_xs_std_2D", data=self.rms_xs_std_2D[0])
            
    
    def read_spectrum(self, indir, inname = None):
        if inname is None:
            path = os.path.join("spectra_2D/", indir)
            path = os.path.join(self.params.power_spectrum_dir, path)
            
            if self.params.psx_null_diffmap:
                path = os.path.join(path, "null_diffmap")                
                path = os.path.join(path, f"{self.null_variable}")


            inname = os.path.join(path, self.outname)
        
        with h5py.File(inname, "r") as infile:
            for key, value in infile.items():
                setattr(self, key, value[()])
    
    def read_and_append_attribute(self, keys, indir, inname = None):
        if inname is None:
            path = os.path.join("spectra_2D/", indir)
            path = os.path.join(self.params.power_spectrum_dir, path)
            
            if self.params.psx_null_diffmap:
                path = os.path.join(path, "null_diffmap")                
                path = os.path.join(path, f"{self.null_variable}")


            inname = os.path.join(path, self.outname)

        with h5py.File(inname, "r") as infile:
            for key, value in infile.items():
                if key in keys:
                    setattr(self, key, value[()])
    