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
    def __init__(self, mappaths, params, cosmology, split_keys, feed1=None, feed2=None):
        
        split_base_number = params.split_base_number

        self.name_of_map = mappaths  
        if len(self.name_of_map) < 2:
            raise ValueError("Can only compute cross-spectra when two map paths are provided.")
        
        self.names = []
        self.maps = []

        n_list = list(range(split_base_number))
        
        all_different_possibilities = list(itr.combinations(n_list, 2))  

        self.how_many_combinations = len(all_different_possibilities)

        combination1 = split_keys[0]
        combination1 = combination1.split("/map_")[-1]
        combination2 = split_keys[1]
        combination2 = combination2.split("/map_")[-1]
        
        
        split_map1 = map_cosmo.MapCosmo(
            params,
            mappaths[0], 
            cosmology,
            feed1, 
            split_keys[0], 
            )

        split_map2 = map_cosmo.MapCosmo(
            params,
            mappaths[1], 
            cosmology,
            feed2, 
            split_keys[1], 
            )

        name1 = f"{combination1}_feed{feed1}"
        name2 = f"{combination2}_feed{feed2}"

        print("XS_code:", name1, name2)

        self.names.append(name1)
        self.names.append(name2)

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
            # print ('Computing 2D xs between ' + calculated_xs[index][1] + ' and ' + calculated_xs[index][2])
            self.normalize_weights(i, j)  # normalize weights for given xs pair of maps

            wi = self.maps[i].w
            wj = self.maps[j].w
            wh_i = np.where((np.log10(wi) < -0.5))
            wh_j = np.where((np.log10(wj) < -0.5))
            wi[wh_i] = 0.0
            wj[wh_j] = 0.0

            # wi[np.isnan(wi)] = 0.0
            # wj[np.isnan(wj)] = 0.0

            full_weight = np.sqrt(wi * wj) / np.sqrt(np.mean((wi * wj).flatten()))
            
            # full_weight[wi * wj == 0] = 0.0
            full_weight[np.isnan(full_weight)] = 0.0
            """
           print("------------------------------------------------")
           print(wi[(np.isnan(wi) == False) & (np.isinf(wi) == False)])
           print(wj[(np.isnan(wj) == False) & (np.isinf(wi) == False)])
           print(np.all(np.log10(wi) < -0.5), np.any(np.log10(wi) < -0.5))
           print(np.all(np.log10(wj) < -0.5), np.any(np.log10(wj) < -0.5))
           print(np.allclose(self.maps[i].map, np.zeros_like(self.maps[i].map)))
           print(np.allclose(self.maps[j].map, np.zeros_like(self.maps[j].map)))
           print(np.any(np.isnan(wi)), np.any(np.isnan(wj)), np.sum(np.isnan(wi)), np.sum(np.isnan(wi) == False), np.prod(wi.shape), np.sum(np.isnan(wi)), np.sum(np.isnan(wi) == False), np.prod(wi.shape))
           
           print("isnan maps and weights", np.any(np.isnan(self.maps[i].map* full_weight)), np.any(np.isnan(self.maps[j].map* full_weight)), np.any(np.isnan(full_weight)), np.sum(np.isnan(full_weight)), np.sum(np.isnan(full_weight) == False), np.prod(full_weight.shape))
           print(np.allclose(wi, np.zeros_like(wi)), np.allclose(wj, np.zeros_like(wj)), np.all(np.isnan(wi)), np.all(np.isnan(wj)), np.any(np.isnan(wi)), np.any(np.isnan(wj)))
           """
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
            self.rms_xs_std.append(np.std(rms_xs, axis=1))
        return np.array(self.rms_xs_mean), np.array(self.rms_xs_std)

    def run_noise_sims_2d(self, n_sims, seed=None, no_of_k_bins=15):
        n_k = no_of_k_bins

        self.k_bin_edges_par = np.logspace(-2.0, np.log10(1.0), n_k)
        self.k_bin_edges_perp = np.logspace(-2.0 + np.log10(2), np.log10(1.5), n_k)

        self.rms_xs_mean_2D = []
        self.rms_xs_std_2D = []
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

            self.reverse_normalization(
                i, j
            )  # go back to the previous state to normalize again with a different map-pair

            self.rms_xs_mean_2D.append(np.mean(rms_xs, axis=2))
            self.rms_xs_std_2D.append(np.std(rms_xs, axis=2))
        return np.array(self.rms_xs_mean_2D), np.array(self.rms_xs_std_2D)

    # MAKE SEPARATE H5 FILE FOR EACH XS
    def make_h5(self, outdir, outname=None):

        for index in range(self.how_many_combinations):
            i = index * 2
            j = i + 1

            if outname is None:
                tools.ensure_dir_exists("spectra/" + outdir)
                outname = (
                    "spectra/"
                    + outdir
                    + "/xs_"
                    + self.get_information()[index][1]
                    + "_and_"
                    + self.get_information()[index][2]
                    + ".h5"
                )
                # outname = 'spectra/xs_' + self.get_information()[index][1] + '_and_'+ self.get_information()[index][2] + '.h5'

            f1 = h5py.File(outname, "w")
            try:
                f1.create_dataset("mappath1", data=self.maps[i].mappath)
                f1.create_dataset("mappath2", data=self.maps[j].mappath)
                f1.create_dataset("xs", data=self.xs[index])
                f1.create_dataset("k", data=self.k[index])
                f1.create_dataset("k_bin_edges", data=self.k_bin_edges)
                f1.create_dataset("nmodes", data=self.nmodes[index])
            except:
                print("No cross-spectrum calculated.")
                return
            try:
                f1.create_dataset("rms_xs_mean", data=self.rms_xs_mean[index])
                f1.create_dataset("rms_xs_std", data=self.rms_xs_std[index])
            except:
                pass

            f1.close()

    # MAKE SEPARATE H5 FILE FOR EACH 2D XS
    def make_h5_2d(self, outdir, data_dir, outname=None):

        for index in range(self.how_many_combinations):
            i = index * 2
            j = i + 1

            if outname is None:
                path = os.path.join("spectra_2D/", outdir)
                path = os.path.join(data_dir, path)

                tools.ensure_dir_exists(path)
                outname = (
                    "xs_2D_"
                    + self.get_information()[index][1]
                    + "_and_"
                    + self.get_information()[index][2]
                    + ".h5"
                )
                outname = os.path.join(path, outname)
                # outname = 'spectra_2D/xs_2D_' + self.get_information()[index][1] + '_and_'+ self.get_information()[index][2] + '.h5'

            f1 = h5py.File(outname, "w")
            try:
                f1.create_dataset("mappath1", data=self.maps[i].mappath)
                f1.create_dataset("mappath2", data=self.maps[j].mappath)
                f1.create_dataset("xs_2D", data=self.xs[index])
                f1.create_dataset("k", data=self.k[index])
                f1.create_dataset("k_bin_edges_perp", data=self.k_bin_edges_perp)
                f1.create_dataset("k_bin_edges_par", data=self.k_bin_edges_par)
                f1.create_dataset("nmodes", data=self.nmodes[index])
            except:
                print("No cross-spectrum calculated.")
                return
            try:
                f1.create_dataset("rms_xs_mean_2D", data=self.rms_xs_mean_2D[index])
                f1.create_dataset("rms_xs_std_2D", data=self.rms_xs_std_2D[index])
            except:
                pass

            f1.close()
