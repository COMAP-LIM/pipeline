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
        
        self.angle2Mpc = split_map1.angle2Mpc.value
        self.map_dx = split_map1.dx
        self.map_dy = split_map1.dy
        self.map_dz = split_map1.dz

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

            full_weight = np.sqrt(wi * wj) / np.sqrt(np.mean((wi * wj).flatten()))
            
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
        if self.maps[0].params.psx_nyquist_bin_limit:
            self.k_bin_edges_par = np.logspace(np.log10(self.maps[0].min_k_z), np.log10(self.maps[0].nyquist_z), number_of_k_bin_edges)
            spacial_bin_max_limit = np.min(
                                    (np.min((self.maps[0].nyquist_x, self.maps[0].nyquist_x)),
                                    np.min((self.maps[1].nyquist_y, self.maps[1].nyquist_y)),)
            )
            spacial_bin_min_limit = np.max(
                                    (np.max((self.maps[0].min_k_x, self.maps[0].min_k_x)),
                                    np.max((self.maps[1].min_k_y, self.maps[1].min_k_y)),)
            )

            self.k_bin_edges_ra = np.logspace(np.log10(spacial_bin_min_limit), np.log10(spacial_bin_max_limit), number_of_k_bin_edges)
            self.k_bin_edges_dec = self.k_bin_edges_ra.copy()
        else:
            self.k_bin_edges_par = np.logspace(
                np.log10(self.maps[0].params.psx_k_spectral_bin_min), 
                np.log10(self.maps[0].params.psx_k_spectral_bin_max), 
                number_of_k_bin_edges
            )
            self.k_bin_edges_ra = np.logspace(
                np.log10(self.maps[0].params.psx_k_angular_bin_min), 
                np.log10(self.maps[0].params.psx_k_angular_bin_max), 
                number_of_k_bin_edges
            )
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

            full_weight = np.sqrt(wi * wj) / np.sqrt(np.mean((wi * wj).flatten()))
            
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


    def run_noise_sims_3d(self, n_sims, seed=None, no_of_k_bins=15):
        n_k = no_of_k_bins

        if self.maps[0].params.psx_nyquist_bin_limit:
            self.k_bin_edges_par = np.logspace(np.log10(self.maps[0].min_k_z), np.log10(self.maps[0].nyquist_z), number_of_k_bin_edges)
            spacial_bin_max_limit = np.min(
                                    (np.min((self.maps[0].nyquist_x, self.maps[0].nyquist_x)),
                                    np.min((self.maps[1].nyquist_y, self.maps[1].nyquist_y)),)
            )
            spacial_bin_min_limit = np.max(
                                    (np.max((self.maps[0].min_k_x, self.maps[0].min_k_x)),
                                    np.max((self.maps[1].min_k_y, self.maps[1].min_k_y)),)
            )

            self.k_bin_edges_ra = np.logspace(np.log10(spacial_bin_min_limit), np.log10(spacial_bin_max_limit), number_of_k_bin_edges)
            self.k_bin_edges_dec = self.k_bin_edges_ra.copy()
        else:
            self.k_bin_edges_par = np.logspace(
                np.log10(self.maps[0].params.psx_k_spectral_bin_min), 
                np.log10(self.maps[0].params.psx_k_spectral_bin_max), 
                number_of_k_bin_edges
            )
            self.k_bin_edges_ra = np.logspace(
                np.log10(self.maps[0].params.psx_k_angular_bin_min), 
                np.log10(self.maps[0].params.psx_k_angular_bin_max), 
                number_of_k_bin_edges
            )
            self.k_bin_edges_dec = self.k_bin_edges_ra.copy()


        self.rms_xs_mean_3D = []
        self.rms_xs_std_3D = []
        self.all_noise_simulations = []

        for i in range(0, len(self.maps) - 1, 2):
            j = i + 1

            self.normalize_weights(i, j)
            wi = self.maps[i].w.copy()
            wj = self.maps[j].w.copy()
            
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
                (len(self.k_bin_edges_dec) - 1, len(self.k_bin_edges_ra) - 1, len(self.k_bin_edges_par) - 1, n_sims)
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

                rms_xs[:, :, :, g] = tools.compute_cross_spec_angular2d_vs_par(
                    (randmap[0] * full_weight, randmap[1] * full_weight),
                    (self.k_bin_edges_perp, self.k_bin_edges_par),
                    dx=self.maps[i].dx,
                    dy=self.maps[i].dy,
                    dz=self.maps[i].dz,
                )[0]
            
            #rms_xs *= self.params.psx_white_noise_transfer_function[..., None]
            
            self.reverse_normalization(
                i, j
            )  # go back to the previous state to normalize again with a different map-pair

            self.white_noise_covariance = np.cov(rms_xs.reshape((n_k - 1) ** 2, n_sims), ddof=1)

            self.rms_xs_mean_3D.append(np.mean(rms_xs, axis=2))
            self.rms_xs_std_3D.append(np.std(rms_xs, axis=2, ddof=1))
            self.all_noise_simulations.append(rms_xs)

            self.all_noise_simulations = np.array(self.all_noise_simulations)[0, ...].transpose(2, 0, 1)

        return np.array(self.rms_xs_mean_3D), np.array(self.rms_xs_std_3D), self.white_noise_covariance

    def run_noise_sims_2d(self, n_sims, seed=None, no_of_k_bins=15):
        n_k = no_of_k_bins

        self.k_bin_edges_par = np.logspace(-2.0, np.log10(1.0), n_k)
        self.k_bin_edges_perp = np.logspace(-2.0 + np.log10(2), np.log10(1.5), n_k)

        self.rms_xs_mean_2D = []
        self.rms_xs_std_2D = []
        self.all_noise_simulations = []

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
            
            rms_xs *= self.params.psx_white_noise_transfer_function[..., None]
            
            self.reverse_normalization(
                i, j
            )  # go back to the previous state to normalize again with a different map-pair

            self.white_noise_covariance = np.cov(rms_xs.reshape((n_k - 1) ** 2, n_sims), ddof=1)

            self.rms_xs_mean_2D.append(np.mean(rms_xs, axis=2))
            self.rms_xs_std_2D.append(np.std(rms_xs, axis=2, ddof=1))
            self.all_noise_simulations.append(rms_xs)

            self.all_noise_simulations = np.array(self.all_noise_simulations)[0, ...].transpose(2, 0, 1)

        return np.array(self.rms_xs_mean_2D), np.array(self.rms_xs_std_2D), self.white_noise_covariance

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
            
            outfile.create_dataset("angle2Mpc", data = self.angle2Mpc)
            outfile.create_dataset("dx", data = self.map_dx)
            outfile.create_dataset("dy", data = self.map_dx)
            outfile.create_dataset("dz", data = self.map_dx)

            outfile["angle2Mpc"].attrs["unit"] = "Mpc/arcmin"
            outfile["dx"].attrs["unit"] = "Mpc"
            outfile["dy"].attrs["unit"] = "Mpc"
            outfile["dz"].attrs["unit"] = "Mpc"
            

            if self.params.psx_white_noise_sim_seed is not None:
                outfile.create_dataset("rms_xs_mean_2D", data=self.rms_xs_mean_2D)
                outfile.create_dataset("rms_xs_std_2D", data=self.rms_xs_std_2D)
                outfile.create_dataset("white_noise_covariance", data=self.white_noise_covariance)
                outfile.create_dataset("white_noise_seed", data = self.params.psx_white_noise_sim_seed)
            else:
                outfile.create_dataset("rms_xs_mean_2D", data=self.rms_xs_mean_2D[0])
                outfile.create_dataset("rms_xs_std_2D", data=self.rms_xs_std_2D[0])
                outfile.create_dataset("white_noise_covariance", data=self.white_noise_covariance)
                outfile.create_dataset("white_noise_simulation", data = self.all_noise_simulations)
    
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
    