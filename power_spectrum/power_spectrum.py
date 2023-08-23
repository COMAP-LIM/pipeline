import numpy as np
import h5py
import tools as tools


class PowerSpectrum:
    def __init__(self, my_map):
        self.map = my_map
        self.weights_are_normalized = False

    def normalize_weights(self):
        self.map.w = self.map.w / np.sqrt(np.mean(self.map.w.flatten() ** 2))
        self.weights_are_normalized = True

    def calculate_ps(self, do_2d=False, number_of_k_bin_edges = 15):

        if not self.weights_are_normalized:
            self.normalize_weights()
        if do_2d:
            self.k_bin_edges_par = np.logspace(-2.0, np.log10(1.0), number_of_k_bin_edges)
            self.k_bin_edges_perp = np.logspace(-2.0 + np.log10(2), np.log10(1.5), number_of_k_bin_edges)

            self.ps_2d, self.k, self.nmodes = tools.compute_power_spec_perp_vs_par(
                self.map.map * self.map.w,
                (self.k_bin_edges_perp, self.k_bin_edges_par),
                dx=self.map.dx,
                dy=self.map.dy,
                dz=self.map.dz,
            )
            return self.ps_2d, self.k, self.nmodes
        else:
            self.k_bin_edges = np.logspace(-2.0, np.log10(1.5), number_of_k_bin_edges)
            self.ps, self.k, self.nmodes = tools.compute_power_spec3d(
                self.map.map * self.map.w,
                self.k_bin_edges,
                dx=self.map.dx,
                dy=self.map.dy,
                dz=self.map.dz,
            )
            return self.ps, self.k, self.nmodes

    def run_noise_sims(self, n_sims):
        if not self.weights_are_normalized:
            self.normalize_weights()

        rms_ps = np.zeros((len(self.k_bin_edges) - 1, n_sims))
        for i in range(n_sims):
            randmap = self.map.rms * np.random.randn(*self.map.rms.shape)

            rms_ps[:, i] = tools.compute_power_spec3d(
                randmap * self.map.w,
                self.k_bin_edges,
                dx=self.map.dx,
                dy=self.map.dy,
                dz=self.map.dz,
            )[0]
        self.rms_ps_mean = np.mean(rms_ps, axis=1)
        self.rms_ps_std = np.std(rms_ps, axis=1)
        return self.rms_ps_mean, self.rms_ps_std

    def make_h5(self, outname=None):
        if outname is None:
            folder = "/mn/stornext/d16/cmbco/comap/protodir/spectra/"
            tools.ensure_dir_exists(folder)
            outname = folder + "ps" + self.map.save_string + ".h5"

        f1 = h5py.File(outname, "w")
        try:
            f1.create_dataset("mappath", data=self.map.mappath)
            f1.create_dataset("ps", data=self.ps)
            f1.create_dataset("k", data=self.k)
            f1.create_dataset("k_bin_edges", data=self.k_bin_edges)
            f1.create_dataset("nmodes", data=self.nmodes)
        except:
            print("No power spectrum calculated.")
            return
        try:
            f1.create_dataset("ps_2d", data=self.ps_2d)
        except:
            pass

        try:
            f1.create_dataset("rms_ps_mean", data=self.rms_ps_mean)
            f1.create_dataset("rms_ps_std", data=self.rms_ps_std)
        except:
            pass
        f1.close()


class CrossSpectrum:
    def __init__(self, my_map, my_map2):
        self.maps = []
        self.maps.append(my_map)
        self.maps.append(my_map2)
        self.weights_are_normalized = False

    def normalize_weights(self):
        norm = np.sqrt(np.mean((self.maps[0].w * self.maps[1].w).flatten()))
        self.maps[0].w = self.maps[0].w / norm
        self.maps[1].w = self.maps[1].w / norm
        self.weights_are_normalized = True

    def calculate_xs(self, number_of_k_bin_edges = 15):
        
        self.k_bin_edges = np.logspace(-2.0, np.log10(1.5), number_of_k_bin_edges)

        if not self.weights_are_normalized:
            self.normalize_weights()

        w = np.sqrt(self.maps[0].w * self.maps[1].w)
        
        self.xs, self.k, self.nmodes = tools.compute_cross_spec3d(
            (self.maps[0].map * w, self.maps[1].map * w),
            self.k_bin_edges,
            dx=self.maps[0].dx,
            dy=self.maps[0].dy,
            dz=self.maps[0].dz,
        )
        return self.xs, self.k, self.nmodes


    def calculate_xs_ra_dec_nu(self, number_of_k_bin_edges = 15):
        
        if self.maps[0].params.psx_nyquist_bin_limit:
            self.k_bin_edges_par = np.logspace(np.log10(self.maps[0].min_k_z), np.log10(self.maps[0].nyquist_z), number_of_k_bin_edges)
            spacial_bin_max_limit = np.min(
                                    (np.min((self.maps[0].nyquist_x, self.maps[1].nyquist_x)),
                                    np.min((self.maps[0].nyquist_y, self.maps[1].nyquist_y)),)
            )
            spacial_bin_min_limit = np.max(
                                    (np.max((self.maps[0].min_k_x, self.maps[1].min_k_x)),
                                    np.max((self.maps[0].min_k_y, self.maps[1].min_k_y)),)
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

        if not self.weights_are_normalized:
            self.normalize_weights()
        
        w = np.sqrt(self.maps[0].w * self.maps[1].w)

        self.xs, self.k, self.nmodes = tools.compute_cross_spec_angular2d_vs_par(
            (self.maps[0].map * w, self.maps[1].map * w),
            (self.k_bin_edges_ra, self.k_bin_edges_dec, self.k_bin_edges_par),
            dx=self.maps[0].dx,
            dy=self.maps[0].dy,
            dz=self.maps[0].dz,
        )
        return self.xs, self.k, self.nmodes

    def calculate_xs_2d(self, number_of_k_bin_edges = 15):
        

        self.k_bin_edges_par = np.logspace(-2.0, np.log10(1.0), number_of_k_bin_edges)
        self.k_bin_edges_perp = np.logspace(-2.0 + np.log10(2), np.log10(1.5), number_of_k_bin_edges)
        
        # if self.maps[0].params.psx_nyquist_bin_limit:
        #     self.k_bin_edges_par = np.logspace(np.log10(self.maps[0].min_k_z), np.log10(self.maps[0].nyquist_z), number_of_k_bin_edges)
        #     spacial_bin_max_limit = np.min(
        #                             (np.min((self.maps[0].nyquist_x, self.maps[1].nyquist_x)),
        #                             np.min((self.maps[0].nyquist_y, self.maps[1].nyquist_y)),)
        #     )
        #     spacial_bin_min_limit = np.max(
        #                             (np.max((self.maps[0].min_k_x, self.maps[1].min_k_x)),
        #                             np.max((self.maps[0].min_k_y, self.maps[1].min_k_y)),)
        #     )

        #     self.k_bin_edges_perp = np.logspace(np.log10(spacial_bin_min_limit), np.log10(spacial_bin_max_limit), number_of_k_bin_edges)
        # else:
        #     self.k_bin_edges_par = np.logspace(
        #         np.log10(self.maps[0].params.psx_k_spectral_bin_min), 
        #         np.log10(self.maps[0].params.psx_k_spectral_bin_max), 
        #         number_of_k_bin_edges
        #     )
        #     self.k_bin_edges_perp = np.logspace(
        #         np.log10(self.maps[0].params.psx_k_angular_bin_min), 
        #         np.log10(self.maps[0].params.psx_k_angular_bin_max), 
        #         number_of_k_bin_edges
        #     )

        if not self.weights_are_normalized:
            self.normalize_weights() 

        w = np.sqrt(self.maps[0].w * self.maps[1].w)
        
        self.xs, self.k, self.nmodes = tools.compute_cross_spec_perp_vs_par(
            (self.maps[0].map * w, self.maps[1].map * w),
            (self.k_bin_edges_perp, self.k_bin_edges_par),
            dx=self.maps[0].dx,
            dy=self.maps[0].dy,
            dz=self.maps[0].dz,
        )
        return self.xs, self.k, self.nmodes

    def run_noise_sims(self, n_sims, seed=None):
        if not self.weights_are_normalized:
            self.normalize_weights()

        if seed is not None:
            if self.maps[0].feed is not None:
                feeds = np.array([self.maps[0].feed, self.maps[1].feed])
            else:
                feeds = np.array([1, 1])

        self.rms_xs = np.zeros((len(self.k_bin_edges) - 1, n_sims))
        for i in range(n_sims):
            randmap = [
                np.zeros(self.maps[0].rms.shape),
                np.zeros(self.maps[0].rms.shape),
            ]
            for j in range(len(self.maps)):
                if seed is not None:
                    np.random.seed(seed * (i + 1) * (j + 1) * feeds[j])
                randmap[j] = np.random.randn(*self.maps[j].rms.shape) * self.maps[j].rms
            w = np.sqrt(self.maps[0].w * self.maps[1].w)
            self.rms_xs[:, i] = tools.compute_cross_spec3d(
                (randmap[0] * w, randmap[1] * w),
                self.k_bin_edges,
                dx=self.maps[0].dx,
                dy=self.maps[0].dy,
                dz=self.maps[0].dz,
            )[0]

        self.rms_xs_mean = np.mean(self.rms_xs, axis=1)
        self.rms_xs_std = np.std(self.rms_xs, axis=1)
        return self.rms_xs_mean, self.rms_xs_std

    def make_h5(self, outname=None, save_noise_realizations=False):
        if outname is None:
            tools.ensure_dir_exists("spectra")
            outname = (
                "spectra/xs"
                + self.maps[0].save_string
                + "_"
                + self.maps[1].map_string
                + ".h5"
            )

        f1 = h5py.File(outname, "w")
        try:
            f1.create_dataset("mappath1", data=self.maps[0].mappath)
            f1.create_dataset("mappath2", data=self.maps[1].mappath)
            f1.create_dataset("xs", data=self.xs)
            f1.create_dataset("k", data=self.k)
            f1.create_dataset("k_bin_edges", data=self.k_bin_edges)
            f1.create_dataset("nmodes", data=self.nmodes)
        except:
            print("No power spectrum calculated.")
            return
        try:
            f1.create_dataset("rms_xs_mean", data=self.rms_xs_mean)
            f1.create_dataset("rms_xs_std", data=self.rms_xs_std)
        except:
            pass
        if save_noise_realizations:
            try:
                f1.create_dataset("rms_xs", data=self.rms_xs)
            except:
                pass
        f1.close()
