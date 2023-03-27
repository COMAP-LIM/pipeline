import numpy as np
import h5py
import os
import git

class level2_file:
    def __init__(self, scanid, mjd_start, mjd_stop, scantype, fieldname, l1_filename, l2_filename, filter_list, params):
        self.params = params
        self.filter_list = filter_list
        self.level2_dir = params.level2_dir
        self.scanid = scanid
        self.scanid_str = f"{scanid:09d}"
        self.obsid_str = self.scanid_str[:-2]
        self.obsid = int(self.obsid_str)
        self.mjd_start = mjd_start
        self.mjd_stop = mjd_stop
        self.fieldname = fieldname
        self.l1_filename = l1_filename
        self.l2_filename = fieldname + "_" + self.scanid_str
        self.feature_bit = scantype
        self.is_sim = False  # Whether this is or contains a simulation
        if scantype&2**4:
            self.scantype = "circ"
        elif scantype&2**5:
            self.scantype = "ces"
        elif scantype&2**15:
            self.scantype = "liss"

    def load_level1_data(self):
        with h5py.File(self.l1_filename, "r") as f:
            self.tod_times = f["/spectrometer/MJD"][()]
            self.hk_times = f["/hk/array/frame/utc"][()]
            # Start and stop the scan at the first points *inside* the edges defined by mjd_start and mjd_stop.
            self.scan_start_idx = np.searchsorted(self.tod_times, self.mjd_start, side="left")
            self.scan_stop_idx = np.searchsorted(self.tod_times, self.mjd_stop, side="right")  # stop_idx is the first index NOT in the scan.
            self.scan_start_idx_hk = np.searchsorted(self.hk_times, self.mjd_start, side="left")
            self.scan_stop_idx_hk = np.searchsorted(self.hk_times, self.mjd_stop, side="right")
            self.tod_times = self.tod_times[self.scan_start_idx:self.scan_stop_idx]
            self.tod_times_seconds = (self.tod_times-self.tod_times[0])*24*60*60
            self.samprate = 1.0/(self.tod_times_seconds[1] - self.tod_times_seconds[0])
            self.feeds          = f["/spectrometer/feeds"][()]
            if 20 in self.feeds:  # If feed 20 is included in the l1 file, slice it away.
                self.feeds_slice = slice(self.feeds.shape[0]-1)
            else:
                self.feeds_slice = slice(self.feeds.shape[0])
            self.feeds = self.feeds[self.feeds_slice]
            self.array_features = f["/hk/array/frame/features"][()]
            self.array_time     = f["/hk/array/frame/utc"][()]
            self.az             = f["/spectrometer/pixel_pointing/pixel_az"][self.feeds_slice,self.scan_start_idx:self.scan_stop_idx]
            self.el             = f["/spectrometer/pixel_pointing/pixel_el"][self.feeds_slice,self.scan_start_idx:self.scan_stop_idx]
            self.ra             = f["/spectrometer/pixel_pointing/pixel_ra"][self.feeds_slice,self.scan_start_idx:self.scan_stop_idx]
            self.dec            = f["/spectrometer/pixel_pointing/pixel_dec"][self.feeds_slice,self.scan_start_idx:self.scan_stop_idx]
            self.tod            = f["/spectrometer/tod"][self.feeds_slice,:,:,self.scan_start_idx:self.scan_stop_idx]
            self.freqs          = f["/spectrometer/frequency"][()]
            self.sb_mean = np.nanmean(self.tod, axis=2)
            self.tod_mean = np.nanmean(self.tod, axis=-1)
            self.tofile_dict = {}  # Dict for adding custom data, which will be written to level2 file.
            self.Nfeeds = self.tod.shape[0]
            self.Nsb = self.tod.shape[1]
            self.Nfreqs = self.tod.shape[2]
            self.Ntod = self.tod.shape[3]
            self.corr_template = np.zeros((self.Nfeeds, self.Nsb*self.Nfreqs, self.Nsb*self.Nfreqs))  # The correlated induced by different filters, to be subtracted in masking.
            dir_path = os.path.dirname(os.path.realpath(__file__))  # Path to current directory.
            self.git_hash = git.Repo(dir_path, search_parent_directories=True).head.object.hexsha  # Current git commit hash.

            ### Preliminary Masking ###
            self.mask_temporal = np.ones((self.Nfeeds, self.Ntod), dtype=bool)
            self.freqmask = np.ones((self.Nfeeds, self.Nsb, self.Nfreqs), dtype=bool)
            self.freqmask_reason = np.zeros_like(self.freqmask, dtype=int)
            self.freqmask_reason_string = []
            self.freqmask_counter = 0
            self.freqmask[self.feeds==20] = False
            self.freqmask_reason[self.feeds==20] += 2**self.freqmask_counter
            self.freqmask_counter += 1
            self.freqmask_reason_string.append("Feed 20")
            self.n_nans = np.sum(~np.isfinite(self.tod), axis=-1)
            self.freqmask[self.n_nans > 0] = False
            self.freqmask_reason[self.n_nans > 0] += 2**self.freqmask_counter
            self.freqmask_counter += 1
            self.freqmask_reason_string.append("NaN or inf in TOD")
            self.freqmask[:,:,:2] = False
            self.freqmask[:,:,512] = False
            self.freqmask_reason[:,:,:2] += 2**self.freqmask_counter
            self.freqmask_reason[:,:,512] += 2**self.freqmask_counter
            if self.params.sbA_num_masked_channels != 0:
                self.freqmask[:,(0,1),-self.params.sbA_num_masked_channels:] = False
                self.freqmask_reason[:,(0,1),-self.params.sbA_num_masked_channels:] += 2**self.freqmask_counter
            self.freqmask_counter += 1
            self.freqmask_reason_string.append("Marked channels")
            
            

            self.tod[~self.freqmask] = np.nan
            self.acceptrate = np.mean(self.freqmask, axis=-1)

            ### Frequency bins ###
            self.flipped_sidebands = []
            self.freq_bin_edges = np.zeros((self.freqs.shape[0], self.freqs.shape[1]+1))
            self.freq_bin_centers = np.zeros_like(self.freqs)
            for isb in range(self.freqs.shape[0]):
                delta_nu = self.freqs[isb,1] - self.freqs[isb,0]
                self.freq_bin_edges[isb,:-1] = self.freqs[isb]
                self.freq_bin_edges[isb,-1] = self.freq_bin_edges[isb,-2] + delta_nu
                if delta_nu < 0:
                    self.flipped_sidebands.append(isb)
                    self.freq_bin_edges[isb] = self.freq_bin_edges[isb,::-1]
                    self.freqs[isb] = self.freqs[isb,::-1]
                    self.tod[:,isb,:] = self.tod[:,isb,::-1]
                    self.tod_mean[:,isb,:] = self.tod_mean[:,isb,::-1]
                    self.freqmask[:,isb,:] = self.freqmask[:,isb,::-1]
                    self.freqmask_reason[:,isb,:] = self.freqmask_reason[:,isb,::-1]
                    self.n_nans[:,isb,:] = self.n_nans[:,isb,::-1]
                for ifreq in range(self.freqs.shape[1]):
                    self.freq_bin_centers[isb,ifreq] = np.mean(self.freq_bin_edges[isb,ifreq:ifreq+2])


    def write_level2_data(self, name_extension=""):
        self.acceptrate = np.mean(self.freqmask, axis=(-1))
        self.sigma0 = np.zeros((self.Nfeeds, self.Nsb, self.Nfreqs))
        self.chi2 = np.zeros((self.Nfeeds, self.Nsb, self.Nfreqs))
        self.Ntod_effective = np.zeros((self.Nfeeds))
        for ifeed in range(self.Nfeeds):
            self.Ntod_effective[ifeed] = int(np.sum(self.mask_temporal[ifeed]))
            tod_local = self.tod[ifeed][:,:,self.mask_temporal[ifeed]]
            self.sigma0[ifeed] = np.std(tod_local[:,:,1:] - tod_local[:,:,:-1], axis=-1)/np.sqrt(2)
            self.chi2[ifeed] = (np.sum(tod_local**2, axis=-1)/self.sigma0[ifeed]**2 - self.Ntod_effective[ifeed])/np.sqrt(2*self.Ntod_effective[ifeed])
        outpath = os.path.join(self.level2_dir, self.fieldname)
        if not os.path.exists(outpath):
            os.mkdir(outpath)
        # We write to a hidden file (starting with a ".") first, and then rename it. This makes it safe to abort the program during writes:
        temp_outfilename = os.path.join(outpath, "." + self.l2_filename + name_extension + ".h5")
        outfilename = os.path.join(outpath, self.l2_filename + name_extension + ".h5")
        with h5py.File(temp_outfilename, "w") as f:
            # Hardcoded level2 parameters:
            if self.params.use_l2_compression:
                f.create_dataset("tod", (self.Nfeeds, self.Nsb, self.Nfreqs, self.Ntod), chunks=(1,1,1,self.Ntod), data=self.tod, compression="gzip", compression_opts=1, shuffle=True)
            else:
                f["tod"] = self.tod
            f["feeds"] = self.feeds
            f["time"] = self.tod_times
            f["mjd_start"] = self.tod_times[0]
            f["samprate"] = self.samprate
            f["mean_tp"] = self.tod_mean
            f["sb_mean"] = self.sb_mean
            f["freq_bin_edges"] = self.freq_bin_edges
            f["freq_bin_centers"] = self.freq_bin_centers
            f["acceptrate"] = self.acceptrate
            f["mask_temporal"] = self.mask_temporal
            f["freqmask_full"] = self.freqmask
            f["freqmask_reason"] = self.freqmask_reason
            f["freqmask_reason_string"] = np.array(self.freqmask_reason_string, dtype="S100")
            f["cal_method"] = 2
            f["feature"] = self.feature_bit
            f["scanid"] = self.scanid
            f["obsid"] = self.obsid
            f["sigma0"] = self.sigma0
            f["n_nan"] = self.n_nans
            f["chi2"] = self.chi2
            f["point_cel"] = np.zeros((self.Nfeeds, self.Ntod, 2))
            f["point_cel"][:,:,0] = self.ra
            f["point_cel"][:,:,1] = self.dec
            f["point_tel"] = np.zeros((self.Nfeeds, self.Ntod, 2))
            f["point_tel"][:,:,0] = self.az
            f["point_tel"][:,:,1] = self.el
            f["decimation_time"] = 1
            f["is_sim"] = self.is_sim
            f["git_hash"] = self.git_hash

            # Custom data (usually from the filters):
            for key in self.tofile_dict:  
                f[key] = self.tofile_dict[key]
            for key in vars(self.params):  # Writing entire parameter file to separate hdf5 group.
                if getattr(self.params, key) == None:  # hdf5 didn't like the None type.
                    f[f"params/{key}"] = "None"
                else:
                    f[f"params/{key}"] = getattr(self.params, key)
            # Copy from l1 file:
            with h5py.File(self.l1_filename, "r") as l1file:
                f["hk_airtemp"]     = l1file["hk/array/weather/airTemperature"][self.scan_start_idx_hk:self.scan_stop_idx_hk]
                f["hk_dewtemp"]     = l1file["hk/array/weather/dewPointTemp"][self.scan_start_idx_hk:self.scan_stop_idx_hk]
                f["hk_humidity"]    = l1file["hk/array/weather/relativeHumidity"][self.scan_start_idx_hk:self.scan_stop_idx_hk]
                f["hk_pressure"]    = l1file["hk/array/weather/pressure"][self.scan_start_idx_hk:self.scan_stop_idx_hk]
                f["hk_rain"]        = l1file["hk/array/weather/rainToday"][self.scan_start_idx_hk:self.scan_stop_idx_hk]
                f["hk_winddir"]     = l1file["hk/array/weather/windDirection"][self.scan_start_idx_hk:self.scan_stop_idx_hk]
                f["hk_windspeed"]   = l1file["hk/array/weather/windSpeed"][self.scan_start_idx_hk:self.scan_stop_idx_hk]
                f["hk_mjd"]         = l1file["hk/array/weather/utc"][self.scan_start_idx_hk:self.scan_stop_idx_hk]
            # pix2ind_python = np.zeros(20, dtype=int) + 999
            pix2ind_fortran = np.zeros(20, dtype=int)
            for ifeed in range(self.Nfeeds):
                # pix2ind_python[self.feeds[ifeed]-1] = ifeed
                pix2ind_fortran[self.feeds[ifeed]-1] = ifeed+1
            # f["pix2ind_python"] = pix2ind_python
            f["pix2ind_fortran"] = pix2ind_fortran
        os.rename(temp_outfilename, outfilename)