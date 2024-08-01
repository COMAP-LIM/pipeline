import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import curve_fit
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
        self.is_wn_sim = False # Whether this is or contains a white noise simulation
        if scantype&2**4:
            self.scantype = "circ"
        elif scantype&2**5:
            self.scantype = "ces"
        elif scantype&2**15:
            self.scantype = "liss"
        elif scantype&2**7:
            self.scantype = "stationary"
        else:
            self.scantype = "unknown"

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
            self.included_feeds_idxs = np.array([i for i in range(len(self.feeds)) if self.feeds[i] in self.params.included_feeds])
            self.feeds = self.feeds[self.included_feeds_idxs]
            self.array_features = f["/hk/array/frame/features"][()]
            self.array_time     = f["/hk/array/frame/utc"][()]
            self.az             = f["/spectrometer/pixel_pointing/pixel_az"][self.included_feeds_idxs,self.scan_start_idx:self.scan_stop_idx]
            self.el             = f["/spectrometer/pixel_pointing/pixel_el"][self.included_feeds_idxs,self.scan_start_idx:self.scan_stop_idx]
            self.ra             = f["/spectrometer/pixel_pointing/pixel_ra"][self.included_feeds_idxs,self.scan_start_idx:self.scan_stop_idx]
            self.dec            = f["/spectrometer/pixel_pointing/pixel_dec"][self.included_feeds_idxs,self.scan_start_idx:self.scan_stop_idx]
            self.tod            = f["/spectrometer/tod"][self.included_feeds_idxs,:,:,self.scan_start_idx:self.scan_stop_idx]
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

            ### Frequency masking setup ###
            self.freqmask_reason_string = ["Imported freqmask",
                                            "NaN or inf in TOD",
                                            "Marked channels",
                                            "Can't read or find Tsys calib file.",
                                            "Both calibs marked as unsuccessful.",
                                            "Tsys NaN or inf",
                                            "Tsys < min_tsys",
                                            "Tsys > running median max",
                                            "Aliasing suppression (AB_mask)",
                                            "Aliasing suppression (leak_mask)",
                                            "Box 32 chi2",  "Box 32 prod",  "Box 32 sum",
                                            "Box 128 chi2", "Box 128 prod", "Box 128 sum",
                                            "Box 512 chi2", "Box 512 prod", "Box 512 sum",
                                            "Stripe 32 chi2",   "Stripe 32 prod",
                                            "Stripe 128 chi2",  "Stripe 128 prod",
                                            "Stripe 1024 chi2", "Stripe 1024 prod",
                                            "Radiometer std",
                                            "Max corr",
                                            "Lonely unmasked channel",
                                            f"Less than 10% of band unmasked",
                                            ]
            self.freqmask_reason_num_dict = {}
            for i, mask_reason in enumerate(self.freqmask_reason_string):
                self.freqmask_reason_num_dict[mask_reason] = i

            # self.freqmask_string_numbering = {
            #     "NaN or inf in TOD" : 1,
            #     "Marked channels" : 2,
            #     "Tsys NaN or inf" : 3,

            # }

            ### Preliminary Masking ###
            self.mask_temporal = np.ones((self.Nfeeds, self.Ntod), dtype=bool)
            self.freqmask = np.ones((self.Nfeeds, self.Nsb, self.Nfreqs), dtype=bool)
            self.freqmask_reason = np.zeros_like(self.freqmask, dtype=int)
            self.freqmask_counter = 0
            self.n_nans = np.sum(~np.isfinite(self.tod), axis=-1)
            self.freqmask[self.n_nans > 0] = False
            self.freqmask_reason[self.n_nans > 0] += 2**self.freqmask_reason_num_dict["NaN or inf in TOD"]
            self.freqmask[:,:,:2] = False
            self.freqmask[:,:,512] = False
            self.freqmask_reason[:,:,:2] += 2**self.freqmask_counter
            self.freqmask_reason[:,:,512] += 2**self.freqmask_counter
            if self.params.sbA_num_masked_channels != 0:
                self.freqmask[:,(0,1),-self.params.sbA_num_masked_channels:] = False
                self.freqmask_reason[:,(0,1),-self.params.sbA_num_masked_channels:] += 2**self.freqmask_reason_num_dict["Marked channels"]
            for i in range(len(self.feeds)):  # Channel ranges we've found to misbehave consistently.
                feed = self.feeds[i]
                if feed == 4 or feed == 10:   # These should be changed (some are backwards, look at git blame for correct version), but it seemed to fuck stuff up...
                    self.freqmask[i,2,952:] = False
                    self.freqmask[i,3,:72] = False
                    self.freqmask_reason[i,2,952:] += 2**self.freqmask_reason_num_dict["Marked channels"]
                    self.freqmask_reason[i,3,:72] += 2**self.freqmask_reason_num_dict["Marked channels"]
                # elif feed == 16:
                    # self.freqmask[i,0,550:900] = False
                # elif feed == 17:
                #     self.freqmask[i,1,276:526] = False
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
                
            self.find_spikes()

    def find_spikes(self):
        # define gaussian function to fit
        def gauss(x, a, b, c, d):
            return a * np.exp(-(x - b)**2 / (2 * c**2)) + d
        time = self.tod_times_seconds  # seconds
        n_fit = 70  # half the length of region used to fit gaussian
        window_size = 151  # window size for savgol_filter
        distance = 50  # closest allowed distance between peaks
        height = 0.01  # amplitude of spike
        

        

        dt = time[1] - time[0]

        self.n_spikes = np.zeros((self.Nfeeds, self.Nsb)).astype(int)
        self.spike_data = np.zeros((self.Nfeeds, self.Nsb, 50, 5))

        for i in range(self.Nfeeds):
            for j in range(self.Nsb):
                data = self.sb_mean[i, j]
                data = data / np.mean(data) - 1
                if np.isnan(data.mean()):
                    continue 
                smoothed_data = savgol_filter(data, window_size, 3) # smooth data
                data_detrended = data - smoothed_data # subtract smoothed data

                # detect spikes
                peaks, _ = find_peaks(data_detrended, height=height, distance=distance)  # adjust the height parameter as per your data
                # print(i, j, peaks)
                n_time = len(time)

                
                self.n_spikes[i, j] = len(peaks)

                # if self.n_spikes[i, j] > 0:
                #     plt.plot(time, data)  # plot original data

                std = data_detrended.std() #(data[1:] - data[:-1]).std() / np.sqrt(2)

                # fit gaussians and save parameters
                for k, peak in enumerate(peaks[:50]):
                    try:
                        bounds = ((0, time[peak]-0.5, 0, -1), (5000, time[peak]+0.5, 2, 1))
                        p0 = (0.1, time[peak], 0.1, 0)
                        earliest_ind = max(0, peak-n_fit)
                        latest_ind = min(n_time, peak+n_fit)
                        popt, _ = curve_fit(gauss, time[earliest_ind:latest_ind], data[earliest_ind:latest_ind], 
                            p0=p0, bounds=bounds
                        )
                        self.spike_data[i, j, k, 0] = popt[0]  # amplitude
                        self.spike_data[i, j, k, 1] = popt[1]  # position
                        self.spike_data[i, j, k, 2] = popt[2]  # standard deviation
                        self.spike_data[i, j, k, 3] = popt[3]  # offset
                        
                        n_time_fit = latest_ind - earliest_ind
                        chi2 = (np.sum(((data[earliest_ind:latest_ind] - gauss(time[earliest_ind:latest_ind], *popt))/ std) ** 2) - n_time_fit) / np.sqrt(2 * n_time_fit)
                        
                        self.spike_data[i, j, k, 4] = chi2  # chi2 goodness-of-fit
                        # plt.plot(time[earliest_ind:latest_ind], gauss(time[earliest_ind:latest_ind], *popt), 'r')  # plot fitted gaussians
                    except:  # in case the fitting fails
                        # print(f"Couldn't fit a Gaussian on the spike at position {time[peak]}, feed {self.feeds[i]}, sb {j+1}")
                        self.spike_data[i, j, k, 1] = time[peak]
                
                # if self.n_spikes[i, j] > 0:
                #     plt.show()
                
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
        
        # If processing signal only TOD sigma0 is substituted with 1s to make coadditions become arithmetic means
        # if "Replace_TOD_With_Signal" in self.params.filters:
        #     print("Overwriting weights in l2 saver with ones due to signal only TOD being processed!")

        #     self.sigma0 = 0.04 * np.ones_like(self.sigma0)
        #     self.sigma0[~self.freqmask_decimated] = np.nan
        
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
                if self.is_sim:
                    f.create_dataset("signal_simulation_tod", (self.Nfeeds, self.Nsb, self.Nfreqs, self.Ntod), chunks=(1,1,1,self.Ntod), data=self.signal_tod, compression="gzip", compression_opts=1, shuffle=True)
                        
            else:
                f["tod"] = self.tod
                if self.is_sim:
                    f["signal_simulation_tod"] = self.signal_tod

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
            f["is_wn_sim"] = self.is_wn_sim
            f["git_hash"] = self.git_hash
            f["l1_filepath"] = self.l1_filename
            f["n_spikes"] = self.n_spikes
            f["spike_data"] = self.spike_data

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
                f["hk_cryostat_pressure"]  = l1file["hk/antenna0/cryostat/pressure"][self.scan_start_idx_hk:self.scan_stop_idx_hk]
                f["hk_cryostat_temp1"]  = l1file["hk/antenna0/cryostat/temp1"][self.scan_start_idx_hk:self.scan_stop_idx_hk]
                f["hk_cryostat_temp2"]  = l1file["hk/antenna0/cryostat/temp2"][self.scan_start_idx_hk:self.scan_stop_idx_hk]
                f["hk_cryostat_temp3"]  = l1file["hk/antenna0/cryostat/temp3"][self.scan_start_idx_hk:self.scan_stop_idx_hk]
                f["hk_cryostat_temp4"]  = l1file["hk/antenna0/cryostat/temp4"][self.scan_start_idx_hk:self.scan_stop_idx_hk]
                f["hk_cryostat_temp5"]  = l1file["hk/antenna0/cryostat/temp5"][self.scan_start_idx_hk:self.scan_stop_idx_hk]
                f["hk_cryostat_temp6"]  = l1file["hk/antenna0/cryostat/temp6"][self.scan_start_idx_hk:self.scan_stop_idx_hk]
            
            # pix2ind_python = np.zeros(20, dtype=int) + 999
            pix2ind_fortran = np.zeros(20, dtype=int)
            for ifeed in range(self.Nfeeds):
                # pix2ind_python[self.feeds[ifeed]-1] = ifeed
                pix2ind_fortran[self.feeds[ifeed]-1] = ifeed+1
            # f["pix2ind_python"] = pix2ind_python
            f["pix2ind_fortran"] = pix2ind_fortran
        os.rename(temp_outfilename, outfilename)

