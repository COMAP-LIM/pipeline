import numpy as np
import h5py
import os

class level2_file:
    def __init__(self, scanid, mjd_start, mjd_stop, scantype, fieldname, l1_filename, params):
        self.level2_dir = params.level2_dir
        self.scanid = scanid
        self.obsid = scanid[:-2]
        self.mjd_start = mjd_start
        self.mjd_stop = mjd_stop
        self.fieldname = fieldname
        self.l1_filename = l1_filename
        self.l2_filename = fieldname + "_" + scanid
        if scantype&2**4:
            self.scantype = "circ"
        elif scantype&2**5:
            self.scantype = "ces"
        elif scantype&2**15:
            self.scantype = "liss"

    def load_level1_data(self):
        with h5py.File(self.l1_filename, "r") as f:
            self.tod_times = f["/spectrometer/MJD"][()]
            self.scan_start_idx = np.argmin(np.abs(self.mjd_start - self.tod_times))
            self.scan_stop_idx = np.argmin(np.abs(self.mjd_stop - self.tod_times)) + 1  # stop_idx is the first index NOT in the scan.
            self.tod_times = self.tod_times[self.scan_start_idx:self.scan_stop_idx]
            self.tod_times_seconds = (self.tod_times-self.tod_times[0])*24*60*60
            self.array_features = f["/hk/array/frame/features"][()]
            self.array_time     = f["/hk/array/frame/utc"][()]
            self.az             = f["/spectrometer/pixel_pointing/pixel_az"][:,self.scan_start_idx:self.scan_stop_idx]
            self.el             = f["/spectrometer/pixel_pointing/pixel_el"][:,self.scan_start_idx:self.scan_stop_idx]
            self.tod            = f["/spectrometer/tod"][:,:,:,self.scan_start_idx:self.scan_stop_idx]
            self.feeds          = f["/spectrometer/feeds"][()]
            self.tofile_dict = {}  # Dict for adding custom data, which will be written to level2 file.
            self.Nfeeds = self.tod.shape[0]
            self.Nsb = self.tod.shape[1]
            self.Nfreqs = self.tod.shape[2]
            self.Ntod = self.tod.shape[3]
    
            self.freqmask = np.ones((self.Nfeeds, self.Nsb, self.Nfreqs), dtype=bool)
            self.freqmask_reason = np.zeros_like(self.freqmask, dtype=int)
            self.freqmask_reason_string = []
            self.freqmask_counter = 0

            self.freqmask[(~np.isfinite(self.tod)).any(axis=-1)] = False
            self.freqmask_reason[(~np.isfinite(self.tod)).any(axis=-1)] += 2**self.freqmask_counter; self.freqmask_counter += 1
            self.freqmask_reason_string.append("NaN or inf in TOD")
            self.freqmask[:,:,:2] = False
            self.freqmask[:,:,512] = False
            self.freqmask_reason[:,:,:2] += 2**self.freqmask_counter
            self.freqmask_reason[:,:,512] += 2**self.freqmask_counter; self.freqmask_counter += 1
            self.freqmask_reason_string.append("Marked channels")

            self.tod[~self.freqmask] = np.nan


    def write_level2_data(self, name_extension=""):
        outpath = os.path.join(self.level2_dir, self.fieldname)
        if not os.path.exists(outpath):
            os.mkdir(outpath)
        outfilename = os.path.join(outpath, self.l2_filename + name_extension + ".h5")
        with h5py.File(outfilename, "w") as f:
            f["tod"] = self.tod
            f["freqmask"] = self.freqmask
            f["freqmask_reason"] = self.freqmask_reason
            f["freqmask_reason_string"] = np.array(self.freqmask_reason_string, dtype="S100")
            for key in self.tofile_dict:  # Writing custom data (usually from the filters) to file.
                f[key] = self.tofile_dict[key]



if __name__ == "__main__":
    l2 = level2_file()
    l2.load_level1_data("/mn/stornext/d22/cmbco/comap/protodir/level1/2019-05/comp_comap-0005990-2019-05-18-143347.hd5", 58621.6160891319, 58621.6186759375)
    print(l2.tod.shape)