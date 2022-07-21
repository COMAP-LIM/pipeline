import numpy as np
import h5py
import os

class level2_file:
    def __init__(self, scanid, mjd_start, mjd_stop, scantype, fieldname, l1_filename):
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
            try:
                self.field      = f["/comap/"].attrs["source"]
            except:
                self.field      = "unknown"
            self.Nfeeds = self.tod.shape[0]
            self.Nsb = self.tod.shape[1]
            self.Nfreqs = self.tod.shape[2]
            self.Ntod = self.tod.shape[3]
    
            self.freqmask = np.ones((self.Nfeeds, self.Nsb, self.Nfreqs), dtype=bool)
            self.freqmask_reason = np.zeros_like(self.freqmask, dtype=int)
            self.freqmask[(~np.isfinite(self.tod)).any(axis=-1)] = False
            self.freqmask_reason[(~np.isfinite(self.tod)).any(axis=-1)] += 2**1
            self.freqmask[:,:,:2] = False
            self.freqmask[:,:,512] = False
            self.freqmask_reason[:,:,:2] += 2**2
            self.freqmask_reason[:,:,512] += 2**2
            with h5py.File("/mn/stornext/d22/cmbco/comap/protodir/auxiliary/aliasing_suppression.h5", "r") as f:
                AB_mask = f["/AB_mask"][()]
                leak_mask = f["/leak_mask"][()]
            self.freqmask[AB_mask[self.feeds-1] < 15] = False
            self.freqmask[leak_mask[self.feeds-1] < 15] = False
            self.freqmask_reason[AB_mask[self.feeds-1] < 15] += 2**3
            self.freqmask_reason[AB_mask[self.feeds-1] < 15] += 2**4

            self.tod[~self.freqmask] = np.nan


    def write_level2_data(self, path, name_extension=""):
        outfilename = os.path.join(path, self.field, self.l2_filename + name_extension + ".h5")
        with h5py.File(outfilename, "w") as f:
            f["tod"] = self.tod
            f["freqmask"] = self.freqmask
            f["freqmask_reason"] = self.freqmask_reason



if __name__ == "__main__":
    l2 = level2_file()
    l2.load_level1_data("/mn/stornext/d22/cmbco/comap/protodir/level1/2019-05/comp_comap-0005990-2019-05-18-143347.hd5", 58621.6160891319, 58621.6186759375)
    print(l2.tod.shape)