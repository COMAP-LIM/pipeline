"""
classes:
1. l2gen_runner     - responsible for reading param files and runlist, and starting l2gen runs in parallel.
2. l2gen_runner     - responsible for applying a series of filters to a single scan.
3. l2gen_filters    - A class taking a l2file and passing it through a filter.
4. l2gen_l1file     - Class containing l1file equivalent content, and methods for reading l1files.
5. l2gen_l2file     - Class containing l2file equivalent content, and methods for writing l2files.

    * l2gen reads the param file and runlist.
    * l2gen passes the information to the l2gen_l1file class, which reads 

Example usage:
mpirun -n 10 -machinefile machinefile.txt python3 -W ignore -u l2gen.py
mpirun -n 120 python3 -W ignore -u l2gen.py
python3 -W ignore l2gen.py
To get Numpy to respect number of threads (per MPI thread):
export OMP_NUM_THREADS=20; export OPENBLAS_NUM_THREADS=20; export MKL_NUM_THREADS=20; python3 -u -W ignore l2gen.py
"""
import time
import numpy as np
from os.path import join
import h5py
import os
from mpi4py import MPI
from l2gen_l2class import level2_file
from l2gen_filters import Tsys_calc, Normalize_Gain, Decimation, Pointing_Template_Subtraction, Masking, Polynomial_filter, Frequency_filter, PCA_filter, PCA_feed_filter, Calibration

L1_PATH = "/mn/stornext/d22/cmbco/comap/protodir/level1"

class l2gen_runner:
    def __init__(self, filter_list, omp_num_threads=2):
        # os.system(f"export OMP_NUM_THREADS={omp_num_threads}")
        os.environ["OMP_NUM_THREADS"] = f"{omp_num_threads}"
        os.environ["OPENBLAS_NUM_THREADS"] = f"{omp_num_threads}"
        os.environ["MKL_NUM_THREADS"] = f"{omp_num_threads}"
        self.filter_list = filter_list
        self.omp_num_threads = omp_num_threads
        self.read_params()

    def run(self):
        comm = MPI.COMM_WORLD
        Nranks = comm.Get_size()
        rank = comm.Get_rank()
        print(f"[{rank}] >>> Spawning rank {rank} of {Nranks}.")

        self.runlist = self.read_runlist()
        Nscans = len(self.runlist)
        for i_scan in range(Nscans):
            if i_scan%Nranks == rank:
                print(f"[{rank}] >>> Starting scan {self.runlist[i_scan][0]} ({i_scan+1}/{Nscans})..."); t0 = time.time()
                l2 = l2gen(self.runlist[i_scan], self.filter_list, self.params, omp_num_threads=self.omp_num_threads)
                l2.run()
                print(f"[{rank}] >>> Fishinsed scan {self.runlist[i_scan][0]} ({i_scan+1:}/{Nscans}) in {(time.time()-t0)/60.0:.1f} minutes. Acceptrate: {np.mean(l2.l2file.acceptrate)*100:.1f}%")

    def read_params(self):
        from l2gen_argparser import parser
        params = parser.parse_args()
        if not params.runlist:
            raise ValueError("A runlist must be specified in parameter file or terminal.")
        self.params = params

    def read_runlist(self):
        with open(self.params.runlist) as my_file:
            lines = [line.split() for line in my_file]
        i = 0
        runlist = []
        n_fields = int(lines[i][0])
        i = i + 1
        for i_field in range(n_fields):
            obsids = []
            runlist = []
            n_scans_tot = 0
            fieldname = lines[i][0]
            n_obsids = int(lines[i][1])
            i = i + 1
            for j in range(n_obsids):
                obsid = "0" + lines[i][0]
                obsids.append(obsid)
                n_scans = int(lines[i][3])
                l1_filename = lines[i][-1]
                l1_filename = l1_filename.strip("/")  # The leading "/" will stop os.path.join from joining the filenames.
                l1_filename = os.path.join(self.params.level1_dir, l1_filename)
                for k in range(n_scans):
                    scan = "0" + lines[i+k+1][0]  # Runlist obsid lacking a leading 0?
                    mjd_start = float(lines[i+k+1][1])
                    mjd_stop = float(lines[i+k+1][2])
                    scantype = int(float(lines[i+k+1][3]))
                    if scantype != 8192:
                        runlist.append([scan, mjd_start, mjd_stop, scantype, fieldname, l1_filename])
                        n_scans_tot += 1
                i = i + n_scans + 1 
        return runlist



class l2gen:
    def __init__(self, scan_info, filter_list, params, omp_num_threads):
        self.l2file = level2_file(*scan_info, filter_list, params)
        self.filter_list = filter_list
        self.omp_num_threads = omp_num_threads
        self.params = params
        self.verbose = self.params.verbose
        self.filter_names = [filter.name for filter in filter_list]
        for i, filter in enumerate(self.filter_list):  # For all filters, check if all dependencies are run first.
            for dependency in filter.depends_upon:
                if not dependency in self.filter_names[:i]:
                    raise RuntimeError(f"Filter '{filter.name}' depends upon filter '{dependency}'.")

    def run(self):
        comm = MPI.COMM_WORLD
        Nranks = comm.Get_size()
        rank = comm.Get_rank()

        if self.verbose:
            print(f"[{rank}] Reading level1 data...")
        t0 = time.time()
        self.l2file.load_level1_data()
        if self.verbose:
            print(f"[{rank}] Finished l1 file read in {time.time()-t0:.1f} s.")

        if self.params.write_inter_files:
            if self.verbose:
                print(f"[{rank}] Writing pre-filtered data to file...")
            t0 = time.time()
            self.l2file.write_level2_data(name_extension="_0")
        for i in range(len(self.filter_list)):
            filter = self.filter_list[i](self.params, omp_num_threads=self.omp_num_threads)
            if self.verbose:
                print(f"[{rank}] [{filter.name}] Starting {filter.name_long}...")
                t0 = time.time()
            filter.run(self.l2file)
            # bad_nans = 0
            # try:
            #     bad_nans = np.sum(~np.isfinite(self.l2file.tod[self.l2file.freqmask]))
            #     print("bad nans:", bad_nans)
            # except:
            #     pass
            # if bad_nans > 0:
            #     raise ValueError(f"NaNs in TOD not masked by freqmask after {filter.name_long}.")
            if self.verbose:
                print(f"[{rank}] [{filter.name}] Finished {filter.name_long} in {time.time()-t0:.1f} s.")
            if self.params.write_inter_files:
                if self.verbose:
                    print(f"[{rank}] [{filter.name}] Writing result of {filter.name_long} to file...")
                t0 = time.time()
                self.l2file.write_level2_data(name_extension=f"_{str(i+1)}_{filter.name}")
            del(filter)
        if self.verbose:
            print(f"[{rank}] Writing level2 file...")
            t0 = time.time()
        self.l2file.write_level2_data()
        if self.verbose:
            print(f"[{rank}] Finished l2 file write in {time.time()-t0:.1f} s.")


if __name__ == "__main__":
    filters = [ Tsys_calc,
                Normalize_Gain,
                Pointing_Template_Subtraction,
                Masking,
                # Polynomial_filter,
                Frequency_filter,
                PCA_filter,
                PCA_feed_filter,
                Calibration,
                Decimation]
    l2r = l2gen_runner(filters, omp_num_threads=2)
    l2r.run()
