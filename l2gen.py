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
"""
import time
import numpy as np
from os.path import join
import h5py
from mpi4py import MPI
from l2gen_l2class import level2_file
from l2gen_filters import Normalize_Gain, Decimation, Pointing_Template_Subtraction, Masking, Polynomial_filter, PCA_filter, Calibration

L1_PATH = "/mn/stornext/d22/cmbco/comap/protodir/level1"

class l2gen_runner:
    def __init__(self, runlist_path, filter_list, checkpoints=None, omp_num_threads=2):
        self.runlist_path = runlist_path
        self.filter_list = filter_list
        self.checkpoints = checkpoints
        self.omp_num_threads = omp_num_threads
        if not self.checkpoints:
            self.checkpoints = [False for i in range(len(self.filter_list))]

    def run(self):
        comm = MPI.COMM_WORLD
        Nranks = comm.Get_size()
        rank = comm.Get_rank()
        print(f"Spawning rank {rank} of {Nranks}.")

        self.runlist = self.read_runlist()
        Nscans = len(self.runlist)
        for i_scan in range(Nscans):
            if i_scan%Nranks == rank:
                print(f"Rank {rank} starting scan {i_scan}...")
                l2 = l2gen(self.runlist[i_scan], self.filter_list, self.checkpoints, omp_num_threads=self.omp_num_threads)
                l2.run()
                print(f"Rank {rank} done with scan {i_scan}.")


    def read_runlist(self):
        with open(self.runlist_path) as my_file:
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
                l1_filename = L1_PATH + l1_filename
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
    def __init__(self, scan_info, filter_list, checkpoints, omp_num_threads):
        self.l2file = level2_file(*scan_info)
        self.filter_list = filter_list
        self.omp_num_threads = omp_num_threads
        self.checkpoints = checkpoints

    def run(self):
        print("Reading level1 data...")
        t0 = time.time()
        self.l2file.load_level1_data()
        print(f"Finished l1 file read in {time.time()-t0:.1f} s.")

        if self.checkpoints[0]:
            print(f"Writing pre-filtered data to file...")
            t0 = time.time()
            self.l2file.write_level2_data("data/", name_extension="_0")
            print(f"Finished pre-filter file write in {time.time()-t0:.1f} s.")
        for i in range(len(self.filter_list)):
            filter = self.filter_list[i](omp_num_threads=self.omp_num_threads)
            print(f"Starting {filter.name_long}...")
            t0 = time.time()
            filter.run(self.l2file)
            print(f"Finished {filter.name_long} in {time.time()-t0:.1f} s.")
            if self.checkpoints[i+1]:
                print(f"Writing result of {filter.name_long} to file...")
                t0 = time.time()
                self.l2file.write_level2_data("data/", name_extension=f"_{str(i+1)}_{filter.name}")
                print(f"Finished {filter.name_long} file write in {time.time()-t0:.1f} s.")
            del(filter)
        print("Writing level2 file...")
        t0 = time.time()
        self.l2file.write_level2_data("data/")
        print(f"Finished l2 file write in {time.time()-t0:.1f} s.")


if __name__ == "__main__":
    # runlist_path = "/mn/stornext/d22/cmbco/comap/jonas/l2gen_python/runlist_test_large.txt"
    runlist_path = "/mn/stornext/d22/cmbco/comap/jonas/l2gen_python/runlist_test_small.txt"
    # runlist_path = "/mn/stornext/d22/cmbco/comap/jonas/l2gen_python/runlist_test_liss_small.txt"
    # runlist_path = "/mn/stornext/d22/cmbco/comap/nils/COMAP_general/src/sim/Parameterfiles_and_runlists/runlist_new_pca.txt"
    filters = [Normalize_Gain, Pointing_Template_Subtraction, Masking, Polynomial_filter, PCA_filter, Calibration, Decimation]
    l2r = l2gen_runner(runlist_path, filters, [True for i in range(len(filters)+1)], omp_num_threads=24)
    l2r.run()
