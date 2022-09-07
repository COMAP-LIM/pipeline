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
import h5py
import logging
import datetime
import os
import shutil
from os.path import join
from mpi4py import MPI
from l2gen_l2class import level2_file
from l2gen_filters import Tsys_calc, Normalize_Gain, Decimation, Pointing_Template_Subtraction, Masking, Polynomial_filter, Frequency_filter, PCA_filter, PCA_feed_filter, Calibration

L1_PATH = "/mn/stornext/d22/cmbco/comap/protodir/level1"

class l2gen_runner:
    def __init__(self, filter_list, omp_num_threads=2):
        self.comm = MPI.COMM_WORLD
        self.Nranks = self.comm.Get_size()
        self.rank = self.comm.Get_rank()

        # os.system(f"export OMP_NUM_THREADS={omp_num_threads}")
        os.environ["OMP_NUM_THREADS"] = f"{omp_num_threads}"
        os.environ["OPENBLAS_NUM_THREADS"] = f"{omp_num_threads}"
        os.environ["MKL_NUM_THREADS"] = f"{omp_num_threads}"
        self.filter_list = filter_list
        self.omp_num_threads = omp_num_threads
        self.read_params()
        self.configure_logging()
        self.runlist = self.read_runlist()


    def run(self):
        self.comm.Barrier()
        print(f"[{self.rank}] >>> Spawning rank {self.rank} of {self.Nranks}.")
        logging.info(f"[{self.rank}] >>> Spawning rank {self.rank} of {self.Nranks}.")

        Nscans = len(self.runlist)
        for i_scan in range(Nscans):
            if i_scan%self.Nranks == self.rank:
                print(f"[{self.rank}] >>> Starting scan {self.runlist[i_scan][0]} ({i_scan+1}/{Nscans})...");
                logging.info(f"[{self.rank}] >>> Starting scan {self.runlist[i_scan][0]} ({i_scan+1}/{Nscans})..."); t0 = time.time(); pt0 = time.process_time()
                l2 = l2gen(self.runlist[i_scan], self.filter_list, self.params, omp_num_threads=self.omp_num_threads)
                l2.run()
                dt = time.time() - t0; pdt = time.process_time() - pt0
                print(f"[{self.rank}] >>> Fishinsed scan {self.runlist[i_scan][0]} ({i_scan+1:}/{Nscans}) in {dt/60.0:.1f} minutes. Acceptrate: {np.mean(l2.l2file.acceptrate)*100:.1f}%")
                logging.info(f"[{self.rank}] >>> Fishinsed scan {self.runlist[i_scan][0]} ({i_scan+1:}/{Nscans}) in {dt/60.0:.1f} minutes. Acceptrate: {np.mean(l2.l2file.acceptrate)*100:.1f}%")


    def configure_logging(self):
        runID = 0
        if self.rank == 0:
            runID = str(datetime.datetime.now())[2:].replace(" ", "-").replace(":", "-")
        runID = self.comm.bcast(runID, root=0)
        
        self.params.runID = int(runID.replace("-", "").replace(".", ""))
        logfilepath = os.path.join(self.params.log_dir, f"l2gen-{runID}.log")
        logging.basicConfig(filename=logfilepath, filemode="a", format="%(levelname)s %(asctime)s - %(message)s", level=logging.DEBUG)
        if self.rank == 0:
            logging.info(f"Log initialized for runID {runID}")
            print(f"Log initialized for runID {runID}")


    def read_params(self):
        from l2gen_argparser import parser
        params = parser.parse_args()
        if not params.runlist:
            raise ValueError("A runlist must be specified in parameter file or terminal.")
        self.params = params


    def read_runlist(self):
        # Create list of already processed scanids.
        existing_scans = []
        for dir in os.listdir(self.params.level2_dir):
            if os.path.isdir(os.path.join(self.params.level2_dir, dir)):
                for file in os.listdir(os.path.join(self.params.level2_dir, dir)):
                    if file[-3:] == ".h5" or file[-4:] == ".hd5":
                        existing_scans.append(int(file.split(".")[0].split("_")[1]))
        if len(existing_scans) > 0:
            if self.rank == 0:
                print(f"Ignoring {len(existing_scans)} already processed scans.")
                logging.info(f"Ignoring {len(existing_scans)} already processed scans.")

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
                    if not int(scan) in existing_scans:
                        if scantype != 8192:
                            runlist.append([scan, mjd_start, mjd_stop, scantype, fieldname, l1_filename])
                            n_scans_tot += 1
                i = i + n_scans + 1 
        return runlist



class l2gen:
    def __init__(self, scan_info, filter_list, params, omp_num_threads):
        self.comm = MPI.COMM_WORLD
        self.Nranks = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
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
        logging.debug(f"[{self.rank}] Reading level1 data...")
        t0 = time.time(); pt0 = time.process_time()
        self.l2file.load_level1_data()
        logging.debug(f"[{self.rank}] Finished l1 file read in {time.time()-t0:.1f} s. Process time: {time.process_time()-pt0:.1f} s.")

        if self.params.write_inter_files:
            logging.debug(f"[{self.rank}] Writing pre-filtered data to file...")
            self.l2file.write_level2_data(name_extension="_0")
        for i in range(len(self.filter_list)):
            filter = self.filter_list[i](self.params, omp_num_threads=self.omp_num_threads)
            logging.debug(f"[{self.rank}] [{filter.name}] Starting {filter.name_long}...")
            t0 = time.time(); pt0 = time.process_time()
            filter.run(self.l2file)
            # bad_nans = 0
            # try:
            #     bad_nans = np.sum(~np.isfinite(self.l2file.tod[self.l2file.freqmask]))
            #     print("bad nans:", bad_nans)
            # except:
            #     pass
            # if bad_nans > 0:
            #     raise ValueError(f"NaNs in TOD not masked by freqmask after {filter.name_long}.")
            logging.debug(f"[{self.rank}] [{filter.name}] Finished {filter.name_long} in {time.time()-t0:.1f} s. Process time: {time.process_time()-pt0:.1f} s.")
            if self.params.write_inter_files:
                logging.debug(f"[{self.rank}] [{filter.name}] Writing result of {filter.name_long} to file...")
                self.l2file.write_level2_data(name_extension=f"_{str(i+1)}_{filter.name}")
            del(filter)
        logging.debug(f"[{self.rank}] Writing level2 file...")
        self.l2file.write_level2_data()
        logging.debug(f"[{self.rank}] Finished l2 file write.")



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
    if "OMP_NUM_THREADS" in os.environ:
        omp_num_threads = os.environ["OMP_NUM_THREADS"]
    else:
        omp_num_threads = 1
    l2r = l2gen_runner(filters, omp_num_threads=omp_num_threads)
    l2r.run()
