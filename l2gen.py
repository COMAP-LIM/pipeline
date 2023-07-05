"""
Example usage:
    mpirun --machinefile machinefile.txt python3 -W ignore -u l2gen.py
    mpirun -n 120 python3 -W ignore -u l2gen.py
    python3 -W ignore l2gen.py
    To get Numpy to respect number of threads (per MPI thread):
    export OMP_NUM_THREADS=20; export OPENBLAS_NUM_THREADS=20; export MKL_NUM_THREADS=20; python3 -u -W ignore l2gen.py
"""
import time
import numpy as np
import logging
import datetime
import os
import psutil
import random
from mpi4py import MPI
from l2gen_l2class import level2_file
import l2gen_filters
from tools.read_runlist import read_runlist
# from l2gen_filters import Tsys_calc, Normalize_Gain, Decimation, Pointing_Template_Subtraction, Masking, Polynomial_filter, Frequency_filter, PCA_filter, PCA_feed_filter, Calibration

import warnings
warnings.filterwarnings("ignore", message="divide by zero encountered in double_scalars")
warnings.filterwarnings("ignore", message="invalid value encountered in double_scalars")
warnings.filterwarnings("ignore", message="invalid value encountered in true_divide")
warnings.filterwarnings("ignore", message="divide by zero encountered in true_divide")
warnings.filterwarnings("ignore", message="Degrees of freedom <= 0 for slice.")
warnings.filterwarnings("ignore", message="Mean of empty slice")



class l2gen_runner:
    def __init__(self, omp_num_threads=2):
        self.comm = MPI.COMM_WORLD
        self.Nranks = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        self.node_name = MPI.Get_processor_name()
        if self.rank == 0:
            print("######## Initializing l2gen ########")

        self.omp_num_threads = omp_num_threads
        self.read_params()
        self.configure_logging()
        self.configure_filters()
        self.read_runlist()
        if self.rank == 0:
            print("######## Done initializing l2gen ########")


    def run(self):
        self.comm.Barrier()
        Nscans = len(self.runlist)
        if Nscans == 0:
            if self.rank == 0:
                print(f"No unprocessed scans in runlist. Exiting.")
            return
        if self.params.distributed_starting:
            time.sleep((self.rank%16)*60.0)
        time.sleep(self.rank*0.01)
        print(f"[{self.rank}] >>> Spawning rank {self.rank} of {self.Nranks} on {self.node_name}.")
        logging.info(f"[{self.rank}] >>> Spawning rank {self.rank} of {self.Nranks} on {self.node_name}.")

        for i_scan in range(Nscans):
            if i_scan%self.Nranks == self.rank:
                while psutil.virtual_memory().available/psutil.virtual_memory().total < 0.2:
                    logging.debug(f"[{self.rank}] Only {psutil.virtual_memory().available/psutil.virtual_memory().total:.1f}% available memory. Checking again in 30 seconds.")
                    time.sleep(30)
                print(f"[{self.rank}] >>> Starting scan {self.runlist[i_scan][0]} ({i_scan+1}/{Nscans})...")
                logging.info(f"[{self.rank}] >>> Starting scan {self.runlist[i_scan][0]} ({i_scan+1}/{Nscans})..."); t0 = time.time(); pt0 = time.process_time()
                l2 = l2gen(self.runlist[i_scan], self.filter_list, self.params, omp_num_threads=self.omp_num_threads)
                l2.run()
                dt = time.time() - t0; pdt = time.process_time() - pt0
                print(f"[{self.rank}] >>> Fishinsed scan {self.runlist[i_scan][0]} ({i_scan+1:}/{Nscans}) in {dt/60.0:.1f} minutes. Acceptrate: {np.mean(l2.l2file.acceptrate)*100:.1f}%")
                logging.info(f"[{self.rank}] >>> Fishinsed scan {self.runlist[i_scan][0]} ({i_scan+1:}/{Nscans}) in {dt/60.0:.1f} minutes. Acceptrate: {np.mean(l2.l2file.acceptrate)*100:.1f}%")
        

    def read_params(self):
        self.params = 0
        if self.rank == 0:
            from l2gen_argparser import parser
            self.params = parser.parse_args()
            if not self.params.runlist:
                raise ValueError("A runlist must be specified in parameter file or terminal.")
            print(f"Filters included: {self.params.filters}")
        self.params = self.comm.bcast(self.params, root=0)


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


    def configure_filters(self):
        self.filter_list = []
        if self.rank == 0:
            for filter_str in self.params.filters:
                filter = getattr(l2gen_filters, filter_str)
                self.filter_list.append(filter)
        self.filter_list = self.comm.bcast(self.filter_list, root=0)


    def read_runlist(self):
        self.runlist = []
        if self.rank == 0:
            self.runlist = read_runlist(self.params, ignore_existing=True)

        self.runlist = self.comm.bcast(self.runlist, root=0)


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
            logging.debug(f"[{self.rank}] [{filter.name}] Finished {filter.name_long} in {time.time()-t0:.1f} s. Process time: {time.process_time()-pt0:.1f} s.")
            if self.params.write_inter_files:
                logging.debug(f"[{self.rank}] [{filter.name}] Writing result of {filter.name_long} to file...")
                self.l2file.write_level2_data(name_extension=f"_{str(i+1)}_{filter.name}")
            del(filter)

        logging.debug(f"[{self.rank}] Writing level2 file...")
        self.l2file.write_level2_data()
        logging.debug(f"[{self.rank}] Finished l2 file write.")



if __name__ == "__main__":
    if "OMP_NUM_THREADS" in os.environ:
        omp_num_threads = int(os.environ["OMP_NUM_THREADS"])
    else:
        omp_num_threads = 1
    l2r = l2gen_runner(omp_num_threads=omp_num_threads)
    l2r.run()