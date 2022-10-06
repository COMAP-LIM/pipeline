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
import logging
import datetime
import os
import psutil
from mpi4py import MPI
from l2gen_l2class import level2_file
import l2gen_filters
# from l2gen_filters import Tsys_calc, Normalize_Gain, Decimation, Pointing_Template_Subtraction, Masking, Polynomial_filter, Frequency_filter, PCA_filter, PCA_feed_filter, Calibration

L1_PATH = "/mn/stornext/d22/cmbco/comap/protodir/level1"

class l2gen_runner:
    def __init__(self, omp_num_threads=2):
        self.comm = MPI.COMM_WORLD
        self.Nranks = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
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
        print(f"[{self.rank}] >>> Spawning rank {self.rank} of {self.Nranks}.")
        logging.info(f"[{self.rank}] >>> Spawning rank {self.rank} of {self.Nranks}.")

        Nscans = len(self.runlist)
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
        from l2gen_argparser import parser
        params = parser.parse_args()
        if not params.runlist:
            raise ValueError("A runlist must be specified in parameter file or terminal.")
        self.params = params
        if self.rank == 0:
            print(f"Parameter file: {params.param}")


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
        for filter_str in self.params.filters:
            filter = getattr(l2gen_filters, filter_str)
            self.filter_list.append(filter)


    def read_runlist(self):
        if self.rank == 0:
            print(f"Creating runlist in specified obsid range [{self.params.obsid_start}, {self.params.obsid_stop}]")
            print(f"Runlist file: {self.params.runlist}")
    
        # Create list of already processed scanids.
        existing_scans = []
        for dir in os.listdir(self.params.level2_dir):
            if os.path.isdir(os.path.join(self.params.level2_dir, dir)):
                for file in os.listdir(os.path.join(self.params.level2_dir, dir)):
                    if file[-3:] == ".h5" or file[-4:] == ".hd5":
                        if len(file) == 16 or len(file) == 17:  # In order to not catch the intermediate debug files.
                            existing_scans.append(int(file.split(".")[0].split("_")[1]))

        with open(self.params.runlist) as my_file:
            lines = [line.split() for line in my_file]
        i = 0
        runlist = []
        n_fields = int(lines[i][0])
        i = i + 1
        for i_field in range(n_fields):
            runlist = []
            n_scans_tot = 0
            n_scans_outside_range = 0
            n_scans_already_processed = 0
            fieldname = lines[i][0]
            n_obsids = int(lines[i][1])
            i = i + 1
            for j in range(n_obsids):
                obsid = "0" + lines[i][0]
                n_scans = int(lines[i][3])
                l1_filename = lines[i][-1]
                l1_filename = l1_filename.strip("/")  # The leading "/" will stop os.path.join from joining the filenames.
                l1_filename = os.path.join(self.params.level1_dir, l1_filename)
                for k in range(n_scans):
                    scantype = int(float(lines[i+k+1][3]))
                    if scantype != 8192:
                        n_scans_tot += 1
                        if self.params.obsid_start <= int(obsid) <= self.params.obsid_stop:
                            scanid = int(lines[i+k+1][0])
                            mjd_start = float(lines[i+k+1][1])
                            mjd_stop = float(lines[i+k+1][2])
                            if not scanid in existing_scans:
                                runlist.append([scanid, mjd_start, mjd_stop, scantype, fieldname, l1_filename])
                            else:
                                n_scans_already_processed += 1
                        else:
                            n_scans_outside_range += 1
                i = i + n_scans + 1
            if self.rank == 0:
                print(f"Field name:                 {fieldname}")
                print(f"Obsids in runlist file:     {n_obsids}")
                print(f"Scans in runlist file:      {n_scans_tot}")
                print(f"Scans included in run:      {len(runlist)}")
                print(f"Scans outside obsid range:  {n_scans_outside_range}")
                print(f"Scans already processed:    {n_scans_already_processed}")

        self.runlist = runlist



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