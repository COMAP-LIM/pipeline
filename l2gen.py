"""
l2gen.py must be run with at least 2 MPI processes, as it utilizes a master-worker approach.
Example usage:
    mpirun --machinefile machinefile.txt python3 -u -m mpi4py l2gen.py
    mpirun -n 120 python3 -u -m mpi4py l2gen.py -p /mn/stornext/d22/cmbco/comap/params/param_co7_apr22_v2.txt
    To get Numpy to respect number of threads (per MPI thread):
    OMP_NUM_THREADS=20 OPENBLAS_NUM_THREADS=20 MKL_NUM_THREADS=20 mpirun -n 10 python3 -u -m mpi4py l2gen.py
the -m mpi4py is necessary for MPI to properly handle errors. See https://mpi4py.readthedocs.io/en/stable/mpi4py.run.html and https://stackoverflow.com/questions/49868333/fail-fast-with-mpi4py.
"""
import time
import numpy as np
import logging
import datetime
import os
from mpi4py import MPI
from tqdm import tqdm, trange
import sys
import h5py
from copy import copy
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


class Printing:
    def __init__(self, params):
        self.verbose = params.verbose
        self.debug = params.debug


    def verbose_print(self, string):
        """ Dual purpose function for normal verbose prints:
            1. Writes the prints to the log.
            2. If verbose parameter is True, also print to terminal.
        """
        logging.info(string)
        if self.verbose:
            print(string)


    def debug_print(self, string):
        """ Dual purpose function for debug prints:
            1. Writes the prints to the log.
            2. If debug parameter is True, also print to terminal.
        """
        logging.debug(string)
        if self.debug:
            print(string)



class Terminal_print:
    def __init__(self, filter_list, Nscans, Nranks):
        self.Nscans = Nscans
        self.N_finished_scans = 0
        self.accumulated_acceptrate = np.zeros((19, 4))
        self.accumulated_feeds = np.zeros((19))
        self.accumulated_filter_runtime = {}
        self.accumulated_filter_runtime["l1_read"] = 0
        for filter in filter_list:
            self.accumulated_filter_runtime[filter.name] = 0
        self.accumulated_filter_runtime["l2_write"] = 0
        self.total_filter_runtime = 1e-10
        self.rank_progress = ["x"]*Nranks
        self.rank_progress[0] = "M"
        self.t0 = time.time()
        self.pt0 = time.process_time()
        self.rewrite_lines = 0


    def get_color(self, value):
        if value > 85:
            return "\033[96m"
        elif value > 70:
            return "\033[94m"
        elif value > 50:
            return "\033[93m"
        else:            
            return "\033[91m"


    def scan_done_update(self, info):
        self.N_finished_scans += 1

        self.accumulated_feeds[info["feeds"]-1] += 1
        self.accumulated_acceptrate[info["feeds"]-1] += info["acceptrate"]

        self.total_filter_runtime = 0.0
        for filter in info["filter_runtimes"].keys():
            self.accumulated_filter_runtime[filter] += info["filter_runtimes"][filter]
            self.total_filter_runtime += self.accumulated_filter_runtime[filter]


    def scan_progress_update(self, info):
        self.rank_progress[info["rank"]] = info["status"]


    def rewrite_terminal(self):
        Nfeeds = 19
        Nsb = 4
        feeds = np.arange(1, 20)

        printstring = ""
        printstring += " " + "_"*107 + "\n"
        asdf = tqdm.format_meter(self.N_finished_scans, self.Nscans, elapsed=time.time()-self.t0, ncols=105)
        printstring += f"{'______Total progress______':<105}\n"
        printstring += f"{asdf}\n"
        printstring += f"{'':<105}\n"
        printstring += f"{'______Average acceptrate by feed and sidebands______':<105s}\n"
        printstring += f"       all"
        for ifeed in range(Nfeeds):
            printstring += f"{feeds[ifeed]:5d}"
        acc = np.sum(self.accumulated_acceptrate)/np.sum(Nsb*self.accumulated_feeds)*100
        printstring += f"\nall  {self.get_color(acc)}{acc:4.0f}%\033[0m"
        for ifeed in range(Nfeeds):
            acc = np.sum(self.accumulated_acceptrate[ifeed])/(Nsb*self.accumulated_feeds[ifeed])*100
            printstring += f"{self.get_color(acc)}{acc:4.0f}%\033[0m"
        for isb in range(Nsb):
            acc = np.sum(self.accumulated_acceptrate[:,isb])/np.sum(self.accumulated_feeds)*100
            printstring += f"\n  {isb}  {self.get_color(acc)}{acc:4.0f}%\033[0m"
            for ifeed in range(Nfeeds):
                acc = np.sum(self.accumulated_acceptrate[ifeed,isb])/self.accumulated_feeds[ifeed]*100
                printstring += f"{self.get_color(acc)}{acc:4.0f}%\033[0m"
        printstring += "\n"
        printstring += f"{'':<105}\n"
        printstring += f"{'______Runtime summary______':<105s}\n"
        for filter in self.accumulated_filter_runtime.keys():
            tempstring = f"{filter:12s}: {self.accumulated_filter_runtime[filter]/60.0:8.1f}m  ({100*self.accumulated_filter_runtime[filter]/self.total_filter_runtime:4.1f} %)"
            printstring += f"{tempstring:<105s}\n"

        printstring += f"{'':<105}\n"
        printstring += f"{'______Rank report______':<105s}\n"
        rankstring = "".join(self.rank_progress)
        for i in range(100):
            if len(rankstring) > 100*i:
                printstring += f"{rankstring[i*100:(i+1)*100]:<105s}"
                printstring += "\n"
            else:
                break
        printstring += f"{'(M = Master, W = Waiting, r = reading, 0-9 = filters, w = writing)':<105s}\n"

        printstring = printstring.replace("\n", " |\n| ")
        printstring = printstring.replace("|", "", 1)
        printstring += "_"*105 + " |\n"

        for i in range(self.rewrite_lines):
            sys.stdout.write("\x1b[1A\x1b[2K")
        print(printstring)
        lines = printstring.count("\n")
        self.rewrite_lines = lines + 1



class l2gen_runner:
    def __init__(self, omp_num_threads=2):
        self.comm = MPI.COMM_WORLD
        self.Nranks = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        self.node_name = MPI.Get_processor_name()
        if self.rank == 0:
            print("\n######## Initializing l2gen ########")

        self.omp_num_threads = omp_num_threads
        self.read_params()
        self.configure_logging()
        self.configure_filters()
        self.read_runlist(self.params)
        if self.rank == 0:
            print("######## Done initializing l2gen ########\n")



    def run(self):
        self.comm.Barrier()
        Nscans = len(self.runlist)
        if Nscans == 0:
            if self.rank == 0:
                print(f"No unprocessed scans in runlist. Exiting.")
            return

        WORK_TAG = 1
        DIE_TAG = 2
        PROGRESS_TAG = 8
        DONE_TAG = 9

        ##### Master #####
        if self.rank == 0:
            if not self.params.verbose:
                term = Terminal_print(self.filter_list, Nscans, self.Nranks)
                term.rewrite_terminal()

            proc_order = np.arange(1, self.Nranks)
            np.random.shuffle(proc_order)
            self.tasks_done = 0
            self.tasks_started = 0
            for irank in range(self.Nranks-1):
                if self.params.distributed_starting:
                    wait_time = min(irank*600.0/self.Nranks, irank*15.0)  # Spawn ranks randomly over 10 minutes, or 15 seconds per rank, whichever is faster.
                else:
                    wait_time = 0
                self.comm.send([self.tasks_started, wait_time], dest=proc_order[irank], tag=WORK_TAG)
                self.tasks_started += 1
                if self.tasks_started == Nscans:  # If we have more processes than tasks, kill the rest, and break the task-assigment loop.
                    for iirank in range(irank, self.Nranks-1):
                        self.comm.send([-1,0], dest=proc_order[iirank], tag=DIE_TAG)
                    break
                time.sleep(0.01)

            while self.tasks_started < Nscans:
                status = MPI.Status()
                info = self.comm.recv(source=MPI.ANY_SOURCE, status=status)
                if status.Get_tag() == DONE_TAG:
                    if not self.params.verbose:
                        term.scan_done_update(info)
                        term.rewrite_terminal()
                    self.tasks_done += 1
                    workerID = status.Get_source()
                    wait_time = 0
                    self.comm.send([self.tasks_started, wait_time], dest=workerID, tag=WORK_TAG)
                    self.tasks_started += 1
                    time.sleep(0.01)
                elif status.Get_tag() == PROGRESS_TAG:
                    if not self.params.verbose:
                        term.scan_progress_update(info)
                        term.rewrite_terminal()
                else:
                    raise ValueError("Unknown tag recieved.")

            while self.tasks_done < Nscans:
                status = MPI.Status()
                info = self.comm.recv(source=MPI.ANY_SOURCE, status=status)
                if status.Get_tag() == DONE_TAG:
                    if not self.params.verbose:
                        term.scan_done_update(info)
                        term.rewrite_terminal()
                    self.tasks_done += 1
                    workerID = status.Get_source()
                    self.comm.send([-1,0], dest=workerID, tag=DIE_TAG)
                    time.sleep(0.01)
                elif status.Get_tag() == PROGRESS_TAG:
                    if not self.params.verbose:
                        term.scan_progress_update(info)
                        term.rewrite_terminal()
                else:
                    raise ValueError("Unknown tag recieved.")

        ##### Workers #####
        else:
            while True:
                status = MPI.Status()
                iscan, wait_time = self.comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
                if status.Get_tag() == DIE_TAG:
                    break
                if wait_time > 0:
                    logging.info(f"[{self.rank}] >>> Asked to wait for {wait_time} seconds before starting (memory load balancing).")
                    info = {}
                    info["rank"] = self.rank
                    info["status"] = "W"
                    self.comm.send(info, dest=0, tag=8)
                    time.sleep(wait_time)
                # print(f"[{self.rank}] >>> Starting scan {self.runlist[iscan][0]} ({iscan+1}/{Nscans})...")
                logging.info(f"[{self.rank}] >>> Starting scan {self.runlist[iscan][0]} ({iscan+1}/{Nscans})..."); t0 = time.time(); pt0 = time.process_time()
                if self.params.verbose:
                    print(f"[{self.rank}] >>> Starting scan {self.runlist[iscan][0]} ({iscan+1}/{Nscans})...")
                l2 = l2gen(self.runlist[iscan], self.filter_list, self.params, omp_num_threads=self.omp_num_threads)
                l2.run()
                dt = time.time() - t0; pdt = time.process_time() - pt0
                # print(f"[{self.rank}] >>> Fishinsed scan {self.runlist[iscan][0]} ({iscan+1:}/{Nscans}) in {dt/60.0:.1f} minutes. Acceptrate: {np.mean(l2.l2file.acceptrate)*100:.1f}%")
                logging.info(f"[{self.rank}] >>> Fishinsed scan {self.runlist[iscan][0]} ({iscan+1:}/{Nscans}) in {dt/60.0:.1f} minutes. Acceptrate: {np.mean(l2.l2file.acceptrate)*100:.1f}%")
                if self.params.verbose:
                    print(f"[{self.rank}] >>> Fishinsed scan {self.runlist[iscan][0]} ({iscan+1:}/{Nscans}) in {dt/60.0:.1f} minutes. Acceptrate: {np.mean(l2.l2file.acceptrate)*100:.1f}%")

                return_dict = {}
                return_dict["filter_runtimes"] = l2.filter_runtimes
                return_dict["acceptrate"] = l2.l2file.acceptrate
                return_dict["feeds"] = l2.l2file.feeds
                self.comm.send(return_dict, dest=0, tag=DONE_TAG)


    def clean_up(self):
        # Function called when all scans are finished processing. Creates a database and ensures all files were successfully created.
        if self.rank == 0:
            print(f"Finished all scans. Starting cleanup...")
        params_temp = copy(self.params)
        params_temp.obsid_start = 0
        params_temp.obsid_stop = 99999999
        self.read_runlist(params_temp, ignore_existing=False, only_existing=True)  # Creating runlist of all existing scans.

        if self.rank == 0:
            print(f"Checking that all l2 files exist...")
            for iscan in trange(len(self.runlist)):
                scanid, mjd_start, mjd_stop, scantype, fieldname, l1_filename, l2_filename = self.runlist[iscan]
                corrupt_files = []
                missing_files = []
                try:
                    f = h5py.File(l2_filename, "r")
                except Exception as error:
                    if error is OSError:
                        corrupt_files.append(l2_filename)
                    elif error is FileNotFoundError:
                        missing_files.append(l2_filename)
                    else:
                        print(f"Unknown error {error} encountered when trying to read hdf5 file!")
            if len(corrupt_files) + len(missing_files) > 0:
                print(f"Found corrupt and/or missing files. Exiting.")
                print("Corrupt files:", corrupt_files)
                print("Missing files:", missing_files)
                return
            else:
                print(f"Found no missing or corrupt files.")

            if self.params.create_l2_summary_database:
                ### L2 summary database. Although all this information is available in the l2 files, it is slow to collect the invidial data when analysing it. ###
                print(f"Creating summary l2 database.")
                for field in self.params.fields:
                    runlist_subset = [item for item in self.runlist if item[4] == field]
                    Nscans = len(runlist_subset)
                    if Nscans == 0:
                        continue
                    database_dict = {
                        "scanid" : np.zeros(Nscans, dtype=np.uint32),
                        "mjd_start" : np.zeros(Nscans) + np.nan,
                        "scan_samples" : np.zeros(Nscans, dtype=np.uint32),
                        "acceptrate" : np.zeros((Nscans, 19, 4)) + np.nan,
                        "freqmask_full" : np.zeros((Nscans, 19, 4, 1024), dtype=bool),
                        "freqmask_reason" : np.zeros((Nscans, 19, 4, 1024), dtype=np.uint64),
                        "n_nan" : np.zeros((Nscans, 19, 4, 1024), dtype=np.uint32),
                        "sigma0" : np.zeros((Nscans, 19, 4, self.params.decimation_freqs)) + np.nan,
                        "chi2" : np.zeros((Nscans, 19, 4, self.params.decimation_freqs)) + np.nan,
                        "Tsys" : np.zeros((Nscans, 19, 4, 1024)) + np.nan,
                        "Gain" : np.zeros((Nscans, 19, 4, 1024)) + np.nan,
                        "n_pca_comp" : np.zeros((Nscans)) + np.nan,
                        "n_pca_feed_comp" : np.zeros((Nscans, 19)) + np.nan,
                        "point_cel_feed1" : np.zeros((Nscans, 30000, 2), dtype=np.float32) + np.nan,
                        "point_tel_feed1" : np.zeros((Nscans, 30000, 2), dtype=np.float32) + np.nan,
                    }
                    for iscan in trange(Nscans):
                        scanid, mjd_start, mjd_stop, scantype, fieldname, l1_filename, l2_filename = runlist_subset[iscan]
                        with h5py.File(l2_filename, "r") as f:
                            Ntod = f["time"].shape[-1]
                            if Ntod <= 30000:
                                database_dict["point_cel_feed1"][iscan,:Ntod] = f["point_cel"][0,:]
                                database_dict["point_tel_feed1"][iscan,:Ntod] = f["point_tel"][0,:]
                            feeds = f["feeds"][()]
                            database_dict["scan_samples"][iscan] = Ntod
                            database_dict["scanid"][iscan] = f["scanid"][()]
                            database_dict["mjd_start"][iscan] = f["mjd_start"][()]
                            database_dict["acceptrate"][iscan,feeds-1] = f["acceptrate"][()]
                            database_dict["freqmask_full"][iscan,feeds-1] = f["freqmask_full"][()]
                            database_dict["freqmask_reason"][iscan,feeds-1] = f["freqmask_reason"][()]
                            database_dict["n_nan"][iscan,feeds-1] = f["n_nan"][()]
                            database_dict["sigma0"][iscan,feeds-1] = f["sigma0"][()]
                            database_dict["chi2"][iscan,feeds-1] = f["chi2"][()]
                            database_dict["Tsys"][iscan,feeds-1] = f["Tsys"][()]
                            database_dict["Gain"][iscan,feeds-1] = f["Gain"][()]
                            try:  # These might not exist, depending on what filters are included in the run.
                                database_dict["n_pca_comp"][iscan] = f["n_pca_comp"][()]
                                database_dict["n_pca_feed_comp"][iscan,feeds-1] = f["n_pca_feed_comp"][()]
                            except:
                                pass

                    with h5py.File(os.path.join(self.params.level2_dir, field + "_database.hd5"), "w") as f:
                        for key in database_dict.keys():
                            f[key] = database_dict[key]

            print("L2gen finished!")




    def read_params(self):
        self.params = 0
        if self.rank == 0:
            from l2gen_argparser import parser
            self.params = parser.parse_args()
            if not self.params.runlist:
                raise ValueError("A runlist must be specified in parameter file or terminal.")
            print(f"Filters included:")
            for i in range(len(self.params.filters)):
                print(f"{i}. {self.params.filters[i]}")
        self.params = self.comm.bcast(self.params, root=0)



    def configure_logging(self):
        runID = 0
        if self.rank == 0:
            runID = str(datetime.datetime.now())[2:].replace(" ", "-").replace(":", "-")
        runID = self.comm.bcast(runID, root=0)
        
        self.params.runID = int(runID.replace("-", "").replace(".", ""))
        if not sys.path.exists(os.path.join(self.params.log_dir)):
            os.makedirs(self.params.log_dir)
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



    def read_runlist(self, params, ignore_existing=True, only_existing=False):
        self.runlist = []
        if self.rank == 0:
            self.runlist = read_runlist(params, ignore_existing=ignore_existing, only_existing=only_existing)

        self.runlist = self.comm.bcast(self.runlist, root=0)




class l2gen:
    def __init__(self, scan_info, filter_list, params, omp_num_threads):
        self.printer = Printing(params)
        self.comm = MPI.COMM_WORLD
        self.Nranks = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        self.l2file = level2_file(*scan_info, filter_list, params)
        self.filter_list = filter_list
        self.omp_num_threads = omp_num_threads
        self.params = params
        self.verbose = self.params.verbose
        self.filter_names = [filter.name for filter in filter_list]
        self.filter_runtimes = {}
        self.filter_processtimes = {}


    def run(self):
        self.printer.debug_print(f"[{self.rank}] Reading level1 data...")
        info = {}
        info["rank"] = self.rank
        info["status"] = "r"
        self.comm.send(info, dest=0, tag=8)
        t0 = time.time(); pt0 = time.process_time()
        self.l2file.load_level1_data()
        self.l2file.printer = self.printer
        t1 = time.time(); pt1 = time.process_time()
        self.filter_runtimes["l1_read"] = t1 - t0
        self.filter_processtimes["l1_read"] = pt1 - pt0
        self.printer.debug_print(f"[{self.rank}] Finished l1 file read in {t1-t0:.1f} s. Process time: {pt1-pt0:.1f} s.")

        if self.params.write_inter_files:
            self.printer.debug_print(f"[{self.rank}] Writing pre-filtered data to file...")
            self.l2file.write_level2_data(name_extension="_0")
        for i in range(len(self.filter_list)):
            filter = self.filter_list[i](self.params, omp_num_threads=self.omp_num_threads)
            if not self.params.verbose:
                info = {}
                info["rank"] = self.rank
                info["status"] = str(i)[0]
                self.comm.send(info, dest=0, tag=8)
            self.printer.debug_print(f"[{self.rank}] [{filter.name}] Starting {filter.name_long}...")
            t0 = time.time(); pt0 = time.process_time()
            filter.run(self.l2file)
            t1 = time.time(); pt1 = time.process_time()
            self.printer.debug_print(f"[{self.rank}] [{filter.name}] Finished {filter.name_long} in {t1-t0:.1f} s. Process time: {pt1-pt0:.1f} s.")
            if not self.params.verbose:
                self.filter_runtimes[self.filter_names[i]] = t1 - t0
                self.filter_processtimes[self.filter_names[i]] = pt1 - pt0
            if self.params.write_inter_files:
                self.printer.debug_print(f"[{self.rank}] [{filter.name}] Writing result of {filter.name_long} to file...")
                self.l2file.write_level2_data(name_extension=f"_{str(i+1)}_{filter.name}")
            del(filter)


        self.printer.debug_print(f"[{self.rank}] Writing level2 file...")
        info = {}
        info["rank"] = self.rank
        info["status"] = "w"
        self.comm.send(info, dest=0, tag=8)
        t0 = time.time()
        self.l2file.write_level2_data()
        t1 = time.time()
        self.filter_runtimes["l2_write"] = t1 - t0
        self.filter_processtimes["l2_write"] = pt1 - pt0
        self.printer.debug_print(f"[{self.rank}] Finished l2 file write.")




if __name__ == "__main__":
    if "OMP_NUM_THREADS" in os.environ:
        omp_num_threads = int(os.environ["OMP_NUM_THREADS"])
    else:
        omp_num_threads = 1
    l2r = l2gen_runner(omp_num_threads=omp_num_threads)
    l2r.run()
    l2r.clean_up()