""" Example MPI usage from another program:
runlist = []
if rank == 0:
    runlist = read_runlist(params)
runlist = self.comm.bcast(runlist, root=0)
"""

import os
import time
import random

def read_runlist(params):
    """ Given a argparse parameter file object, returns the runlist specified (in the object), within the cut parameters set.
        The relevant cut parameters from the parameter object are:
            * obsid_start
            * obsid_stop
            * time_start_cut
            * time_stop_cut

    Returns:
        list: A 2D nested list, with the first dimension being each scan in the runlist, and the second being info about that scan:
                0. scanid (int) - The scanID of the respective scan.
                1. mjd_start (float) - Start of scan time in MJD.
                2. mjd_stop (float) - Stop of scan time in MJD.
                3. scantype (int) - Type of scan, in the bitmask format (16 = circular, 32 = CES, 32768 = Liss)
                4. fieldname (str) - Name of field, usually "co2", "co6" or "co7".
                5. l1_filename (str) - Full path to the level1 file where this scan is located.
    """

    print(f"Creating runlist in specified obsid range [{params.obsid_start}, {params.obsid_stop}]")
    print(f"Runlist file: {params.runlist}")
    if len(params.allowed_scantypes) > 1 or 32 not in params.allowed_scantypes:
        print("######### WARNING #########")
        print("RUNNING WITH NON-CES SCANS!")

    # Create list of already processed scanids.
    existing_scans = []
    for dir in os.listdir(params.level2_dir):
        if os.path.isdir(os.path.join(params.level2_dir, dir)):
            for file in os.listdir(os.path.join(params.level2_dir, dir)):
                if file[0] == ".":  # Delete any left-over hidden files from previously aborted runs.
                    if os.path.exists(os.path.join(os.path.join(params.level2_dir, dir), file)) and time.time() - os.stat(os.path.join(os.path.join(params.level2_dir, dir), file)).st_mtime > 60:  # If file still exists, and is more than 60 seconds old (margin to not accidentally delete files in the middle of writing).
                        os.remove(os.path.join(os.path.join(params.level2_dir, dir), file))
                elif file[-3:] == ".h5" or file[-4:] == ".hd5":
                    if len(file) == 16 or len(file) == 17:  # In order to not catch the intermediate debug files.
                        existing_scans.append(int(file.strip(".").split(".")[0].split("_")[1]))

    with open(params.runlist) as my_file:
        lines = [line.split() for line in my_file]
    i = 0
    runlist = []
    n_fields = int(lines[i][0])
    i = i + 1
    for i_field in range(n_fields):
        n_scans_tot = 0
        n_scans_outside_range = 0
        n_scans_already_processed = 0
        n_scans_too_short = 0
        n_right_scantype = 0
        fieldname = lines[i][0]
        n_obsids = int(lines[i][1])
        i = i + 1
        for j in range(n_obsids):
            obsid = "0" + lines[i][0]
            n_scans = int(lines[i][3])
            if fieldname in params.fields:
                l1_filename = lines[i][-1]
                l1_filename = l1_filename.strip("/")  # The leading "/" will stop os.path.join from joining the filenames.
                l1_filename = os.path.join(params.level1_dir, l1_filename)
                for k in range(n_scans):
                    scantype = int(float(lines[i+k+1][3]))
                    if scantype != 8192:
                        n_scans_tot += 1
                    if scantype in params.allowed_scantypes:
                        n_right_scantype += 1
                        if params.obsid_start <= int(obsid) <= params.obsid_stop:
                            scanid = int(lines[i+k+1][0])
                            l2_filename = f"{fieldname}_{scanid:09}.h5"
                            l2_filename = os.path.join(params.level2_dir, fieldname, l2_filename)
                            mjd_start = float(lines[i+k+1][1]) + params.time_start_cut/(60*60*24)  # seconds -> MJD
                            mjd_stop = float(lines[i+k+1][2]) - params.time_stop_cut/(60*60*24)
                            scan_length_seconds = (mjd_stop - mjd_start)*60*60*24
                            if scan_length_seconds > params.min_allowed_scan_length:
                                if not scanid in existing_scans:
                                    runlist.append([scanid, mjd_start, mjd_stop, scantype, fieldname, l1_filename, l2_filename])
                                else:
                                    n_scans_already_processed += 1
                            else:
                                n_scans_too_short += 1
                        else:
                            n_scans_outside_range += 1
            i = i + n_scans + 1

        
        if fieldname in params.fields:
            print(f"Field name:                 {fieldname}")
            print(f"Obsids in runlist file:     {n_obsids}")
            print(f"Scans in runlist file:      {n_scans_tot}")
            print(f"Scans of right scantype:    {n_right_scantype}")
            print(f"Scans included in run:      {len(runlist)}")
            print(f"Scans too short:            {n_scans_too_short}")
            print(f"Scans outside obsid range:  {n_scans_outside_range}")
            print(f"Scans already processed:    {n_scans_already_processed}")

    random.seed(42)
    random.shuffle(runlist)  # Shuffling the runlist helps with load distribution, as some (especially early) scans are larger than others.        

    ### Runlist splitting options ###
    if params.runlist_split_num_i >= params.runlist_split_in_n:
        raise ValueError(f"--runlist_split_num_i (currently {params.runlist_split_num_i}) cannot be equal or larger than --runlist_split_in_n (currently {params.runlist_split_in_n}).")
    if params.runlist_split_in_n > 1:
        len_runlist = len(runlist)
        runlist_start_idx = (len_runlist*params.runlist_split_num_i)//params.runlist_split_in_n
        runlist_stop_idx = (len_runlist*(params.runlist_split_num_i+1))//params.runlist_split_in_n
        print(runlist_start_idx, runlist_stop_idx)
        if params.runlist_split_num_i == params.runlist_split_in_n - 1:
            runlist_stop_idx = len_runlist
        runlist = runlist[runlist_start_idx:runlist_stop_idx]
        print(f"Runlist splitting enabled: Running part {params.runlist_split_num_i} of {params.runlist_split_in_n}.")
        print(f"Runlist cut: [{runlist_start_idx}:{runlist_stop_idx}], {runlist_stop_idx-runlist_start_idx} scans.")

    return runlist