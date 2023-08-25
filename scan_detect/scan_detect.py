from __future__ import print_function
import pickle
import errno
import h5py
import numpy as np
#import matplotlib.pyplot as plt
import glob
import os
import pwd
import grp
import sys
import math

featname = ["f0", "optical_pointing", "radio_pointing", "ground_scan",
            "circular_scan", "constant_elevation_scan", "ambient_load",
            "stationary", "sky_dip", "raster"]

# blacklist = np.load("/mn/stornext/d22/cmbco/comap/protodir/auxiliary/blacklists/blacklist_observerlog.npy")

class MetaData:
    # class containing all interesting info about a file.
    def __init__(self, obs_id, field, scan_mode,
                 scan_mode_bit, time_range,
                 scan_ranges, features, mean_az, mean_el,
                 el_std, file_path):
        self.obs_id = obs_id
        self.field = field
        self.scan_mode = scan_mode
        self.scan_mode_bit = scan_mode_bit
        self.time_range = time_range
        self.scan_ranges = scan_ranges
        self.features = features
        self.mean_az = mean_az
        self.mean_el = mean_el
        self.el_std = el_std
        self.file_path = file_path


def find_feat(time, time_feat, feat):  # hat tip: https://stackoverflow.com/a/26026189/5238625
    idx = np.searchsorted(time_feat, time, side="left")
    if idx > 0 and (idx == len(time_feat) or math.fabs(time - time_feat[idx-1]) < math.fabs(time - time_feat[idx])):
        return feat[idx-1]
    else:
        return feat[idx]


def find_tsys_ranges(mjd, feat):
    tsys_bit = 13
    i = 0
    starts = []
    ends = []

    was_on_scan = False

    i = 0
    for time in mjd:
        if (feat[i] & 1 << tsys_bit):
            is_on_scan = True
        else:
            is_on_scan = False

        if (not was_on_scan) and is_on_scan:
            starts.append(time)

        if was_on_scan and (not is_on_scan):
            ends.append(time)
        i += 1
        was_on_scan = is_on_scan

    if len(starts) == len(ends) + 1:
        ends.append(mjd[-1])

    return np.array((starts, ends)).transpose()


def find_scan_ranges(mjd, status, scan_mode, feat, feat_mjd, field, params):
    # find the ranges of each scan and return list with start and end times
    scan_status = 1
    reaquire_status = 2
    stationary = 0
    starts = []
    ends = []

    tsys_status = 8192

    if ((scan_mode == 'raster') and (field == 'jupiter')):
        status = feat
        mjd = feat_mjd
        scan_status = 512
        reaquire_status = 8192 + 512
        tsys_status = 8192

    was_on_scan = False
    # if all of one file is stationary, treat as single scan. Typical for ambient load etc.
    if all(s == 0 for s in status):
        starts.append(mjd[0])
        ends.append(mjd[-1])
        if (field == 'halt'):
            n = math.floor((ends[0] - starts[0]) * (24 * 60 / 4.0))
            tm = np.linspace(starts[0], ends[0], n + 1)
            starts = tm[:-1]
            ends = tm[1:]
        elif (field == 'NCP') or (field == 'ncpfixed'):
            n = math.floor((ends[0] - starts[0]) * (24 * 60 / 10.0))
            tm = np.linspace(starts[0], ends[0], n + 1)
            starts = tm[:-1]
            ends = tm[1:]
        return np.array((starts, ends)).transpose()

    dt = mjd[1] - mjd[0]
    i = 0
    for time in mjd:
        if status[i] == scan_status:
            is_on_scan = True
        elif status[i] == reaquire_status:
            is_on_scan = False
        elif status[i] == tsys_status:
            is_on_scan = False
        elif status[i] == stationary:
            is_on_scan = False
        else:
            is_on_scan = False

            print('Unknown status, time:', time, ' status:', status[i])
            # sys.exit()
        if (not was_on_scan) and is_on_scan:
            starts.append(time)

        if was_on_scan and (not is_on_scan):
            ends.append(time)
        i += 1
        was_on_scan = is_on_scan

    if len(starts) == len(ends) + 1:
        ends.append(mjd[-1])

    if (field == 'halt'):  # long halts are separated into many scans of ~4 mins each
        if (len(starts) == 1):
            n = int(np.floor((ends[0] - starts[0]) * (24 * 60 / 4.0)))

            tm = np.linspace(starts[0], ends[0], n + 1)
            starts = tm[:-1]
            ends = tm[1:]

    n_scans = len(starts)
    for i in range(n_scans):
        index = n_scans - i - 1
        # if (ends[index] - starts[index] < params['scandetect_minimum_scan_length'] / (24 * 60)):
        #     ends.pop(index)
        #     starts.pop(index)

    return np.array((starts, ends)).transpose()


def find_mean_values(scan_ranges, t_point, az, el):
    # Find useful stuf for each scan
    n_scans = len(scan_ranges)
    mean_az = np.zeros(n_scans)
    mean_el = np.zeros(n_scans)
    el_std = np.zeros(n_scans)
    i = 0
    for start, end in scan_ranges:
        start_ind = np.argmax(t_point > start)
        if t_point[-1] > end:
            end_ind = np.argmax(t_point > end) - 1
        else:
            end_ind = len(t_point)
        mean_az[i] = np.mean(az[start_ind:end_ind])
        mean_el[i] = np.mean(el[start_ind:end_ind])
        el_std[i] = np.std(el[start_ind:end_ind])
        i += 1
    return mean_az, mean_el, el_std


def find_file_dict(foldername, params, verb, mem={}, bad=[]):
    # Makes list of files with all relevant info
    file_dict = {}

    # os.chdir(foldername)
    # for file in glob.glob(foldername + "/**/*.hd5", recursive=True):
    # for file in glob.glob(foldername + "/**/comp_*.hd5", recursive=True):
    from tqdm import tqdm
    for file in tqdm(glob.glob(foldername + "/**/comap-*.hd5", recursive=True)):
        if file in bad:
            pass
        else:
            try:
                with h5py.File(file, mode="r") as fd:
                    # Get the attributes
                    try:
                        att = fd['comap'].attrs
                        hk = fd['hk']
                        # print(att['source'])
                        if att['obsid'] == '':
                            obs_id = int(file[-8:-5])
                            field = att['source']
                        else:
                            obs_id = int(att['obsid'])
                            try:
                                field = att['source'].decode('utf-8')  # 'jupiter'
                            except:  # Got an error on the decode. Inserted try-except.  -Jonas
                                field = att['source']
                        new_format = True
                    except KeyError:
                        new_format = False

                        print('Wrong format:', file)
                        obs_id = -1
                        bad.append(file)

                    if new_format:
                        # correct_scan_type = False
                        # scan_mode = "none"
                        # for scantype_feature in params["scantypes"]:
                            # if att["features"]
                        # for power in range(32):
                        #     if 2**power & att['features'] and 2**power in params["allowed_scantypes"]:
                        #         correct_scan_type = True
                        # if not correct_scan_type:
                        #     pass
                        # for bit, name in enumerate(featname):
                        #     if (att['features'] & 1 << bit):
                        #         scan_mode = featname[bit]
                        # if params["ces_only"] and scan_mode != "constant_elevation_scan":
                        #     pass
                        # else:
                        is_in_range = params['obsid_stop'] >= obs_id >= params['obsid_start']
                        # is_blacklisted = obs_id in blacklist
                        if (str(obs_id) in mem.keys()) and is_in_range: # and not is_blacklisted:
                            file_dict[str(obs_id)] = mem[str(obs_id)]
                        elif is_in_range: # and not is_blacklisted:
                            # print('File: '+file)
                            file_path = file.replace(foldername, '')
                            if obs_id < 200:
                                field = att['source']
                            else:
                                try:
                                    field = att['source'].decode('utf-8')
                                except:  # Got an error on the decode. Inserted try-except.  -Jonas
                                    field = att['source']
                            # if field == "NCP": ### comment out
                            #     print('NCP', obs_id)   ####

                            if field[0:4] == 'co7,':
                                field = 'co7'
                            if field[0:4] == 'co4f':
                                field = 'co4'
                            if field[0:4] == 'co2f':
                                field = 'co2'
                            if field[0:5] == 'TauAf':
                                field = 'TauA'
                            if field[0:7] == 'jupiter':
                                field = 'jupiter'
                            #if field[0:7] == 'jupiter': field = 'jupiter'
                            print('File: ' + file, field)
                            features = att['features']  # 2 #
                            is_one_feat = True  # testing for multiple features
                            scan_mode = None
                            for bit, name in enumerate(featname):
                                if (features & 1 << bit):
                                    #print("(f%d) %s" % (bit, featname[bit]))
                                    scan_mode = featname[bit]
                                    # These features are treated as separate fields
                                    if scan_mode == "ambient_load":
                                        field = scan_mode
                                        break
                                    if scan_mode == "ground_scan":
                                        field = scan_mode
                                        break
                                    if scan_mode == "stationary":
                                        if field == "NCP":
                                            pass
                                        elif field == "ncpfixed":
                                            pass
                                        else:
                                            field = scan_mode
                                        break
                                    if scan_mode == "sky_dip":  # seems like sky dip is associated with other field
                                        field = scan_mode
                                        break
                                    scan_mode_bit = bit
                                    # print(is_one_feat)
                                    ########## Test if multiple features ###########
                                    if not is_one_feat:
                                        print('Warning, multiple features')
                                        for bit, name in enumerate(featname):
                                            if (features & 1 << bit):
                                                print("(f%d) %s" %
                                                    (bit, featname[bit]))
                                    is_one_feat = False
                                    # break  # We can in principle have multiple features, for now we just choose first
                            if (features & 1 << 15):
                                # print(features)
                                # print('lissajous!!!!!!!!!!')
                                scan_mode = "lissajous"
                            if scan_mode is None:
                                scan_mode = "other"
                                field = scan_mode

                            scan_mode_bit = features
                            time = fd[u'spectrometer/MJD']  # ['time']
                            time_range = (time[0], time[-1])
                            # ['hk/time_track']
                            t_status = fd[u'hk/antenna0/deTracker/utc']
                            # ['hk/lissajous_status']
                            status = fd[u'hk/antenna0/deTracker/lissajous_status']
                            el = fd[u'pointing/elActual']  # ['point_tel'][0, :, 1]
                            az = fd[u'pointing/azActual']  # ['point_tel'][0, :, 0]

                            feat_arr = fd[u'/hk/array/frame/features']
                            t_feat = fd[u'/hk/array/frame/utc']
                            tsys_ranges = find_tsys_ranges(t_feat, feat_arr)
                            scan_ranges = find_scan_ranges(
                                t_status, status, scan_mode,
                                feat_arr, t_feat, field, params
                            )
                            if scan_mode == 'stationary':
                                try:
                                    scan_ranges[0, 0] = tsys_ranges[0, 1]
                                    scan_ranges[-1, 1] = tsys_ranges[-1, 0]
                                except:
                                    print('problem with end of tsys', obs_id)
                            scan_ranges = np.concatenate(
                                (tsys_ranges, scan_ranges), axis=0)
                                
                            scan_ranges = scan_ranges[scan_ranges[:, 0].argsort()]
                            # print(scan_ranges)
                            features = np.zeros(len(scan_ranges[:, 0]))
                            for i, t in enumerate(scan_ranges.mean(1)):
                                features[i] = find_feat(t, t_feat, feat_arr)
                                tsys_bit = 13
                                if (int(features[i]) & 1 << tsys_bit):
                                    features[i] = 8192
                                # liss_bit = 15
                                # if (int(features[i]) & 1<<tsys_bit):
                                #     features[i] = 32768
                            t_point = fd[u'pointing/MJD']  # ['time_point']
                            #plt.plot(t_point, el)
                            # plt.show()
                            if len(scan_ranges) > 0:
                                mean_az, mean_el, el_std = find_mean_values(
                                    scan_ranges, t_point, az, el)
                                metadata = MetaData(
                                    obs_id, field, scan_mode,
                                    scan_mode_bit, time_range,
                                    scan_ranges, features, mean_az,
                                    mean_el, el_std, file_path)
                                file_dict[str(obs_id)] = metadata
                                mem[str(obs_id)] = metadata
            except OSError:
                if verb:
                    print('\nUnable to read file:')
                    print(file, '\n')
                pass
            except KeyError:
                bad.append(file)
                if verb:
                    print('\nMissing key in file:')
                    print(file, '\n')
                pass

    return file_dict, mem, bad


# hat tip: https://stackoverflow.com/a/19201448/5238625
def save_obj(obj, name, folder):
    with open(folder + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
#        uid = pwd.getpwnam('haavarti').pw_uid
#        gid = grp.getgrnam('astcomap').gr_gid
#        os.chown(folder + name + '.pkl', uid, gid)


def load_obj(name, folder):
    with open(folder + name + '.pkl', 'rb') as f:
        return pickle.load(f)


# From Tony Li
def ensure_dir_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


# param_file = sys.argv[1]
try:
    param_file = sys.argv[1]
except IndexError:
    print('You need to provide param file as command-line argument')
    sys.exit()


sys.path.append("/mn/stornext/d22/cmbco/comap/jonas/pipeline")  # TODO: Not use hard-coded path.
from l2gen_argparser import parser
params = parser.parse_args()
if not params.runlist:
    raise ValueError("A runlist must be specified in parameter file or terminal.")
param_file = params.param
params = vars(params)  # Accept-mod was written with params as a dict, so we just do Namespace -> dict so I don't have to rewrite stuff.


runlist_name = params['runlist']
foldername = params['level1_dir']
require_tsys = True  # params['REQUIRE_TSYS']
verb = False  # params['VERBOSE_PRINT']
remove_eng = True  # params['REMOVE_ENGINEERING_RUNS']

aux_data_path = "/mn/stornext/d22/cmbco/comap/protodir/auxiliary/obj/"  # params['AUX_SAVED_DATA']
memory_file = 'mem'
bad_file = 'bad'


ensure_dir_exists(aux_data_path)

# /home/havard/Documents/COMAP/scan_detect'  # ['lvl1', 'ex']#'2018/10/'

try:
    mem = load_obj(memory_file, aux_data_path)
except FileNotFoundError:
    mem = {}


try:
    bad = load_obj(bad_file, aux_data_path)
except FileNotFoundError:
    bad = []

file_dict, mem, bad = find_file_dict(
    foldername, params, verb, mem=mem, bad=bad)

#print(file_dict)

save_obj(mem, memory_file, aux_data_path)

save_obj(bad, bad_file, aux_data_path)
# obs_ids = list()#[int(f.obs_id for f in file_list]

# def argsort(seq):  # sort files by obsid
#     return sorted(range(len(seq)), key=seq.__getitem__)

# # indices = argsort(obs_id_list)
sorted_obs_ids = sorted([int(obs_id) for obs_id in file_dict.keys()])

#print(sorted_obs_ids)
file_list = [file_dict[str(obsid)] for obsid in sorted_obs_ids]
#print(file_list[14].features)

n_files = len(file_list)


if require_tsys:
    f = open("tsys_error.txt", "w")
    for i in range(n_files):
        index = n_files - i - 1
        #print("what we have: ",file_list[index].obs_id, file_list[index].field, file_list[index].scan_ranges, file_list[index].features)
        
        if not ((file_list[index].features[0] == 8192) and (file_list[index].features[-1] == 8192)):
            if file_list[index].field[:2] == 'co':
                f.write(str(file_list[index].obs_id) + '\n')
            file_list.pop(index)
    f.close()

n_files = len(file_list)
if remove_eng:  # Do not include engineering runs
    for i in range(n_files):
        index = n_files - i - 1
        if (len(file_list[index].features) > 1):
            if (int(file_list[index].features[1]) & 1 << 10): 
                file_list.pop(index)
#print(file_list)



n_files = len(file_list)

if verb:
    print('Total number of files in runlist: ', n_files)

# print(file_list[0].time_range[1])

field_names = ('jupiter', 'venus', 'TauA', 'shela', 'hetdex', 'patch1',
               'patch2', 'co1', 'co2', 'co3', 'co4', 'co5', 'co6', 'co7', 'mars',
               'fg1', 'fg2', 'fg3', 'fg4', 'fg5', 'fg6', 'fg7', 'ambient_load',
               'ground_scan', 'stationary', 'sky_dip', 'CasA', 'CygA', 'other',
               'NCP', 'ncpfixed')

# try:
#     fieldn = sys.argv[2]
#     field_names = (fieldn,)
#     runlist_name = runlist_name[:-4] + '_' + fieldn + runlist_name[-4:]
# except IndexError:
#     pass

print("Runlist: ", runlist_name)


def get_jackknives(field):
    n_jk = 10
    n_obs = len(field)
    jackknives = np.zeros((n_obs, n_jk))
    n_split = n_obs // 2

    # even/odd split
    for i, current_file in enumerate(field):
        jackknives[i, 0] = current_file.obs_id % 2

    # day/night split
    hours = [(0.5 * (f.time_range[0] + f.time_range[-1]) * 24 - 7) %
             24 for f in field]
    closetonight = [min(abs(2 - hour), abs(26 - hour)) for hour in hours]
    nightrank = np.argsort(np.argsort(closetonight))
    
    for i, current_file in enumerate(field):
        if (nightrank[i] >= n_split):
            jackknives[i, 1] = 1
        else:
            jackknives[i, 1] = 0

    # half mission split
    for i, current_file in enumerate(field):
        if (i >= n_split):
            jackknives[i, 2] = 1
        else:
            jackknives[i, 2] = 0

    # mean el
    elevations = [np.mean(current_file.mean_el) for current_file in field]
    elrank = np.argsort(np.argsort(elevations))

    for i, current_file in enumerate(field):
        if (elrank[i] >= n_split):
            jackknives[i, 3] = 1
        else:
            jackknives[i, 3] = 0

    return jackknives


def write_runlist(file_list, runlist_name):
    # Writes runlist based on the file list
    out_file = open(runlist_name, "w")
    n_files_used = 0
    field_list = []
    for field in field_names:
        files_in_field = []
        for current_file in file_list:
            if current_file.field == field:
                if len(current_file.scan_ranges) > 2:
                    files_in_field.append(current_file)

        n_files_in_field = len(files_in_field)

        if n_files_in_field > 0:
            n_files_used += len(files_in_field)
            field_list.append(files_in_field)

    n_fields = len(field_list)
    out_file.write("%d \n" % n_fields)
    print('Number of fields observed: ', n_fields)
    for field in field_list:
#        field = field[-100:] # this line should be commented out!!
        out_file.write("%s   %d \n" % (field[0].field, len(field)))

        jackknives = get_jackknives(field)
        for i, current_file in enumerate(field):
            n_scans = len(current_file.scan_ranges)
            out_file.write("  %06i  %17.10f %17.10f %02i %s \n" %
                           (current_file.obs_id, current_file.time_range[0],
                            current_file.time_range[-1], n_scans,
                            current_file.file_path))

            jk_string = ""
            for jk in jackknives[i]:
                jk_string += ' %i' % jk
            jk_string += '  \n'

            for j, scan_range in enumerate(current_file.scan_ranges):
                scanstring = "     %06i%02i %17.10f %17.10f %i %10.6f %10.6f %10.6f" % (
                    current_file.obs_id, j + 1,
                    scan_range[0],
                    scan_range[1],
                    current_file.features[j],
                    current_file.mean_az[j],
                    current_file.mean_el[j],
                    current_file.el_std[j],
                )
                out_file.write(scanstring + jk_string)

    out_file.close()
    print('Number of files used: ', n_files_used)

write_runlist(file_list, runlist_name)
