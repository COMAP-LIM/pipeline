import warnings
import h5py 
import numpy as np
import pickle
# import healpy as hp
import matplotlib.pyplot as plt
# from scipy.fftpack import rfft, irfft
# from scipy.fftpack import fft, ifft, rfft, irfft, fftfreq, rfftfreq, next_fast_len
from scipy.fftpack import  next_fast_len
# import scipy as sp
# from scipy import ndimage

from pathlib import Path
from astropy.io import fits
from astropy.wcs import WCS
from pathlib import Path
from numpy.fft import fft, ifft, rfft, irfft, fftfreq, rfftfreq
import time as timer
from plot import *
from tod_generator import TOD_Gen
np.random.seed(4)



file = Path(__file__)
src = Path(__file__).parent.absolute()
tod_gen_folder = src.parent.absolute() # leah folder
conv_folder = Path(tod_gen_folder/ 'data_tground')

# data_folder=  Path(tod_gen_folder/ 'level2_dir_leah6_time_no_poly/co7')
# data_folder_constant =  Path(tod_gen_folder/ 'level2_dir_leah5/co7')

# outfile_figs = Path(tod_gen_folder/'pipeline/figs_testing/azimuth_ground_model')


if __name__ == "__main__":
        
        
        # filter_dic = {'normalized':'_3_norm', 'pointing':'_4_pnt_bi', 'poly':'_5_poly'}
        filter_dic = {'poly':''}
        version = 'constant_Tsys'
        ######
        # timezone = 'Constant'
        # data_folder=  Path(tod_gen_folder/ 'level2_dir_leah5/co7')
        # outfile_figs = Path(tod_gen_folder/'pipeline/figs_testing/figs_tground_model/with_pointing')
        
        for filtr in filter_dic:
                print(filtr)
                timezone = 'Time varying'
                print(f'Timezone is {timezone}')
                data_folder=  Path(tod_gen_folder/ f'level2_dir_leah_time_{filtr}/co7_{version}/co7_v1')
                outfile_figs = Path(tod_gen_folder/f'figs/figs_{version}')
                ######


                # scans = ['co7_001196211','co7_001320603', 'co7_001454102', 'co7_001556802', 'co7_001832603', 'co7_002233305', 'co7_002489606', 'co7_002733208', 'co7_003004205','co7_003174311', 'co7_003431206', 'co7_003754110','co7_003961103' ]
                
                scans = ['co7_003004205' ]

                images = []
                minmax = []
                titles = []
                

                images_trest = []
                minmax_trest = []
                titles_trest = []
                for scan in scans:
                ############################################################
                        tod_gen = TOD_Gen()
                        

                        params = ['time', 'Tsys', 'freq_bin_centers', 'tod', 'Gain', 'point_tel',  'T_rest', 'Tsys', 'hk_time', 'air_temp', 'scan_start_idx_hk', 'scan_stop_idx_hk', 'sigma0']#, 'scale']
                
                        read_file_start = timer.time()
                        # tod, params =  tod_gen.read_file(data_folder/'co7_002911511_3_norm.h5', model=True, parameter_list=params, units=False)
                        tod, params =  tod_gen.read_file(data_folder/f'{scan}{filter_dic[filtr]}.h5', model=True, parameter_list=params, units=False)
                        
                        read_file_end = timer.time()
                        print(f'Time of reading files {read_file_end-read_file_start}')
                        
                        with open(f"{outfile_figs}/params/params_{scan}.npy", 'wb') as f:
                                pickle.dump(params, f, protocol=4)


                
                        ############################################################
                        

                        with open(f"{outfile_figs}/params/params_{scan}.npy", "rb") as f:
                                params = pickle.load(f)
                                
                        params = dict(params.items())
                        # print(params.keys())

                        tod = params['tod']
                        Gain = params['Gain']
                        Tsys = params['Tsys']
                        T_rest = params['T_rest']
                        sigma0 = params['sigma0']
                        

                        # correlated = params['correlated']
                        # white_noise = params['white_noise']
                        freqs = params['freq_bin_centers']

                        time = params['time'] #Time 0 is 0.0 time -1 is 379.9808580195531
                        time = (time - time[0])*24*60*60
                        # print(f' Time is from {time[0]} to {time[-1]}')

                        point_tel = params['point_tel']
                        hk_air_temp = params['air_temp']
                        hk_time = params['hk_time']
                        hk_time = (hk_time - hk_time[0])*24*60*60

                        start_scan_idx = params['scan_start_idx_hk']
                        stop_scan_idx = params['scan_stop_idx_hk']

                        
                        # print(f'start_scan_idx is {start_scan_idx}, stop scan idx is {stop_scan_idx}')
                        
                        
                        az = point_tel[:, :, 0] # Current point_tel is [feed, time, az/el]
                        el = point_tel[:, :, 1]

                        scale = 0 # 1/10000 # Scale for whitenoise. This is just a reminder, scaling actually happens before pipeline
                        # scale = params['scale']

                        extra_params_dict = {'hk_time': hk_time, 'hk_air_temp':hk_air_temp, 'scan_start_idx_hk': start_scan_idx, 'scan_stop_idx_hk':stop_scan_idx}

                        
                        tod_gen = TOD_Gen(tod=tod, az=az, el=el, Gain=Gain, Tsys=Tsys, time=time, freq_bin_centers = freqs, extra_params_dict=extra_params_dict) 
                        
                
                        convolution_files = [ conv_folder / 'conv_26GHz_NSIDE256.npy', conv_folder / 'conv_30GHz_NSIDE256.npy', conv_folder / 'conv_34GHz_NSIDE256.npy',]

                        # d_model, G, correlated, Tsys, white_noise, Trest = tod_gen.get_data_model(tground_files=convolution_files, corr_noise=False, downsampled=False, point=[az, el], scale = scale)
                        # air_interp, all_air_smooth, original_interp = tod_gen.interpolating_air_temp(hk_air_temp, hk_time, hk_start=start_scan_idx, hk_end=stop_scan_idx)



                        print(f'Shape of TOD is {np.shape(tod)}')
                        # T_rest_waterfall, trest_minmax = waterfall(Trest[0], len(d_model[0, 0, 0, :]), title=f'{timezone} ground pickup for {scan} with wn scaled by {scale}', save=f'{outfile_figs}/T_rest/T_rest_{scan}', shape = 4*1024, minmax=True)
                        # d_model_waterfall, d_minmax = waterfall(d_model[0], len(d_model[0, 0, 0, :]), title=f'TOD for {scan} with wn scaled by {scale}', save=f'{outfile_figs}/TOD/TOD_{scan}', shape = 4*1024, minmax=True)
                        d_model_waterfall, d_minmax = waterfall(tod[0], len(tod[0, 0, 0, :]), title=f'TOD for {scan} with wn scaled by {scale}', save=f'{outfile_figs}/TOD/TOD_{scan}', shape = 4*64, minmax=True)

                        tod_flat = tod[0, :, :, 2500].flatten()
                        print(np.shape(tod_flat))
                        plt.figure()
                        plt.plot(tod_flat)
                        plt.ylabel('K')
                        plt.xlabel('freq')
                        plt.savefig(f'{outfile_figs}/TOD/timestep_over_freq')
                        plt.close()

                        
                        plt.figure()
                        plt.plot(sigma0[0].flatten())
                        # plt.ylabel('K')
                        plt.xlabel('freq')
                        plt.title(f'Feed 0')
                        plt.savefig(f'{outfile_figs}/TOD/sigma0')
                        plt.close()
                        # # sys_time_end = (np.where(hk_time < time[-1]+0.01)[0][-1])
                        # fig, (ax1, ax2) = plt.subplots(2, 1)
                
                        # ax1.plot(air_interp, label = 'smoothed')
                        # ax1.plot(original_interp, label ='unsmoothed')
                        # # ax1.plot(time, t_obs[0, 0, 0, :], label='T_obs')
                        # # ax1.set_xlabel('Timesteps')
                        # ax1.set_ylabel('K')
                        # ax1.set_xlabel('obs time')
                        # ax1.legend()
                        # ax1.set_title('Smoothed vs unsmoothed air temp in obs')

                        # ax2.plot(hk_time, hk_air_temp)
                        # ax2.plot(hk_time, all_air_smooth)
                        # ax2.plot(hk_time[start_scan_idx:stop_scan_idx], hk_air_temp[start_scan_idx:stop_scan_idx])

                        # # ax2.plot(time, air)
                        # ax1.set_ylabel('K')
                        # ax2.set_xlabel('Timesteps in hk')
                        # ax2.set_title('Hk air temp over time')
                        # fig.tight_layout()
                        # fig.savefig(f'{outfile_figs}/Air/Smooth_vs_unsmooth_air_over_time_{scan}')
                        
                
        exit()




# MAPS
if __name__ == "__main__":

        #################################################################
        """
        Make map with constant ground pickup model
        - Constant model is in level2_dir_leah5
        - maps_leah2

        And time varying model
        - Which is in level2_dir_leah6
        - maps_leah


        """
        #################################################################
        

        outfile_figs = Path("/mn/stornext/d16/cmbco/comap/leah/figs/figs_constant_Tsys")
        outfile_figs = Path("/mn/stornext/d16/cmbco/comap/leah/figs/figs_constant_BB")

        # timezone = 'time_varying'
        filtrs = ['normalized', 'pointing', 'poly']
        # filtrs = ['pointing']
        scale = 0#1/100000

        # version = 'co7_constant_Tsys'
        version = 'co7_constant_BB'
        # version = ''

        for i in range(len(filtrs)):
                filtr = filtrs[i]
                path_maps1 = Path(tod_gen_folder / f"maps_leah_time/{filtr}/{version}") # time varying
                path_maps2 = Path(tod_gen_folder / f"maps_leah_constant/{filtr}/{version}") 
                file1 =  Path(path_maps1/"co7_Groundpickup_bb.h5")
                file2 =  Path(path_maps2/"co7_Groundpickup_bb.h5")

                parameter_list = ["map", "nhit"]
                parameter_list_c = ["map", "nhit"]

                params_time = {}
                params_constant = {}

                with h5py.File(file1, 'r') as infile:

                        # print(infile.keys())
                        for key in parameter_list:
                        
                                params_time.update({f'{key}': infile[key][()]})
                
                with h5py.File(file2, 'r') as f:

                        # print(f.keys())
                        for key in parameter_list:
                        
                                params_constant.update({f'{key}': f[key][()]})
                
                
                feed = 15
                sideband = 3
                freq_index = 25


                
                maps_1 = params_time['map']
                pix1 = maps_1[feed, sideband, :, 40, 60]*1e6
                map_1 = maps_1[feed, sideband, freq_index, ...]*1e6
        

                maps_2 = params_constant['map']
                pix2 = maps_2[feed, sideband, :, 40, 60]*1e6
                map_2 = maps_2[feed, sideband, freq_index, ...]*1e6
                
                hit_maps_1 = params_time['nhit']
                hit_pix1 = hit_maps_1[feed, sideband, :, 40, 60]*1e6
                hit_map_1 = hit_maps_1[feed, sideband, freq_index, ...]*1e6
        

                hit_maps_2 = params_constant['nhit']
                hit_pix2 =  hit_maps_2[feed, sideband, :, 40, 60]*1e6
                hit_map_2 = hit_maps_2[feed, sideband, freq_index, ...]*1e6
                
                        
                # print(f"Is map_1 0 anywhere for filter {filtr}? {np.any(map_1==0)}")
                # print(f"Is map_2 0 anywhere for filter {filtr}? {np.any(map_2==0)}\n")

                # print(f"Is map_1 nan anywhere for filter {filtr}? In {len(np.where(np.isnan(map_1))[0])} places")
                # print(f"Is map_2 nan anywhere for filter {filtr}? In {len(np.where(np.isnan(map_2))[0])} places")
                # print(f"Are they nan in the same places? {np.where(np.isnan(map_1))==np.where(np.isnan(map_2))}\n")

                # print(f"Min abs of map_1 is {np.nanmin(abs(map_1))}")
                # print(f"Min abs of map_2 is {np.nanmin(abs(map_2))}\n")
                print(maps_1.shape)
                dapper_mapper = [[2, 1, 25], [2, 3, 25 ], [2, 1, 25], [2, 3, 25 ], [6, 1, 25], [6, 3, 25 ], [10, 1, 25 ], [10, 3, 25 ], [13, 1, 25 ], [13, 3, 25 ], [15, 1, 25 ], [15, 3, 25 ], [18, 1, 25 ], [18, 3, 25 ]]
                # dapper_mapper = []
                # for i in range(0, len(maps_1), 5):
                #         for j in [25]:#range(0, len(maps_1[0, 0, :]), 30):
                #                 dapper_mapper.append([i, 0, j])
                #                 dapper_mapper.append([i, 1, j])
                #                 dapper_mapper.append([i, 2, j])
                #                 dapper_mapper.append([i, 3, j])
                #                 dapper_mapper.append([i, 0, j+2])
                #                 dapper_mapper.append([i, 1, j+2])
                #                 dapper_mapper.append([i, 2, j+2])
                #                 dapper_mapper.append([i, 3, j+2])


                plt.figure()
                fig, axs = plt.subplots(len(dapper_mapper), 2, figsize=(15, 5 * len(dapper_mapper)))

                for row, item in enumerate(dapper_mapper):
                        feed, sideband, freq_index = item
                        # print(f"feed = {feed}, sideband = {sideband}, freq_index = {freq_index}")
                        
                        freq_index = freq_index   #Trying different frequencies
                        freq = freq_index  # assuming direct mapping to GHz

                        map_t = maps_1[feed, sideband, freq_index, ...] * 1e6  # time-varying
                        map_c = maps_2[feed, sideband, freq_index, ...] * 1e6  # constant

                        cmap = plt.get_cmap("bwr")
                        cmap.set_bad("gray")

                        # Time-varying (top row)
                        im1 = axs[row, 0].imshow(map_t, cmap=cmap, interpolation='none',
                                                vmin=-np.nanmax(abs(map_t)), vmax=np.nanmax(abs(map_t)))
                        axs[row, 0].set_title(f'Time-varying model. Feed = {feed}, sideband = {sideband}, freq_idx = {freq_index}', pad=14)
                        axs[row, 0].invert_xaxis()
                        axs[row, 0].invert_yaxis()

                        axs[row, 0].set_xlabel("RA [pixel]")
                        axs[row, 0].set_ylabel("Dec [pixel]")


                        # Constant (bottom row)
                        im2 = axs[row, 1].imshow(map_c, cmap=cmap, interpolation='none',
                                                vmin=-np.nanmax(abs(map_c)), vmax=np.nanmax(abs(map_c)))
                        axs[row, 1].set_title(f'Constant model. Feed = {feed}, sideband = {sideband}, freq_idx = {freq_index}', pad=14)
                        axs[row, 1].invert_xaxis()
                        axs[row, 1].invert_yaxis()

                        axs[row, 1].set_xlabel("RA [pixel]")
                        axs[row, 1].set_ylabel("Dec [pixel]")

                        # Add colorbars (optional: one for each row or shared per column)
                        fig.colorbar(im1, ax=axs[row, 0], orientation='vertical', fraction=0.046, pad=0.02, label="μK")
                        fig.colorbar(im2, ax=axs[row, 1], orientation='vertical', fraction=0.046, pad=0.02, label="μK")

                        
                        


                        # # Shared colorbar
                        # cbar = fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.7)
                        # cbar.set_label("Temperature Difference [μK]")


                plt.tight_layout()
                plt.savefig(outfile_figs/f"MAP_C_T_{filtr}_random.png")
                plt.close()

                plt.figure()
                fig, axs = plt.subplots(len(dapper_mapper), 2, figsize=(15, 5 * len(dapper_mapper)))

                for row, item in enumerate(dapper_mapper):
                        feed, sideband, freq_index = item
                        # print(f"feed = {feed}, sideband = {sideband}, freq_index = {freq_index}")
                        
                        freq_index = freq_index   #Trying different frequencies
                        freq = freq_index  # assuming direct mapping to GHz

                        map_t = hit_maps_1[feed, sideband, freq_index, ...] * 1e6  # time-varying
                        map_c = hit_maps_2[feed, sideband, freq_index, ...] * 1e6  # constant

                        cmap = plt.get_cmap("bwr")
                        cmap.set_bad("gray")

                        # Time-varying (top row)
                        im1 = axs[row, 0].imshow(map_t, cmap=cmap, interpolation='none',
                                                vmin=-np.nanmax(abs(map_t)), vmax=np.nanmax(abs(map_t)))
                        axs[row, 0].set_title(f'Time-varying model. Feed = {feed}, sideband = {sideband}, freq_idx = {freq_index}', pad=14)
                        axs[row, 0].invert_xaxis()
                        axs[row, 0].invert_yaxis()

                        axs[row, 0].set_xlabel("RA [pixel]")
                        axs[row, 0].set_ylabel("Dec [pixel]")


                        # Constant (bottom row)
                        im2 = axs[row, 1].imshow(map_c, cmap=cmap, interpolation='none',
                                                vmin=-np.nanmax(abs(map_c)), vmax=np.nanmax(abs(map_c)))
                        axs[row, 1].set_title(f'Constant model. Feed = {feed}, sideband = {sideband}, freq_idx = {freq_index}', pad=14)
                        axs[row, 1].invert_xaxis()
                        axs[row, 1].invert_yaxis()

                        axs[row, 1].set_xlabel("RA [pixel]")
                        axs[row, 1].set_ylabel("Dec [pixel]")

                        # Add colorbars (optional: one for each row or shared per column)
                        fig.colorbar(im1, ax=axs[row, 0], orientation='vertical', fraction=0.046, pad=0.02, label="μK")
                        fig.colorbar(im2, ax=axs[row, 1], orientation='vertical', fraction=0.046, pad=0.02, label="μK")

                        
                        


                        # # Shared colorbar
                        # cbar = fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.7)
                        # cbar.set_label("Temperature Difference [μK]")


                plt.tight_layout()
                plt.savefig(outfile_figs/f"HIT_MAP_C_T_{filtr}_random.png")
                plt.close()
                
                     
                """
                
                nans_constant = (np.where(np.isnan(map_1)))
                # nans_time = (np.where(np.isnan(map_2)))
                # print(f'shape  of map is {np.shape(map_1)}')
                # map_1[30:90, 30:50] = (map_1[30:90, 30:50])*1e6*1e6
                # print(f'map time = {map_1[80,80]}\n and {map_1[40,40]}')
                # print(f'map constant = {map_2[80,80]}\n and {map_2[40,40]}')

                plt.figure()
                im = plt.imshow(map_1, cmap='seismic', interpolation='none', vmin=-np.nanmax(abs(map_1)), vmax=np.nanmax(abs(map_1)))
                cbar = plt.colorbar(im)
                # Reverse x-axis
                plt.gca().invert_xaxis()
                # Reverse y-axis
                plt.gca().invert_yaxis()
                cbar.set_label('micro K')
                plt.title(f'Map of time-varying ground pickup model. Wn scaled by {scale}')
                plt.xlabel('Ra [pixel]')
                plt.ylabel('Dec [pixel]')
                plt.savefig(outfile_figs/f'figs_through_filter_time_varying/{filtr}/maps'/f'map_test_time_varying')


                plt.figure()
                im = plt.imshow(map_2, cmap='seismic', interpolation='none', vmin=-np.nanmax(abs(map_2)), vmax=np.nanmax(abs(map_2)))
                cbar = plt.colorbar(im)
                # Reverse x-axis
                plt.gca().invert_xaxis()
                # Reverse y-axis
                plt.gca().invert_yaxis()
                cbar.set_label('micro K')
                plt.title(f'Map of constant ground pickup model. Wn scaled by {scale}. Many scans.')
                plt.xlabel('Ra [pixel]')
                plt.ylabel('Dec [pixel]')
                plt.savefig(outfile_figs/f'figs_through_filter_constant/{filtr}/maps'/'map_test_constant')
                plt.close()
                """
                plt.figure()
                plt.title(f'Plot of pixel in time-varying and constant model after {filtr}. Wn scaled by {scale}. Many scans.')
                plt.xlabel('Frequency [GHz]')
                plt.ylabel('micro K')
                plt.plot(pix1, label='time-varying')
                plt.plot(pix2, label='constant')
                plt.legend()
                plt.savefig(outfile_figs/f'{filtr}_pixl_compare')
                plt.close()

                plt.figure()
                plt.title(f'Plot of pixel in time-varying and constant model after {filtr}. Wn scaled by {scale}. Many scans.')
                plt.xlabel('Frequency [GHz]')
                plt.ylabel('micro K')
                plt.plot(hit_pix1, label='time-varying')
                plt.plot(hit_pix2, label='constant')
                plt.legend()
                plt.savefig(outfile_figs/f'{filtr}_hit_pixl_compare')
                plt.close()

       
        
        exit()


     



# MANY SCANS
if __name__ == "__main__":
        permission = input("Is ground pickup constant or time varying? (t/c): ").strip().lower()
        

        if permission == 't':
                data_folder=  Path(tod_gen_folder/ 'level2_dir_leah6_time_no_poly/co7')
                outfile_figs = Path(tod_gen_folder/'pipeline/figs_testing/figs_tground_model/many_scans')

        if permission == 'c':
                data_folder=  Path(tod_gen_folder/ 'level2_dir_leah5_constant_no_poly/co7')
                outfile_figs = Path(tod_gen_folder/'pipeline/figs_testing/figs_tground_model/many_scans_constant')

        
        
        scans = ['co7_001196211','co7_001320603', 'co7_001454102', 'co7_001556802', 'co7_001832603', 'co7_002233305', 'co7_002489606', 'co7_002733208', 'co7_003004205','co7_003174311', 'co7_003431206', 'co7_003754110','co7_003961103' ]
        
        # scans = ['co7_003004205' ]

        images = []
        minmax = []
        titles = []
        

        images_trest = []
        minmax_trest = []
        titles_trest = []
        for scan in scans:
        ############################################################
                tod_gen = TOD_Gen()
                

                params = ['time', 'Tsys', 'freq_bin_centers', 'tod', 'Gain', 'point_tel',  'T_rest', 'Tsys', 'hk_time', 'air_temp']
        
                read_file_start = timer.time()
                # tod, params =  tod_gen.read_file(data_folder/'co7_002911511_3_norm.h5', model=True, parameter_list=params, units=False)
                tod, params =  tod_gen.read_file(data_folder/f'{scan}_3_norm.h5', model=True, parameter_list=params, units=False)
                
                read_file_end = timer.time()
                print(f'Time of reading files {read_file_end-read_file_start}')
                
                with open(f"{outfile_figs}/params/params_{scan}.npy", 'wb') as f:
                        pickle.dump(params, f, protocol=4)


        
                ############################################################
                

                with open(f"{outfile_figs}/params/params_{scan}.npy", "rb") as f:
                        params = pickle.load(f)
                        
                params = dict(params.items())

                tod = params['tod']
                Gain = params['Gain']
                Tsys = params['Tsys']
                T_rest = params['T_rest']
                

                # correlated = params['correlated']
                # white_noise = params['white_noise']
                freqs = params['freq_bin_centers']

                time = params['time'] #Time 0 is 0.0 time -1 is 379.9808580195531
                time = (time - time[0])*24*60*60
                # print(f' Time is from {time[0]} to {time[-1]}')

                point_tel = params['point_tel']
                hk_air_temp = params['air_temp']
                hk_time = params['hk_time']
                hk_time = (hk_time - hk_time[0])*24*60*60

                
                
                az = point_tel[:, :, 0] # Current point_tel is [feed, time, az/el]
                el = point_tel[:, :, 1]

                scale = 1/10000 # Scale for whitenoise
                
                extra_params_dict = {'hk_time': hk_time, 'hk_air_temp':hk_air_temp}

                # tod_gen = TOD_Gen(tod=tod, az=az, el=el, Gain=Gain, Tsys=Tsys, time=time, freq_bin_centers = freqs, extra_params_dict=extra_params_dict) 
                
                # air_smoothed, all_smooth, no_smooth_interp = tod_gen.interpolating_air_temp(hk_air_temp, hk_time)

                # convolution_files = [ conv_folder / 'conv_26GHz_NSIDE256.npy', conv_folder / 'conv_30GHz_NSIDE256.npy', conv_folder / 'conv_34GHz_NSIDE256.npy',]

                # d_model, G, correlated, Tsys, white_noise, Trest = tod_gen.get_data_model(tground_files=convolution_files, corr_noise=False, downsampled=False, point=[az, el], scale = scale)
                
                # print(f'SHAPE OF GAIN IS {np.shape(Gain)}')
                
                plt.figure()
                for i in range(len(Gain)):
                        flat_gain = Gain[i].flatten()
                        plt.plot(flat_gain, label=f'feed {i}')
                plt.xlabel('# Frequency')
                plt.title(f'Gain for {scan} over freq')
                plt.legend()
                plt.savefig(f'{outfile_figs}/Gains/Gains_{scan}')
                plt.close()

                # # t_obs = tod_gen.tground_blackbody(air_smoothed)
                # plt.figure()
                # # Trest_waterfall, trest_minmax = waterfall(T_rest[0]-m, len(d_model[0, 0, 0, :]), f"Ground pickup from map lin interpolated, mean removed", f'{outfile_figs}/Tground_no_corr_no_white_noise_offset', shape = 4*1024, minmax=True)
                # model_waterfall, model_minmax = waterfall(tod[3], len(tod[0, 0, 0, :]), f"Waterfall plot of datamodel with white noise scaled by {scale} for {scan}", f'{outfile_figs}/waterfall_datamodel_{scan}', shape = 4*1024, minmax=True)
        
                
                # plt.close()

                # plt.figure()
                # # Trest_waterfall, trest_minmax = waterfall(T_rest[0], len(d_model[0, 0, 0, :]), f"Ground pickup from map lin interpolated", f'{outfile_figs}/Trest', shape = 1, minmax=True)
                # T_rest_waterfall, trest_minmax = waterfall(T_rest[0], len(tod[0, 0, 0, :]), f"Ground pickup from map lin interpolated. Whitenoise scaled by {scale} for {scan}", f'{outfile_figs}/Trest_{scan}', shape = 4*1024, minmax=True)
                # plt.close()


                # images.append(model_waterfall)
                # minmax.append(model_minmax)
                # titles.append(f"TOD. Wn scaled by {scale} for {scan}")

                # images.append(T_rest_waterfall)
                # minmax.append(trest_minmax)
                # titles.append(f"Ground pickup from map lin interpolated. Whitenoise scaled by {scale} for {scan}")

        

        # plt.figure()
        # sub_plot_TOD = subplot_waterfalls(images=images, titles=titles, minmax=minmax, save=f'{outfile_figs}/TOD_through_pipeline', n_cols = 4, n_rows = 4)
        # plt.close()

        # plt.figure()
        # sub_plot_TOD = subplot_waterfalls(images=images_trest, titles=titles_trest, minmax=minmax_trest, save=f'{outfile_figs}/Trest_through_pipeline', n_cols = 4, n_rows = 4)
        # plt.close()

                # print(f'Trest of shape {np.shape(T_rest)} has last 100 is {T_rest[0, 2, 26, -100:-1]}')
                # print(f'Trest first 100 is {T_rest[0, 2, 26, 0:100]}')
                # print('\n')
                # print(f'time of shape {np.shape(time)} has first 100 is {time[0:10]} last 100 is {time[-10:-1]}\n')

        
                # sys_time_end = (np.where(hk_time < time[-1]+0.01)[0][-1])
                # fig, (ax1, ax2) = plt.subplots(2, 1)
        
                # ax1.plot(air_smoothed, label = 'smoothed')
                # ax1.plot(no_smooth_interp)
                # # ax1.plot(time, t_obs[0, 0, 0, :], label='T_obs')
                # # ax1.set_xlabel('Timesteps')
                # ax1.set_ylabel('K')
                # ax1.legend()
                # ax1.set_title('Smoothed vs unsmoothed air temp over time')

                # ax2.plot(hk_air_temp)
                # ax2.plot(all_smooth)
                # ax2.plot(hk_air_temp[:sys_time_end])

                # # ax2.plot(time, air)
                # ax1.set_ylabel('K')
                # ax2.set_xlabel('Timesteps')
                # ax2.set_title('Hk air temp over time')
                # fig.savefig(f'{outfile_figs}/Smooth_vs_unsmooth_air_over_time')
                
                # print(f'TOD first 100 is {tod[0, 2, 26, 0:100]}\n')
                # print(f'TOD last 100 is {tod[0, 2, 26, 100:-100]}')
        exit()





if __name__ == "__main__":
        """
        Generating the plots of ground pickup in the TOD model.
        - Plot of TOD model through the pipeline normalization and calibration (pointing filter is not on)
        - Plot of just ground pickup modelled from convolution maps and ground modelled as a Black body
        """
        outfile_figs = Path(tod_gen_folder/'pipeline/figs/figs_low_level/Tground')

        files = ['_1_tsys', '_2_Leah', '_3_norm', '_4_calib']
        
        images = []
        minmax = []
        titles = []
        

        images_trest = []
        minmax_trest = []
        titles_trest = []
        for i in range(len(files)):
                
                print(i)
                
                file = f'co7_002911511{files[i]}.h5'


                tod_gen = TOD_Gen()
                params = ['time', 'freq_bin_centers', 'tod']
                # params = ['time', 'tod','freq_bin_centers', 'point_tel', 'Gain']
                
                if i > 0:
                        params = ['time', 'freq_bin_centers', 'tod', 'T_rest',  'hk_time', 'air_temp']
                #         params = ['time', 'Tsys', 'freq_bin_centers', 'tod', 'Gain', 'point_tel',  'T_rest', 'hk_time', 'hk_air_temp']
                        
                tod, params =  tod_gen.read_file(data_folder/file, model=True, parameter_list=params, units=False)

                tod = params['tod']
                
                if i > 0:
                        T_rest = params['T_rest']
                        # Gain = params['Gain']
                        # Tsys = params['Tsys']
                        

                        hk_air_temp = params['air_temp']
                        hk_time = params['hk_time']
                        hk_time = (hk_time - hk_time[0])*24*60*60

                        # point_tel = params['point_tel']

                        # az = point_tel[:, :, 0] # Current point_tel is [feed, time, az/el]
                        # el = point_tel[:, :, 1]
                        
                

                # correlated = params['correlated']
                # white_noise = params['white_noise']
                freqs = params['freq_bin_centers']

                time = params['time'] #Time 0 is 0.0 time -1 is 379.9808580195531
                time = (time - time[0])*24*60*60
                # print(f' Time is from {time[0]} to {time[-1]}')

                scale = 1/10000 # Scale for whitenoise
                
                
                plt.figure()
                # Trest_waterfall, trest_minmax = waterfall(T_rest[0]-m, len(d_model[0, 0, 0, :]), f"Ground pickup from map lin interpolated, mean removed", f'{outfile_figs}/Tground_no_corr_no_white_noise_offset', shape = 4*1024, minmax=True)
                
                model_waterfall, model_minmax = waterfall(tod[0], len(tod[0, 0, 0, :]), f"Waterfall plot of datamodel with white noise scaled by {scale}", f'{outfile_figs}/waterfall_datamodel_{i}', shape = 4*1024, minmax=True)
                
                if i < 2:
                        model_minmax = [None, None]
                if i > 0:
                        plt.figure()
                        # Trest_waterfall, trest_minmax = waterfall(T_rest[0], len(d_model[0, 0, 0, :]), f"Ground pickup from map lin interpolated", f'{outfile_figs}/Trest', shape = 1, minmax=True)
                        Trest_waterfall, trest_minmax = waterfall(T_rest[0], len(tod[0, 0, 0, :]), f"Ground pickup from map lin interpolated. Whitenoise scaled by {scale}", f'{outfile_figs}/Trest_{i}', shape = 4*1024, minmax=True)

                        # images_trest.append(Trest_waterfall)
                        # minmax_trest.append(trest_minmax)
                        # titles_trest.append(f"Ground pickup in model for file{files[i]}. Wn scaled by {scale}")

                images.append(model_waterfall)
                minmax.append(model_minmax)
                titles.append(f"TOD for file{files[i]}. Wn scaled by {scale}")
                
                if i == 3:
                        smooth, air_temp_interp = tod_gen.interpolating_air_temp(air_temp = hk_air_temp, sys_time = hk_time)
                        
                        print(f'Length of time {len(time)}, length of smooth is {len(smooth)}')
                        plt.figure()
                        # plt.plot(time, smooth)
                        plt.plot(hk_air_temp)
                        plt.title('House keeping air temp over the TOD time')
                        plt.savefig(f'{outfile_figs}/Air_temp_hk_over_time')

                


        plt.figure()
        sub_plot_TOD = subplot_waterfalls(images=images, titles=titles, minmax=minmax, save=f'{outfile_figs}/TOD_through_pipeline')
     
        
        exit()


if __name__ == "__main__":
        
 

        ############################################################
        tod_gen = TOD_Gen()
        

        params = ['time', 'Tsys', 'freq_bin_centers', 'tod', 'Gain', 'point_tel'] #, 'air_temp', 'hk_time']
     
        read_file_start = timer.time()
        tod, params =  tod_gen.read_file(data_folder/'co7_003949207.h5', model=True, parameter_list=params, units=False)

        read_file_end = timer.time()
        print(f'Time of reading files {read_file_end-read_file_start}')
        
        with open("params.npy", 'wb') as f:
                pickle.dump(params, f, protocol=4)

       
        ############################################################
        

        with open("params.npy", "rb") as f:
                params = pickle.load(f)
                
        params = dict(params.items())

        tod = params['tod']
        Gain = params['Gain']
        Tsys = params['Tsys']
        freqs = params['freq_bin_centers']
        time = params['time'] #Time 0 is 0.0 time -1 is 379.9808580195531
        time = (time - time[0])*24*60*60
        # print(f' Time is from {time[0]} to {time[-1]}')

        point_tel = params['point_tel']
        hk_air_temp = params['hk_air_temp']
        hk_time = params['hk_time']
        hk_time = (hk_time - hk_time[0])*24*60*60

        
        
        az = point_tel[:, :, 0] # Current point_tel is [feed, time, az/el]
        el = point_tel[:, :, 1]

        scale = 1/1000 # Scale for whitenoise

        exit()

        # az = az.T  # Reshaping [time] # DONT TRANSPOSE
        # el = el.T # Reshaping [time]
        extra_params_dict = {'hk_time': hk_time, 'hk_air_temp':hk_air_temp}

        tod_gen = TOD_Gen(tod=tod, az=az, el=el, Gain=Gain, Tsys=Tsys, time=time, freq_bin_centers = freqs, extra_params_dict=extra_params_dict) 
        # tod_gen = TOD_Gen(tod=tod, az=az, el=el, Gain=Gain, Tsys=Tsys, freq=None, time=time, sigma=None, units=False, freq_bin_centers = freqs) 
        air_temp_interpolated, non_smooth_air = tod_gen.interpolating_air_temp(air_temp=hk_air_temp, sys_time=hk_time)

        # brightness_temp_bb = tod_gen.tground_blackbody(air_temp_interpolated)
        

        convolution_files = [ conv_folder / 'conv_26GHz_NSIDE256.npy', conv_folder / 'conv_30GHz_NSIDE256.npy', conv_folder / 'conv_34GHz_NSIDE256.npy',]

        d_model, G, correlated, Tsys, white_noise, T_rest = tod_gen.get_data_model(tground_files=convolution_files, corr_noise=False, downsampled=False, point=[az, el], scale = scale)
        
        components = {'data model':d_model, 'Gain':G, 'Correlated noise':correlated, 'Tsys':Tsys, 'white noise':white_noise, 'T_rest':T_rest}
        # np.save("datamodel_components.npy", components)

        with open("datamodel_components.npy", 'wb') as f:
                pickle.dump(components, f, protocol=4)

        # comps = np.load('datamodel_components.npy')
        # sim = dict(comps.items())

        
        print(f'Shape of G is {G.shape}')
        m = np.mean(T_rest[0])

        # vmin = 'np.nanpercentile(np.abs(signal.flatten()), 2),'
        # vmax = 'np.nanpercentile(np.abs(signal.flatten()), 98),)'
        plt.figure()
        # Trest_waterfall, trest_minmax = waterfall(T_rest[0]-m, len(d_model[0, 0, 0, :]), f"Ground pickup from map lin interpolated, mean removed", f'{outfile_figs}/Tground_no_corr_no_white_noise_offset', shape = 4*1024, minmax=True)
        model_waterfall, model_minmax = waterfall(d_model[3], len(d_model[0, 0, 0, :]), f"Waterfall plot of datamodel with white noise scaled by {scale}", f'{outfile_figs}/waterfall_datamodel', shape = 4*1024, minmax=True)
        plt.figure()
        # Trest_waterfall, trest_minmax = waterfall(T_rest[0], len(d_model[0, 0, 0, :]), f"Ground pickup from map lin interpolated", f'{outfile_figs}/Trest', shape = 1, minmax=True)
        Trest_waterfall, trest_minmax = waterfall(T_rest[0], len(d_model[0, 0, 0, :]), f"Ground pickup from map lin interpolated. Whitenoise scaled by {scale}", f'{outfile_figs}/Trest', shape = 4*1024, minmax=True)
        
        plt.figure()
        plt.plot(air_temp_interpolated)
        plt.xlabel('time')
        plt.ylabel('air_temp_interpolated smoothed')
        # plt.legend()
        plt.savefig( f'{outfile_figs}/smooth_air_temp_overtime')
        
        # freqspace = np.linspace(26, 34, (4*1024))
        # plt.figure()
        # plt.plot(T_rest[0, 0, :, 10000])
        # plt.ylabel('K')
        # plt.xlabel('freq samples')
        # plt.savefig(f'{outfile_figs}/Tground_interp_check')

        # T_rest = T_rest.reshape(19, 4*1024, len(d_model[0, 0, 0, :]))

        # plt.figure()
        # plt.plot(freqspace, T_rest[0, :, 10000])
        # plt.ylabel('K')
        # plt.xlabel('GHz')
        # plt.savefig(f'{outfile_figs}/Tground_interp_check2')


        # with open("Model_params.txt", 'wb') as f:
        #         """
        #         Write a file containg the following:
        #         white noise: True
        #         correlated noise: None or Value
        #         T_rest: 

        #         """

        exit()
        """Normalized datamodel"""


        means = []
        for i in range(19):

                _mean = []
                for j in range(4):    
                    m_model =  data_model[i][j]
                    mean = tod_gen._lowpass_filter(m_model)

                    _mean.append(mean)
                # print(_mean.shape)
                means.append(_mean)

        means = np.array(means)
        print(np.all(np.isnan(means)))
        print(means.shape)


        data_nor = data_model / means - 1 # data_model/np.mean(data_model, axis=-1, keepdims=True) - 1 # 
        data_normalized, norm_minmax = waterfall(data_nor[0], len(data_model[0, 0, 0, :]), f"Waterfall plot of normalized data model", f'{tod_gen_folder}/figs/DELETE_d_norm_waterfall', shape = 4*64, minmax=True)
        mean_waterfall, mean_minmax  = waterfall(means[0], len(data_model[0, 0, 0, :]), f"Waterfall plot of mean data used in normalizing", f'{tod_gen_folder}/figs/DELETE_mean_waterfall', shape = 4*64, minmax=True)




        """ Polyfiltered datamodel """
        d_model = data_nor #data_model#.reshape(19, 4*64, 20175)
        data_mean = np.nanmean(d_model, axis=2, keepdims=True) 
        # data_p, _ = #tod_gen._polyfilter(d_model) # d_model - data_mean
        data_p = d_model - data_mean
        # print(f'Data mean shape is {data_mean[0, 0, 0, 0]}, d_model shape is {d_model[0, 0, 0, 0]}, {data_p[0]}')
        data_polyfiltered, p_minmax  = waterfall(data_p[0], len(data_model[0, 0, 0, :]), f"Waterfall plot of data model with mean f subtracted", f'{tod_gen_folder}/figs/DELETE_d_pfilter_waterfall', shape = 4*64, minmax=True)


        # SUBPLOT 
        subplot_norm = subplot_waterfalls([Trest_waterfall, model_waterfall, data_normalized, data_polyfiltered,  mean_waterfall], ['Waterfall plot of ground pickup from linear model','Waterfall plot of data model', 'Waterfall plot of normalized data model', 'Waterfall plot of data model after polyfilter and normalizing',  'Waterfall plot of mean data subtracted when normalizing'], minmax=[trest_minmax, model_minmax, norm_minmax, p_minmax, mean_minmax], ylabel="# Frequency sample", xlabel="# Time sample", colormap="RdBu_r", save=f'{tod_gen_folder}/figs/results/subplot_summary_T_linear')

    
