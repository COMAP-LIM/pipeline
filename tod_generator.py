import warnings
import h5py 
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
# from scipy.fftpack import rfft, irfft
# from scipy.fftpack import fft, ifft, rfft, irfft, fftfreq, rfftfreq, next_fast_len
from scipy.fftpack import  next_fast_len
import scipy as sp
from pathlib import Path
from astropy.io import fits
from astropy.wcs import WCS
from pathlib import Path
from numpy.fft import fft, ifft, rfft, irfft, fftfreq, rfftfreq

from astropy.modeling.models import BlackBody
import astropy.units as u

import time as timer
import sys


# from plot import * #***
np.random.seed(4)
"""
NOTES
- Use pep8 format
- Add a read the docs eventually


"""
file = Path(__file__)
src = Path(__file__).parent.absolute()
tod_gen_folder = src.parent.absolute()
data_folder = Path(tod_gen_folder/ 'data_tground')


# print(np.fft.rfft(np.zeros(100)).shape)

# x = np.zeros(100)
# print(rfft(x).shape, rfftfreq(100).shape)

  
class TOD_Gen:


    def __init__(self, tod=None, az=None, el=None, Gain=None, Tsys=None, freq=None, time=None, sigma=None, units=False, freq_bin_centers = None, extra_params_dict = None):
        """

        If you want to make up your own parameters, or change specific values you can do this here.
        tod(array, float):
            Time Ordered Data, typically read from a file.

        az(array, float):
            Azimuth direction of pointing. Typically read from file then used in initialization of class.
        
        el(array, float):
            Elevation direction of pointing. Typically read from file then used in initialization of class.
        
        Gain(array, float):
            Instrumental gain. Typically read from level1 or level2 files, and then used in initialization of class.

        Tsys(array, float):
            System temperature. Typically read from level1 or level2 files, and then used in initialization of class.

        freq(array, float):
            Frequency array. Typically read from file then used in initialization of class.

        time(array, float):
            Time array.Typically read from file then used in initialization of class.

        sigma(float):
            Sigma parameter value used to define the standard deviation of a Gaussion distribution.
            Ideally calculated and not given during intialization, but this is possible.
        
        extra_params_dict: 
            A dictionary which contains parameter names and values for other parameters you would like to add when you
            don't want to read them from a file. F.ex. if you already have a parameter file, or if you want to try out different values for a parameter.

            Example:
            In case of modelling the ground pickup as a black body, one requires the house keeping air temp and time. 
            In these cases I have a .npy parameter file saved which contains these values, and I wanted to use these instead of 
            continuosly using the read function which can take time.
        
        """
        self.tod = tod 
        
        self.time = time

        self.freq = freq

        self.Gain = Gain

        self.Tsys = Tsys

        self.sigma = sigma      #Should be found later, but if we want to try out different sigma we can

        self.az = az #*

        self.el = el #*

        self.freq_bin_centers = freq_bin_centers # ***

        if units:
            self.time = self.time * 24 * 60 * 60 # Turning time into seconds
            self.freq_bin_centers = self.freq_bin_centers *1e9
        
        if extra_params_dict:
            # print('(  .  _  . )')
            for name, item in extra_params_dict.items():
                setattr(self, name, item)
                


        
           
    def read_file(self, filename, model=False, parameter_list=['time', 'Tsys', 'freq_bin_centers', 'tod', 'Gain',  'el_az_amp', 'point_tel'], units=True):
        """
        Function to read datafile and update paramters

        parameter_list(list, str):  
            Send in a list of parameters that you want to extract from datafile

         
        
        Makes sure that params are overwritten if they have the same name.
        If read_file is ran again with new params and same output filename, the function should add to the filename without deleting the old params
        """

        if model:
            params = {}
            

            with h5py.File(filename, 'r') as infile:
                # print(f'{infile.keys()}')
             
                # Read parameters, checking if they are attributes or datasets
                for key in parameter_list:
                    # print(key)
                    setattr(self, key, infile[key][()])

                    params.update({f'{key}': infile[key][()]})
                    # print(key, 'updated')
                tod = infile['tod'][()]

                # setattr(self, 'hk_time', infile['hk_time'][()])
            
                # # print('hk_time', 'updated')
                # setattr(self, 'hk_air_temp', infile['air_temp'][()])
                
                # print('hk_air_temp', 'updated')
                # This is a must for the model
                # params.update({f'hk_air_temp': infile['air_temp'][()]})
                # params.update({f'hk_time': infile['hk_time'][()]})
                
            
            return tod, params
        else:
            with h5py.File(filename, 'r') as f:
                print(f' {f.keys()}')
        
                
            with h5py.File(filename, 'r') as infile:
                for key in parameter_list:
                    setattr(self, key, infile[key][()])
                tod = infile['tod'][()]
                    

            # print(f'TOD shape is {self.tod.shape}')
            if units:
                self.time = self.time * 24 * 60 * 60 # Turning time into seconds
                self.freq_bin_centers = self.freq_bin_centers *1e9


           
            return tod
        
 
    def write_to_file(self, outfile, directory):
        """
        Will save all the self variables, and all data model variables
        """

        #If filename does note exist at path -> create the file
        path = Path(directory)

        if not path.exists():
            raise NotADirectoryError(f'The path "{path}" does not exist.')
        
        file_path = path / Path(outfile)
        
        name = str(outfile)
        suffix = name.split('.')[-1]

        if suffix != 'h5':
            raise TypeError('Outfile type must be .h5')

        _, sigma = self.get_white_noise()
        setattr(self, 'sigma', sigma)

        with h5py.File(file_path, "w") as hdf5_file:
            for attr, value in self.__dict__.items():
                # Check the type and save it appropriately
                if isinstance(value, (int, float, str)):
                    hdf5_file.create_dataset(attr, data=value) # Only thing that worked
                    
                    #hdf5_file.attrs[f'{attr}'] = value
                elif isinstance(value, (list, np.ndarray)):
                    
                    hdf5_file.create_dataset(attr, data=value)

            d, G, correlated, Tsys, white_noise, T_rest = self.get_data_model()
            hdf5_file.create_dataset("model_d", data=d)
            hdf5_file.create_dataset("model_G", data=G)
            hdf5_file.create_dataset("model_correlated", data=correlated)
            hdf5_file.create_dataset("model_Tsys", data=Tsys)
            hdf5_file.create_dataset("model_white_noise", data=white_noise)
            hdf5_file.create_dataset("model_T_rest", data=T_rest)
            

        return 

    def get_white_noise(self, sideband=0, mu=0.0):
        """
        tsys(array, float):
            Tsys, system temperature. Typically an array of floating point values with len(time).
            Tsys is ideally collected from a datafile so that the standard devation sigma can be 
            extracted through the Radiometer equation.

        time(array, float):
            Has a 1D shape
        
        nu(array, float):
            Has a 2D shape, with a side band and its frequencies. From file is freq_bin_centers

        tod(array, float):
            The TOD gathered from datafile, or self-specified. Mainly here it is only needed to assure
            that the white noise generated has the same shape as the TOD.
        
        sideband(int):
            Normally a value between 0 and 3. Sideband contaning frequency values.

        mu(float):
            The mean of the Gaussian distribution which we use to creat the white noise.


        Output:
            white_noise(array, float):
                An array representing white noise. Same shape as TOD

        """
        start_wn = timer.time()
        # print('def get_white_noise')
        
        tsys = self.Tsys #[feed][sideband]
        # print(f'Tsys rand value {tsys[3][0][:]}')
        
        time = self.time 
        
        
        if self.freq is None:
            nu = self.freq_bin_centers 
            dnu = nu[sideband][1]-nu[sideband][0]
        else:
            nu = self.freq
            #dnu = nu[1]-nu[0]
            dnu = nu[sideband][1]-nu[sideband][0]
          
        dt = time[1] - time[0]          
        
        tod = self.tod 

        # Sigma gain             
         
        if self.sigma is None:             
            sigma = 1/np.sqrt( dt * dnu  )  # 
        else:
            sigma = self.sigma  

        # print(f'Sigma {self.sigma}')

        # feeds, x, y, z  = tod.shape 

        white_noise = sigma*np.random.normal(mu, 1, tod.shape)

        end_wn = timer.time()

        wn_time = end_wn - start_wn
        # print(f'Def get_white_noise time: {wn_time} \n')
        return white_noise, sigma
    
    def calculate_PS(self, data):
        """
        Data needs shape TOD for a specific feed.
        """
        # print('def calculate_PS')
        Ps_start = timer.time()
        white_noise, _ = self.get_white_noise()
        
        ps = np.abs(np.fft.rfft(data)**2)[1:]/data.shape[-1]
        fft_freq = np.fft.rfftfreq(white_noise.shape[-1])[1:]

        # fft_freq = np.fft.rfftfreq(white_noise.shape[-1])
        # ps = np.abs(np.fft.rfft(data)**2)/len(data) 
        
        # fft_freq = np.fft.fftfreq(white_noise.shape[-1])
        # ps = np.abs(np.fft.fft(data)**2)/len(data) 
        Ps_end = timer.time()

        PS_time = Ps_end - Ps_start
        # print(f'Def calculate_PS time: {PS_time} \n')
        return ps, fft_freq
    
    def get_one_over_f_noise(self, wn, sigma, f = None, feed=0, alpha=-1.8, f_knee=0.6, channel=11, sideband=0):
        """
        
        
        """
        # print('def get_one_over_f_noise')
        one_f_start = timer.time()
        if f is None:
            dt = self.time[1] - self.time[0]
            Ntime = len(self.time)
            f = np.fft.fftfreq(Ntime, dt)
            # print('f shape ', f.shape, '\n')
            # print('Ntime is ', Ntime)

        if sideband is not None:
            tod = self.tod[feed][sideband][channel]
        else:
            tod = self.tod[feed][channel] 
         


        Ntod = len(tod)  # Bc we calculated PS with real fft

        # if wn:
        white_noise_all = wn 
        sigma0 = sigma
        # else:
        #     white_noise_all, sigma0 = self.get_white_noise()
        # print(f'Sigma0 = {sigma0}')

        # sigma0 = sigma0/(np.nanmean(self.tsys)) # Using sigma gain and not tsys sigma from radiometer eq
        psd_corr = sigma0 ** 2 * (np.abs(f) / f_knee) ** alpha
        psd_corr[f == 0] = 0 # Removing the singular frequency

        # Generate 1/f correlated noise from generated white noise array 
        correlated_noise = np.random.normal(0, 1, Ntod)
        fft_noise = np.fft.fft(correlated_noise, axis = -1)

        # Inverse transform to get the correlated noise
        correlated_noise = np.fft.ifft(np.sqrt(psd_corr) * fft_noise, axis = -1).real

        PS, freq = self.calculate_PS(correlated_noise)

        # print('Is all of TOD nan? ', np.isnan(tod).all())
        # print('Is all of PS nan? ', np.isnan(PS).all())
        # print('sqrt(PS) shape ', np.sqrt(PS).shape, 'tod shape ', tod.shape, 'freq shape ', freq.shape)

        # Ntod = (len(tod))//2+1  # Bc we calculated PS with real fft
        # one_over_f_noise = irfft(np.random.normal(0, np.sqrt(PS), Ntod-1)) # -1 bc we skipped the first data value in calculating PS

        one_f_end = timer.time()
        one_f_time = one_f_end - one_f_start
        # print(f'Def get_one_over_f_noise time: {one_f_time} \n')
        return correlated_noise, self.time, f

    def get_pointing(self, feed = 0):
        """
        Pointing
        """
        pointing_start = timer.time()
        if self.az is None: # In case of reading from a file where the parameters point_tel have been asked to be saved 
            point_az = self.point_tel[feed][:, 0]
            point_el = self.point_tel[feed][:, 1]
            # print(f'Az and el for feed {feed}')
        else:  # In case when az and el are given from init of class TOD_Gen
            point_az = self.az
            point_el = self.el
        
        pointing_end = timer.time()
        pointing_time = pointing_end - pointing_start
        # print(f'Def get_pointing time: {pointing_time} \n')

        return point_az, point_el
      
    def downsampler(self, data, sigma, shape=(19, 4, 64), downsample=16):
        data_down = np.zeros(shape) # New array with downsampled data

        for f in range(len(data)):
            for i in range(len(data[0][:][:])):
                for j in range(0, len(data_down[0][0][:])):
                    # print(f, i, j)
                    start_idx = j*downsample
                    end_idx = start_idx + downsample
                

                    #weights
                    weights = 1 / sigma**2 #1/sigma**2 weighting
                    # print('sigma[i, start_idx:end_idx] = ', sigma[i, start_idx:end_idx])
                    # weights = 1/sigma**2
        
                    # print('weights = ', weights)

                    sum_weights = np.sum(weights, axis=0) 
                    # print('sum weights = ', sum_weights)

                    # print(f'len of sum weights {len(sum_weights)}')
                    
                    # print('data[i, start_idx:end_idx, :]', data[i, start_idx:end_idx, :].shape)
                    # Weighted average of data
                    weighted_sum = np.sum(data[f, i, start_idx:end_idx] * weights, axis=0)
                    # print('weighted sum shape', weighted_sum.shape) 

                    data_down[f, i, j] = weighted_sum / sum_weights
                
        return data_down
    
    def get_gain(self, downsampled=False):

        """
        Gain is retrieved from datafile.
        Gain shape is (19, 4, 1024)
        
        """
        # print('def get_gain')
        def downsampler(data, sigma, shape=(19, 4, 64), downsample=16):

            data_down = np.zeros(shape) # New array with downsampled data

            for f in range(len(data)):
                for i in range(len(data[0][:][:])):
                    for j in range(0, len(data_down[0][0][:])):
                        # print(f, i, j)
                        start_idx = j*downsample
                        end_idx = start_idx + downsample
                    

                        #weights
                        weights = 1 / sigma**2 #1/sigma**2 weighting
                        # print('sigma[i, start_idx:end_idx] = ', sigma[i, start_idx:end_idx])
                        # weights = 1/sigma**2
            
                        # print('weights = ', weights)

                        sum_weights = np.sum(weights, axis=0) 
                        # print('sum weights = ', sum_weights)

                        # print(f'len of sum weights {len(sum_weights)}')
                        
                        # print('data[i, start_idx:end_idx, :]', data[i, start_idx:end_idx, :].shape)
                        # Weighted average of data
                        weighted_sum = np.sum(data[f, i, start_idx:end_idx] * weights, axis=0)
                        # print('weighted sum shape', weighted_sum.shape) 

                        data_down[f, i, j] = weighted_sum / sum_weights
                    
            return data_down
        

        if downsampled:
            gain = downsampler(data = self.Gain, sigma = self.sigma)
            # print('Gain shape is ', gain.shape)
            # print('DOWNSAMPLING')
            return gain

        # print(self.Gain)
       
        return self.Gain
          
    def get_data_model(self, tground_files=None, corr_noise=True, downsampled=True, point = None, hk_time = None, hk_air_temp = None, scale = 1, constant_ground = False, BB_test=False):

        data_model_start = timer.time()
        # print(f'TOD shape is {self.tod.shape}')
        # print('def get_data_model')
        white_noise, sigma = self.get_white_noise()
        

        if downsampled:
            Tsys = self.downsampler(data=self.Tsys, sigma=sigma)
            G = self.get_gain(downsampled=True)
            freqs = 4*64  
        else:
            Tsys = self.Tsys
            freqs = self.tod.shape[1]*self.tod.shape[2]
            G = self.Gain
          

        if tground_files: # If ground pickup is to be found from reading a convolution file  
            tground_start = timer.time() 

            if hk_time is None:
                hk_air_temp = self.hk_air_temp
                hk_time = self.hk_time
            # print('about to open ground files ... ')
            
            Tground = self.tground_interpolation(tground_files, nfreq=freqs, pointing = point) # shape feed, freq, time
            
            if constant_ground is False:
                # print(f'self.hk_air_temp has shape {np.shape(hk_air_temp)}, self.hk_time has shape {np.shape(hk_time)}')
                air_temp_interp, _, _ = self.interpolating_air_temp(air_temp=hk_air_temp, sys_time=hk_time, hk_start=self.scan_start_idx_hk, hk_end=self.scan_stop_idx_hk)

                if BB_test:
                    # air_temp_interp = np.zeros_like(air_temp_interp) + 300
                    Blackbody_factor = (air_temp_interp+ 273.15)/300 #self.tground_blackbody(air_temp_interp)/300 # Dividing by 300 as this is the assumed temp
                    # print(f'Blackbody_factor has shape {np.shape(Blackbody_factor)}')
                    # print(f"Tground shape is {np.shape(Tground)}")
                    Blackbody_factor = Blackbody_factor[None, None, ...]
                    Tground = Blackbody_factor*Tground
                    
                    
                if BB_test == False:
                    Blackbody_factor = self.tground_blackbody(air_temp_interp)/300 # Dividing by 300 as this is the assumed temp
                    # print(f'Blackbody_factor has shape {np.shape(Blackbody_factor)}')
                    Tground = Blackbody_factor*Tground

                # print(f'Tground has shape {np.shape(Tground)}')

                T_rest = Tground

            if constant_ground:
                T_rest = Tground

            tground_end = timer.time()
            tground_time = tground_end - tground_start
            # print(f'Def tground_files time: {tground_time} \n')
        else:  # If ground pickup instead is predetermined, as from file written by the TOD_Gen class
            print('No tground files found')
            Tground, _ = self.azimuth_template(az = self.az)
            
            # print(f'TGROUND SHAPE IS {np.shape(Tground)}')
            T_rest = Tground  # This will change when there are more components to add
      
        """
        Ensuring that the components of the data model all have the same shapes, so that the datamodel equation can be computed
        """
        components = {"Gain":G, "Tsys":Tsys, "white_noise":white_noise, "T_rest":T_rest}  #Dictionary of components
        # print(f'SHAPE OF T_REST IS {T_rest.shape}')
        for component, variable in components.items(): # Component is name of component, variable is the component data
            
            if variable.ndim < self.tod.ndim:     
                var_dim = [dim for i, dim in enumerate(variable.shape)] # Making a list of the dimensions of the variable. Ex. tod would be [feeds, sidebands, freqs, time]
                tod_dim = [dim for i, dim in enumerate(self.tod.shape)] # Same with tod

                var_dim = np.array(var_dim) # Making the list arrays
                tod_dim = np.array(tod_dim)

                for i in range(len(tod_dim)):  # This loop compare the dimensions of the variable and makes sure to insert an empty axis in order to make all components compatible for
                                                # multiplication in the datamodel
                      
                        if i >= len(var_dim) and variable.ndim != self.tod.ndim:
                                # print(f'i is {i} and len(var_dim) is {len(var_dim)}')
                                variable = variable[..., None]
                                

                        elif i < len(var_dim) and var_dim[i] != tod_dim[i]:
                                # print(f'var_dim[i] {var_dim[i]} and tod_dim[i] {tod_dim[i]}\n')
                                variable = variable[..., None, :]
                                # print(variable.shape)

                        # print(f'for i = {i} {component}, {variable.shape}, len of tod {len(tod_dim)}')
                components[component] = variable
            # print(f'{component}, {variable.shape} \n')
    


        
        
        G = components['Gain']
        Tsys = components['Tsys']
        T_rest = components['T_rest']
        white_noise = components['white_noise']*scale


        if corr_noise: # Adding correlated noise in the low-level simulation is optional
            correlated, _, _ = self.get_one_over_f_noise(wn = white_noise, sigma = sigma)
            correlated = correlated[None, None, None, :]

            d = G*(1+correlated)*(Tsys + Tsys*white_noise + T_rest)
    
        else:
            d = G*(Tsys + Tsys*white_noise + T_rest)
            correlated = None
 

        data_model_end = timer.time()
        data_model_time = data_model_end - data_model_start
        # print(f'Def get_data_model time: {data_model_time} \n')
       
        return d, G, correlated, Tsys, white_noise, T_rest
        
    def azimuth_template(self, az = None, az_0 = None, K=0.5, feed=0, sideband=0, channel=11):
        """
        - make azimuth template for tground
        d_pointing = g/(sin(el(t))) + A *az(t) + B + n

        T_ground = 0.5K * (az - az_0)

        in h5 file:
        el_az_amp
        
        """
        tod = self.tod
 
        d_T_ground = np.zeros((np.shape(self.tod)))
        print(f'TOD shape is {np.shape(self.tod)}')
        for feed in range(len(tod)):
            print(f'feed {feed}')

            az_0 = np.nanmean(az[feed])
            # print(f'Shape az_0 {np.shape(az_0)}, shape az {np.shape(az)}')
            d_T_ground[feed] = K*(az[feed]-az_0) 



        if az is None:
            for feed in range(len(tod)):
                print(f'feed {feed}')
                az, _ = self.get_pointing(feed = feed)
                az_0 = np.nanmean(az)
                print(f'Shape az_0 {np.shape(az_0)}, shape az {np.shape(az)}')
                d_T_ground[feed] = K*(az-az_0) 

        return d_T_ground, az

    def newprojplot_with_sensible_units(self, az_deg, el_deg, **kwargs):
        """
        ***
        This function is written by Nils Ole Stutzer

        Is used in fits_file_ground
        """
        return hp.newprojplot(
            theta = np.pi / 2 - np.radians(el_deg), 
            phi = np.radians(az_deg) - (2 * np.pi * (az_deg > 180).astype(np.int32)), 
            **kwargs,
        )
        
    def fits_file_ground(self, filename, pointing = None, scale=300, outfile=f'{tod_gen_folder}/figs/testing/correct_sigma/healpy_TOD_of_ground_pickup2_from_map', file_fits=True):
        """
        !!!! ISN'T IN THE MODEL ***

        Is used to make horizon plots of the convolution maps
        
        """
        
        # m = hp.read_map(filename)

        if pointing is None:
            az, el = self.get_pointing(feed=0)
        else:
            az, el = pointing

        # 
        # print('def fits_file_ground')
        # print('val.shape', val.shape)

        ground_profile = np.loadtxt(data_folder/'Horizon_hwt.txt').T
        path = filename
        
        if file_fits:
            m = hp.read_map(path)
            val = scale*hp.pixelfunc.get_interp_val(m, az, el, nest=False, lonlat=True)

            fig = plt.figure()
            img = hp.projview(scale * m, cmap ="inferno", fig = fig, min = 0, max = 8, xsize = 200, projection_type="cart", graticule_labels=True, graticule_color="lightgray", graticule = True, rot_graticule = False, rot = [90, 0], custom_ytick_labels = [None, -90, -60, -30, 0, 30, 60, 90], custom_xtick_labels = [None, 0, 60, 120, 180, 240, 300, 360], unit="Brightness [K]", reuse_axes = True, cb_orientation="vertical")

            self.newprojplot_with_sensible_units(ground_profile[0]+180, ground_profile[1][::-1], color = "lime", lw = 2)

        else:
            m = np.load(path)
            m[m > 0.90] = np.nan
            # nside = hp.npix2nside(m.size)
            # m = np.arrange(m.size)
            val = 'N/A'
        
            fig = plt.figure()
            img = hp.projview(scale * m, cmap ="inferno", fig = fig, min = 0, max = 8, xsize = 200, projection_type="cart", graticule_labels=True, graticule_color="lightgray", graticule = True, rot_graticule = False, rot = [90, 0], custom_ytick_labels = [None, -90, -60, -30, 0, 30, 60, 90], custom_xtick_labels = [None, 0, 60, 120, 180, 240, 300, 360], unit="Brightness [K]", reuse_axes = True, cb_orientation="vertical")

            self.newprojplot_with_sensible_units(ground_profile[0]+180, ground_profile[1][::-1], color = "lime", lw = 2)


        ax = plt.gca()
        ax.set_ylabel("Elevation [deg]")
        ax.set_xlabel("Azimuth [deg]")

        # fig = plt.figure()
        # hp.projview(300 * m, cmap ="inferno", fig = fig, min = 0, max = 8, xsize = 2000, projection_type="cart")
        # hp.newprojplot(theta = np.pi / 2 - np.radians(el), phi = np.radians((az + 360) % 360), color = "g")

        # # hp.newprojplot(phi = az, theta = np.deg2rad(el), color = "g", lonlat = True)
        plt.figtext(0.5, 0.02, f"Temperature map of ground pickup. Horizon is blocked out in grey.", ha="center", fontsize=10, wrap=True)
        plt.savefig(outfile)

        return val, img, [0, 8] # *** make this subplottable

    def tground_interpolation(self, freq_files, nside=256, pointing=None, nfreq=4*1024):
        """
        freq_files is a list of maps to interpolate over

        scp leaheh@stjerne16.uio.no:/mn/stornext/d16/cmbco/comap/jonas/analysis/beam/conv_26GHz_NSIDE256.npy \
            leaheh@stjerne16.uio.no:/mn/stornext/d16/cmbco/comap/jonas/analysis/beam/conv_30GHz_NSIDE256.npy \
            leaheh@stjerne16.uio.no:/mn/stornext/d16/cmbco/comap/jonas/analysis/beam/conv_34GHz_NSIDE256.npy \
            /Users/lh/Documents/Uni/THE_MASTER/MASTER/tod_gen/data

        /mn/stornext/d16/cmbco/comap/jonas/analysis/beam/conv_26GHz_NSIDE256.npy
        /mn/stornext/d16/cmbco/comap/jonas/analysis/beam/conv_30GHz_NSIDE256.npy
        /mn/stornext/d16/cmbco/comap/jonas/analysis/beam/conv_34GHz_NSIDE256.npy
        """
        if pointing is None:
            az, el = self.get_pointing()

        else:
            az, el = pointing
        N_feed, N_time = az.shape


        maps_at_pointing = [] # 3 maps in total: one at 26GHz, one at 30GHz and one at 34GHz
        for file in freq_files:
        
            m = np.load(file)
            m_tod = hp.pixelfunc.get_interp_val(m, az, el, nest=False, lonlat=True)

            # m_at_pointing = [ m[pixel_index[i]] for i in range(len(pixel_index)) ]

            maps_at_pointing.append(m_tod)

            # print(len(m_at_pointing), m_at_pointing[:20])
        maps_at_pointing = np.array(maps_at_pointing)
       
        interp_maps = np.zeros((nfreq, N_feed, N_time))  # Shape is (freq, feed, time)
        
        frequencies = np.linspace(26, 34, nfreq)
        divider = int(nfreq/4)
        
        for i in range(N_feed):
            for j in range(N_time):                              
                interp_maps[:, i, j] = np.interp(frequencies, np.array([26, 30, 34]), maps_at_pointing[:, i, j])
        
        #from frq, feed, time -> feed, freq, time
        interp_maps = np.array((interp_maps[:divider, :], interp_maps[divider:2*divider, :], interp_maps[2*divider:3*divider, :], interp_maps[3*divider:, :]))
        
        interp_maps = interp_maps.transpose(2, 0, 1, 3)*300 # The convolution maps are in normalized units, where 300 is assumed to be the ground temp

        

        return interp_maps

    def interpolating_air_temp(self, air_temp, sys_time, hk_start, hk_end, tol = 0.01):
        """
        Interpolates air temp array to have the same time steps as the TOD.

        Air temp is the hk_data air temperature.
        Sys time is the time recorded for the hk air temp. This has a larger timestep than the observation time, and thus needs interpolating.

        hk_start and hk_end is the hk time index for when the observation started and ended in the hk time array.

        NB: Make sure to distinguish between system time, the time at which the air temp was gathered, and the TOD time sampling
        """
       #****
        time_array = self.time
        

        # print(f'Initial time is {time_array[0]}, intial sys time is {sys_time[hk_start]}')
        # print(f'Time shape is {np.shape(time_array)}. The time array every 1000 times {time_array[::1000]}')
        # print(f'Sys time shape is {np.shape(sys_time)}. Sys time in obs range every 100 times{(sys_time[hk_start:hk_end:100])}')
        
        # Find out where system time is time[-1], and then interpolate that section of system time
        # sys_time_end1 = np.where(sys_time < (time_array[-1]+tol))[0][-1] #sys_time[sys_time==time_array[-1]]

        sys_time_test = np.where(sys_time < (time_array[-1]+tol))
        # print(f'Sys_time_test is {sys_time_test}')

        sys_time_end = np.where(sys_time < (time_array[-1]+tol))[0][-1]


        # smooth_air = sp.ndimage.gaussian_filter1d(air_temp, sigma=900) # Smoothing 15 min
        smooth_air = sp.ndimage.gaussian_filter1d(air_temp, sigma=200) # Smoothing less 15 min
        interp_sys_time = np.arange(len(sys_time[hk_start:hk_end]))
        
        air_temp = air_temp[hk_start:hk_end]
      
      

        original_air_temp_interp = np.interp(time_array, interp_sys_time, air_temp) #Unsmoothed interpolation, for comparison
        air_temp_interpolated = np.interp(time_array, interp_sys_time, smooth_air[hk_start:hk_end]) #Smoothed and then interpolated, used in datamodel


        # interpolated and smoothed from scan, all hk air temp smoothed, interpolated but not smoothed in scan
        return air_temp_interpolated, smooth_air, original_air_temp_interp     

    def tground_blackbody(self, air_temp_interpolated, test=False):
        """
        T_obs is meant to be divided by 300 and multiplied to specific convolution map values in datamodel
        """
        from astropy.constants import c, k_B
        # Given air temp 

        if self.freq is None:
            freqs = (self.freq_bin_centers*1e9)
        else:
            freqs = self.freq
        freqs = freqs * u.Hz
       
        T_obs = np.zeros(np.shape(self.tod[0]))
        
        # Must give air_temp_interpolated class Quantity and units K
        air_temp_interpolated = (air_temp_interpolated + 273.15)* u.K
        dirt_temp = air_temp_interpolated 
        
        bb = BlackBody(temperature=dirt_temp[None, None, :])
        T = bb(freqs[..., None]).to(u.K, equivalencies= u.brightness_temperature(freqs[..., None]))
        # T = (c**2 * I)/(2*k_B*freqs[..., None]**2)
        
        T_obs = T[None, ...]
        
        return T_obs.value#ground_pickup_map