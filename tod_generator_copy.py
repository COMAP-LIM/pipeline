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
data_folder = Path(tod_gen_folder/ 'data')


# print(np.fft.rfft(np.zeros(100)).shape)

# x = np.zeros(100)
# print(rfft(x).shape, rfftfreq(100).shape)


class TOD_Gen:
    """
    - Needs to read TOD from comap data and read and store all the key files that fit variable names

    - *** Add a get more samples from same data function? Use bootstrap once you have used TOD_gen to gather parameter info from enough datafiles

    Example run:

    TOD_Gen(time_array, tod_array, tsys_array, sigma_value)
    
    """

    def __init__(self, tod=np.zeros(10), az=np.zeros(10), el=np.zeros(10), Gain=np.zeros(10), Tsys=np.zeros(10), freq=np.zeros(10), time=np.zeros(10), sigma=None):
        """
        PLACEHOLDER UNTIL I GET SOMETHING BETTER.

        ***  Make sure that useful parameters can be changed  ***
   

        If you want to make up your own values, or change specific values you can do this here.
        
        time(array, float):
            Time array

        tsys(array, float):
            An array of len(time) containing Tsys data.

        sigma(int):
            Sigma parameter value used to define the standard deviation of a Gaussion distribution.
            Ideally calculated and not given.
        
        nu(array, float):
            Frequency array. Ideally found in file.
        """
        self.tod = tod 
        
        self.az = az

        self.el = el

        self.time = time

        self.freq = freq

        self.Gain = Gain

        self.Tsys = Tsys

        self.sigma = sigma      #Should be found later, but if we want to try out different sigma we can
           
    def read_file(self, filename, parameter_list=['time', 'Tsys', 'freq_bin_centers', 'tod', 'Gain',  'el_az_amp', 'point_tel'], model=False):
        """
        Function to read datafile and update paramters

        parameter_list(list, str):  
            Send in a list of parameters that you want to extract from datafile
        
        """
        # print('def read_file')

        if model == False:
            with h5py.File(filename, 'r') as f:
                print(f' {f.keys()}')
        
                
            with h5py.File(filename, 'r') as infile:
                for key in parameter_list:
                    setattr(self, key, infile[key][()])
                tod = infile['tod'][()]
                    

            print(f'TOD shape is {self.tod.shape}')
            self.time = self.time * 24 * 60 * 60 # Turning time into seconds
            self.freq_bin_centers = self.freq_bin_centers *1e9


            """
            TODO: Write to file
            - Make sure that params are overwritten if they have the same name.
            - If read_file is ran again with new params and same output filename, the function should add to the filename without deleting the old params
            """
            return tod
        
        if model:
            # with h5py.File(filename, 'r') as f:
            #     print(f'{f.keys()}')
            
            params = []
            with h5py.File(filename, 'r') as infile:
                # for key in parameter_list:
                #     setattr(self, key, infile[key][()])

                # Read parameters, checking if they are attributes or datasets
                for key in parameter_list:
                    if key in infile.attrs:
                        setattr(self, key, infile.attrs[key])  # Read from attributes
                    else:
                        setattr(self, key, infile[key][()])  # Read from datasets
                    
                    tod = infile['tod'][()]
                    setattr(self, 'tod', infile['tod'][()])
                    params.append(infile[key][()])
           
            # 
            # for param in parameter_list:
            #     params.append(f'self.{param}')
            
            # params = np.array(params)
            # params = np.float(params)

            return tod, params
    
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

        # if not file_path.exists():
        #     h5py.File(file_path, 'x')
        
        # Write an hd5 file containing all self params, and datamodel params
        # Write out all self variables
        # Write out datamodel variables last
        # tod = self.read_file(filename=path)
        _, sigma = self.get_white_noise()
        setattr(self, 'sigma', sigma)
      
        # print(f"Sigma added to __dict__: {self.__dict__.keys()}")
        
        # print(self.__dict__.keys())
        with h5py.File(file_path, "w") as hdf5_file:
            for attr, value in self.__dict__.items():
                # Check the type and save it appropriately
                if isinstance(value, (int, float, str)):
                    hdf5_file.create_dataset(attr, data=value) # Only thing that worked
                    #hdf5_file.attrs[f'{attr}'] = value
                elif isinstance(value, (list, np.ndarray)):
                    
                    hdf5_file.create_dataset(attr, data=value)
                
            #hdf5_file.attrs['sigma'] = sigma
            # hdf5_file.create_dataset("sigma", data=sigma)
            d, G, correlated, Tsys, white_noise, T_rest = self.get_data_model()
            hdf5_file.create_dataset("model_d", data=d)
            hdf5_file.create_dataset("model_G", data=G)
            hdf5_file.create_dataset("model_correlated", data=correlated)
            hdf5_file.create_dataset("model_Tsys", data=Tsys)
            hdf5_file.create_dataset("model_white_noise", data=white_noise)
            hdf5_file.create_dataset("model_T_rest", data=T_rest)
            

        return 

    def get_white_noise(self, tsys=None, time=None, nu=None, dnu = None, tod=None, feed = 0, sideband=0, mu=0.0):
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
        print('def get_white_noise')
        if tsys is None:
            tsys = self.Tsys #[feed][sideband]
            # print(f'Tsys rand value {tsys[3][0][:]}')
            

        if time is None:
            time = self.time 

        if nu is None:
            nu = self.freq
            print(self.freq.shape)
            dnu = nu[1]-nu[0]
            
            if self.freq is None:
                nu = self.freq_bin_centers 
                dnu = nu[sideband][1]-nu[sideband][0]
          
        dt = time[1] - time[0]
        if tod is None:            
            tod = self.tod 

        


        # Sigma gain                             
        self.sigma = 1/np.sqrt( dt * dnu  )  # *** Is nanmean tsys okay? 

        # print(f'Sigma {self.sigma}')

        # feeds, x, y, z  = tod.shape 

        white_noise = self.sigma*np.random.normal(mu, 1, tod.shape)


        return white_noise, self.sigma
    
    def calculate_PS(self, data):
        """
        Data needs shape TOD for a specific feed.
        """
        print('def calculate_PS')
        white_noise, _ = self.get_white_noise()
        
        ps = np.abs(np.fft.rfft(data)**2)[1:]/data.shape[-1]
        fft_freq = np.fft.rfftfreq(white_noise.shape[-1])[1:]

        # fft_freq = np.fft.rfftfreq(white_noise.shape[-1])
        # ps = np.abs(np.fft.rfft(data)**2)/len(data) 
        
        # fft_freq = np.fft.fftfreq(white_noise.shape[-1])
        # ps = np.abs(np.fft.fft(data)**2)/len(data) 
      
        return ps, fft_freq
    
    def get_one_over_f_noise(self, f = None, feed=0, alpha=-1.8, f_knee=0.6, channel=11, sideband=0):
        """
        
        
        """
        print('def get_one_over_f_noise')
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
        white_noise_all, sigma0 = self.get_white_noise()
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

        return correlated_noise, self.time, f

    def get_pointing(self, feed = 0):
        """
        Pointing
        """

        point_az = self.point_tel[feed][:, 0]
        point_el = self.point_tel[feed][:, 1]

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
    
    def get_gain(self, downsmapled=False):

        """
        Gain is retrieved from datafile.
        Gain shape is (19, 4, 1024)
        
        """
        print('def get_gain')
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
        

        if downsmapled:
            gain = downsampler(data = self.Gain, sigma = self.sigma)
            # print('Gain shape is ', gain.shape)
            return gain

        # print(self.Gain)
       
        return self.Gain
          
    def get_data_model(self, tground_files=None, corr_noise=True):

        print('def get_data_model')
        correlated, _, _ = self.get_one_over_f_noise()
        white_noise, sigma = self.get_white_noise()

        

        if tground_files:
            Tground = self.tground_interpolation(tground_files, nfreq=4*64).T
            Tground = Tground.reshape(4, 64, 20175)
            
            T_rest = Tground[ None, :, :]   

            print(f'Shape of Tground is {T_rest.shape}')

            # exit()
        else:
            Tground, az = self.azimuth_template()
            T_rest = Tground  #++   
            T_rest = T_rest[None, None, None, :]


        G = self.get_gain(downsmapled=True)
        #G = np.nan_to_num(G, nan=0)

        # print(f'Gain max is {np.max(G)}, nanmean G is {np.nanmean((G))}')
        # T_sys = (self.Tsys).reshape(19, 4*1024) # *** 

        Tsys = self.downsampler(data=self.Tsys, sigma=sigma)
        #Tsys = np.nan_to_num(Tsys, nan=0)
        Tsys = Tsys[:, :, :, None] 
        
        white_noise = white_noise[:, :, :, :] 
        G = G[:, :, :, None] 

        correlated = correlated[None, None, None, :]
                    

        # data model d
        # print(f'Correlated noise shape {correlated.shape} \nWhite noise shape {white_noise.shape} \nGain shape is {G.shape}\nTsys is {Tsys.shape} \nT_rest is {T_rest.shape}')
        if corr_noise:
            d = G*(1+correlated)*(Tsys + Tsys*white_noise + T_rest)
        else:
            d = G*(Tsys + Tsys*white_noise + T_rest)
 
        # print("d", d)
        # print("G" ,G)
        # print("Tsys", Tsys)
        # print("T", T_rest)
       
        return d, G, correlated, Tsys, white_noise, T_rest
    
    def _mapmaker(self, tod, sigma, pointing_idx): 
        """
        *** STRAIGHT UP COPIED FROM PROJECT WORK, will need some editing ***



        Input: Downsampled tod data and tod uncertainty, pick one band.
                TOD and sigma must be 1d arrays
                
        Output: signal map, uncertainty map and hit count map
        """

        n_pix = 120 * 120  # pix length

        signal_map = np.zeros(n_pix)
        sigma_map = np.zeros(n_pix)
        # hit_count_map = np.zeros(n_pix)

        hit_count_map, _ = np.histogram(pointing_idx, bins = np.arange(n_pix+1))
        
        hit_pix = np.unique(pointing_idx)
        # print(f'The tod shape is {tod.shape}, \n')
        for pix in hit_pix:
            idx_tod = np.where(pointing_idx==pix)[0]
            d_array = np.array([tod[idx] for idx in idx_tod ])
            sigma_array = np.array([sigma[idx] for idx in idx_tod ])

            numerator = np.sum(d_array/sigma_array**2)
            denominator = np.sum(1/sigma_array**2)
            signal_map[pix] = numerator/denominator
            sigma_map[pix] = 1/np.sqrt(np.sum(1/sigma_array**2))#/len(idx_tod) ### *** Divide by len okay???

        
        return signal_map, sigma_map, hit_count_map

    def azimuth_template(self, az = None, az_0 = None, K=0.5, feed=0, sideband=0, channel=11):
        """
        - make azimuth template for tground
        d_pointing = g/(sin(el(t))) + A *az(t) + B + n

        T_ground = 0.5K * (az - az_0)

        in h5 file:
        el_az_amp
        
        """
 
        if az is None:

            az, _ = self.get_pointing(feed = feed)

        if az_0 is None:
            az_0 = np.nanmean(az)

        d_T_ground = K*(az-az_0) 

        return d_T_ground, az

    def newprojplot_with_sensible_units(self, az_deg, el_deg, **kwargs):
        """
        ***
        Written by Nils Ole Stutzer
        """
        return hp.newprojplot(
            theta = np.pi / 2 - np.radians(el_deg), 
            phi = np.radians(az_deg) - (2 * np.pi * (az_deg > 180).astype(np.int32)), 
            **kwargs,
        )
        
    def fits_file_ground(self, filename, pointing = None, scale=300, outfile=f'{tod_gen_folder}/figs/testing/correct_sigma/healpy_TOD_of_ground_pickup2_from_map', file_fits=True):
        """
        !!!! ISN'T IN THE MODEL ***
        
        """
        
        # m = hp.read_map(filename)

        if pointing is None:
            az, el = self.get_pointing(feed=0)
        else:
            az, el = pointing

        # 
        print('def fits_file_ground')
        # print('val.shape', val.shape)

        ground_profile = np.loadtxt(tod_gen_folder/'data/Horizon_hwt.txt').T
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
            az, el = self.get_pointing(feed=0)

        else:
            az, el = pointing

        theta = np.pi / 2 - np.radians(el)
        phi = np.radians(az) - (2 * np.pi * (az > 180).astype(np.int32))

        pixel_index = hp.ang2pix(nside, theta, phi)

        maps_at_pointing = [] # 3 maps in total: one at 26GHz, one at 30GHz and one at 34GHz
        for file in freq_files:
            

            m = np.load(file)

            m_at_pointing = [ m[pixel_index[i]] for i in range(len(pixel_index)) ]

            maps_at_pointing.append(m_at_pointing)

            # print(len(m_at_pointing), m_at_pointing[:20])
        
        interp_maps = []  # Shape is (time, freq)
       
        # interp = hp.pixelfunc.get_interp_val(maps_at_pointing, theta, phi, lonlat=True)
        # interp_maps.append(interp)

        frequencies = np.linspace(26, 34, nfreq)
        for i in range(len(pixel_index)):
            interp = np.interp(frequencies, np.array([26, 30, 34]), np.array([maps_at_pointing[0][i], maps_at_pointing[1][i], maps_at_pointing[2][i]]))
            interp_maps.append(interp)

        interp_maps = np.array(interp_maps)
        print((interp_maps.shape))
        # print(len(interp_maps[0]))

        # plt.figure()
        # plt.title('Ground pickup interpolated')
        # plt.plot(frequencies, interp_maps[0], label='time {0}')
        # plt.plot(frequencies, interp_maps[100], label='time {100}')
        # plt.plot(frequencies, interp_maps[1000], label='time {1000}')

        # plt.plot(frequencies, interp_maps[5000], label='time {5000}')
       
        # plt.plot(frequencies, interp_maps[10000], label='time {10000}')
        # plt.plot(frequencies, interp_maps[15000], label='time {15000}')
        # plt.plot(frequencies, interp_maps[20000], label='time {20000}')
        # plt.legend()
        # plt.xlabel('frequencies [GHz]')
        # plt.ylabel('temperature [K]')
        # plt.show()

        # self.fits_file_ground(interp_maps, pointing = None, scale = 1, outfile= f'{tod_gen_folder}/figs/results/TEST_INTERP_map_of_ground_pickup_from_fits_file_{data}')
        # *** Dont know how to plot this into a fits file


        return interp_maps

    def _fits_file_ground(self, filename):
        
        # Open the .fits file
        with fits.open(filename) as hdul:
            # Print the header of the primary HDU (Header/Data Unit)
            header = hdul[0].header
            print(hdul[0].header)
            print(hdul.info())
             
            # Access the data (e.g., image or table) in the primary HDU
            data = hdul[1].data
            wcs = WCS(header)  # Initialize WCS
            # print('data shape ' ,data.shape)   # Print dimensions of data array
        

        # temperature_data = np.zeros((19, 1024, 20175))
        # for f in range(19):
        #     az, el = self.get_pointing(feed = f)
        #     # Transform world coordinates (az, el) to pixel coordinates
        #     pixel_coords = wcs.world_to_pixel_values(az, el)  # returns (x, y) in pixel space
            
        #     # # Ensure pixel coordinates are within data bounds
        #     # x, y = int(pixel_coords[0]), int(pixel_coords[1])
        #     x, y = pixel_coords[0], pixel_coords[1]
        #     # print(y)
        #     # print(len(x), len(y))
            
        #     # print(len(data[0][0]))

        #     _temperature_data = np.zeros( ( len(x), len(data[0][0] ) ) )
        #     for i in range(len(x)):
        #         x_ = int(x[i])
        #         y_ = int(y[i])
        #         # print(len(data[x_][0]))
        #         _temperature_data[i] = data[x_][0]

        #     temperature_data[f] = _temperature_data.T

     
        az, el = self.get_pointing(feed = 0)
        print(f'Pointing shape is {np.array((az, el)).shape}')
        
        # Transform world coordinates (az, el) to pixel coordinates
        pixel_coords = wcs.world_to_pixel_values(az, el)  # returns (x, y) in pixel space
        
        # # Ensure pixel coordinates are within data bounds
        # x, y = int(pixel_coords[0]), int(pixel_coords[1])
        x, y = pixel_coords[0], pixel_coords[1]
        print(y)
        print(len(x), len(y))
        
        # print(len(data[0][0]))

        temperature_data = np.zeros( ( len(x), len(data[0][0] ) ) )
        for i in range(len(x)):
            x_ = int(x[i])
            y_ = int(y[i])
            # print(len(data[x_][0]))
            temperature_data[i] = data[x_][0]

        temperature_data = temperature_data.T
        print('temp data shape ',temperature_data.shape)
        # if 0 <= x < data.shape[1] and 0 <= y < data.shape[0]:
        #     temperature_data = data[y, x]
        # else:
        #     temperature_data = None  # Out of bounds

    
        return temperature_data

    def _lowpass_filter(self, signal, fknee=0.01, alpha=4.0, samprate=50):
        """ 
            Returns the lowpass filtered version of the input (same shape as input).
            Assumes signal is of shape [freq, time]. However, each frequency is calculated entirely independently.

            *** np.fft.
        """

        Ntod = signal.shape[1]
        Nfreq = signal.shape[0]
        signal_padded = np.zeros((Nfreq, 2*Ntod))
        signal_padded[:,:Ntod] = signal
        signal_padded[:,Ntod:] = signal[:,::-1]

        freq_padded = np.fft.rfftfreq(2*Ntod)*samprate
        W = 1.0/(1 + (freq_padded/fknee)**alpha)

        signal_padded = np.fft.irfft(np.fft.rfft(signal_padded)*W)
        return signal_padded[:,:Ntod]
                
            # Ntod = signal.shape[1]
            # Nfreq = signal.shape[0]
            # # print(f'Ntod {Ntod},  Nfreq {Nfreq}')

            # fastlen = next_fast_len(int(1.5*Ntod))
            # # print(f'Fastlen is {fastlen}')

            # center_slice = slice(int(0.25*Ntod), int(1.25*Ntod))
            # signal_padded = np.zeros((Nfreq, fastlen))
            # # print(f'Signal padded zero {signal_padded.shape}')

            # signal_padded[:,center_slice] = signal
            # # print(f'Signal padded {signal_padded.shape}')

            # freq_padded = np.fft.rfftfreq(fastlen)*samprate
            # W = 1.0/(1 + (freq_padded/fknee)**alpha)

            
            # # print(Ntod, fastlen, W.shape, signal_padded.shape, np.fft.rfft(signal_padded).shape, np.fft.rfft(np.zeros(100)).shape)
            # signal_padded = np.fft.irfft(np.fft.rfft(signal_padded)*W)
            # return signal_padded[:,center_slice]

    def _polyfilter(self, data, feed=0):
        print('def polyfilter')

        # print(f'All data nan? {np.all(np.isnan(data))}'
        def polyfilter_by_channel(data, sideband=0): 
            """ 
            Polynomial filter which will remove the first order polynomial from the TOD
            """
            _tod = data[feed, sideband, :, :]
            # print(f'tod_ shape is {tod_.shape}')
            time = np.arange(len(_tod[0, :]))
            freq = np.arange(len(_tod)) 
            print(f'Shape of freq is {freq.shape}, shape of time is {time.shape}')

            #Finding a linear best fit trend in the data
            fit = np.zeros((len(freq), len(time)))


            weight = np.full(len(freq), 1/self.sigma) 
            print(f'sigma is {self.sigma}')
            exit()
            
            
            for i in range(len(time)): 
                                
                # polynomial coefficients of a first order fit, highest order coefficients are ordered first
                coefficients = np.polyfit(freq, _tod[:, i], deg=1, w=weight)

                #Polynomial first order trend
                poly_trend = np.poly1d(coefficients)
                # print('poly_trend', poly_trend)

                fit[:, i] = poly_trend(freq) # Finding poly fit for each freq
                # print(f'fit is shape {fit.shape}, is fit all nan? {np.all(np.isnan(fit))} \n is fit all 0? {np.all(np.isnan(0))}')
                


            # print(f'All fit nan? {np.all(np.isnan(fit))}')
            #Subtracting the trend from the TOD
            tod_filtered = _tod - fit
            
            # print(f'tod_filtered shape is {tod_filtered.shape}, fit is {fit[:5]} and _tod {_tod[:5]}')

            return tod_filtered, fit
        

        tod_filtered = []
        fit = []

        for i in range(len(data[0])):

            filtered, _fit = polyfilter_by_channel(data, sideband=i)
            print(f'All filtered nan? {np.all(np.isnan(filtered))}')
            tod_filtered.append(filtered)
            fit.append(_fit)
        
        return np.array(tod_filtered), np.array(fit)




#SCRAPS




    def _fits_file_ground(self, filename):
        
        # Open the .fits file
        with fits.open(filename) as hdul:
            # Print the header of the primary HDU (Header/Data Unit)
            header = hdul[0].header
            print(hdul[0].header)
            print(hdul.info())
             
            # Access the data (e.g., image or table) in the primary HDU
            data = hdul[1].data
            wcs = WCS(header)  # Initialize WCS
            # print('data shape ' ,data.shape)   # Print dimensions of data array
        

        # temperature_data = np.zeros((19, 1024, 20175))
        # for f in range(19):
        #     az, el = self.get_pointing(feed = f)
        #     # Transform world coordinates (az, el) to pixel coordinates
        #     pixel_coords = wcs.world_to_pixel_values(az, el)  # returns (x, y) in pixel space
            
        #     # # Ensure pixel coordinates are within data bounds
        #     # x, y = int(pixel_coords[0]), int(pixel_coords[1])
        #     x, y = pixel_coords[0], pixel_coords[1]
        #     # print(y)
        #     # print(len(x), len(y))
            
        #     # print(len(data[0][0]))

        #     _temperature_data = np.zeros( ( len(x), len(data[0][0] ) ) )
        #     for i in range(len(x)):
        #         x_ = int(x[i])
        #         y_ = int(y[i])
        #         # print(len(data[x_][0]))
        #         _temperature_data[i] = data[x_][0]

        #     temperature_data[f] = _temperature_data.T

     
        az, el = self.get_pointing(feed = 0)
        print(f'Pointing shape is {np.array((az, el)).shape}')
        
        # Transform world coordinates (az, el) to pixel coordinates
        pixel_coords = wcs.world_to_pixel_values(az, el)  # returns (x, y) in pixel space
        
        # # Ensure pixel coordinates are within data bounds
        # x, y = int(pixel_coords[0]), int(pixel_coords[1])
        x, y = pixel_coords[0], pixel_coords[1]
        print(y)
        print(len(x), len(y))
        
        # print(len(data[0][0]))

        temperature_data = np.zeros( ( len(x), len(data[0][0] ) ) )
        for i in range(len(x)):
            x_ = int(x[i])
            y_ = int(y[i])
            # print(len(data[x_][0]))
            temperature_data[i] = data[x_][0]

        temperature_data = temperature_data.T
        print('temp data shape ',temperature_data.shape)
        # if 0 <= x < data.shape[1] and 0 <= y < data.shape[0]:
        #     temperature_data = data[y, x]
        # else:
        #     temperature_data = None  # Out of bounds

    
        return temperature_data

    def _lowpass_filter(self, signal, fknee=0.01, alpha=4.0, samprate=50):
        """ 
            Returns the lowpass filtered version of the input (same shape as input).
            Assumes signal is of shape [freq, time]. However, each frequency is calculated entirely independently.

            *** np.fft.
        """

        Ntod = signal.shape[1]
        Nfreq = signal.shape[0]
        signal_padded = np.zeros((Nfreq, 2*Ntod))
        signal_padded[:,:Ntod] = signal
        signal_padded[:,Ntod:] = signal[:,::-1]

        freq_padded = np.fft.rfftfreq(2*Ntod)*samprate
        W = 1.0/(1 + (freq_padded/fknee)**alpha)

        signal_padded = np.fft.irfft(np.fft.rfft(signal_padded)*W)
        return signal_padded[:,:Ntod]
                
            # Ntod = signal.shape[1]
            # Nfreq = signal.shape[0]
            # # print(f'Ntod {Ntod},  Nfreq {Nfreq}')

            # fastlen = next_fast_len(int(1.5*Ntod))
            # # print(f'Fastlen is {fastlen}')

            # center_slice = slice(int(0.25*Ntod), int(1.25*Ntod))
            # signal_padded = np.zeros((Nfreq, fastlen))
            # # print(f'Signal padded zero {signal_padded.shape}')

            # signal_padded[:,center_slice] = signal
            # # print(f'Signal padded {signal_padded.shape}')

            # freq_padded = np.fft.rfftfreq(fastlen)*samprate
            # W = 1.0/(1 + (freq_padded/fknee)**alpha)

            
            # # print(Ntod, fastlen, W.shape, signal_padded.shape, np.fft.rfft(signal_padded).shape, np.fft.rfft(np.zeros(100)).shape)
            # signal_padded = np.fft.irfft(np.fft.rfft(signal_padded)*W)
            # return signal_padded[:,center_slice]

    def _polyfilter(self, data, feed=0):
        print('def polyfilter')

        # print(f'All data nan? {np.all(np.isnan(data))}'
        def polyfilter_by_channel(data, sideband=0): 
            """ 
            Polynomial filter which will remove the first order polynomial from the TOD
            """
            _tod = data[feed, sideband, :, :]
            # print(f'tod_ shape is {tod_.shape}')
            time = np.arange(len(_tod[0, :]))
            freq = np.arange(len(_tod)) 
            print(f'Shape of freq is {freq.shape}, shape of time is {time.shape}')

            #Finding a linear best fit trend in the data
            fit = np.zeros((len(freq), len(time)))


            weight = np.full(len(freq), 1/self.sigma) 
            print(f'sigma is {self.sigma}')
            exit()
            
            
            for i in range(len(time)): 
                                
                # polynomial coefficients of a first order fit, highest order coefficients are ordered first
                coefficients = np.polyfit(freq, _tod[:, i], deg=1, w=weight)

                #Polynomial first order trend
                poly_trend = np.poly1d(coefficients)
                # print('poly_trend', poly_trend)

                fit[:, i] = poly_trend(freq) # Finding poly fit for each freq
                # print(f'fit is shape {fit.shape}, is fit all nan? {np.all(np.isnan(fit))} \n is fit all 0? {np.all(np.isnan(0))}')
                


            # print(f'All fit nan? {np.all(np.isnan(fit))}')
            #Subtracting the trend from the TOD
            tod_filtered = _tod - fit
            
            # print(f'tod_filtered shape is {tod_filtered.shape}, fit is {fit[:5]} and _tod {_tod[:5]}')

            return tod_filtered, fit
        

        tod_filtered = []
        fit = []

        for i in range(len(data[0])):

            filtered, _fit = polyfilter_by_channel(data, sideband=i)
            print(f'All filtered nan? {np.all(np.isnan(filtered))}')
            tod_filtered.append(filtered)
            fit.append(_fit)
        
        return np.array(tod_filtered), np.array(fit)

    def _mapmaker(self, tod, sigma, pointing_idx): 
        """
        *** STRAIGHT UP COPIED FROM PROJECT WORK, will need some editing ***



        Input: Downsampled tod data and tod uncertainty, pick one band.
                TOD and sigma must be 1d arrays
                
        Output: signal map, uncertainty map and hit count map
        """

        n_pix = 120 * 120  # pix length

        signal_map = np.zeros(n_pix)
        sigma_map = np.zeros(n_pix)
        # hit_count_map = np.zeros(n_pix)

        hit_count_map, _ = np.histogram(pointing_idx, bins = np.arange(n_pix+1))
        
        hit_pix = np.unique(pointing_idx)
        # print(f'The tod shape is {tod.shape}, \n')
        for pix in hit_pix:
            idx_tod = np.where(pointing_idx==pix)[0]
            d_array = np.array([tod[idx] for idx in idx_tod ])
            sigma_array = np.array([sigma[idx] for idx in idx_tod ])

            numerator = np.sum(d_array/sigma_array**2)
            denominator = np.sum(1/sigma_array**2)
            signal_map[pix] = numerator/denominator
            sigma_map[pix] = 1/np.sqrt(np.sum(1/sigma_array**2))#/len(idx_tod) ### *** Divide by len okay???

        
        return signal_map, sigma_map, hit_count_map