stats_list = [
    'obsid',         # Obisd of the scan
    'scanid',        # scanid
    'mjd',           # mean MJD of scan
    'night',         # distance in h from 2 AM (UTC - 7)
    'sidereal',      # sidereal time in degrees (up to a phase)
    'az',            # mean azimuth of scan
    'el',            # mean elevation of scan
    'chi2',          # chi2 statistic for all timestreams of whole sb
    'acceptrate',    # acceptrate of sb
    'acceptrate_specific', # acceptrate of sb
    'az_chi2',       # chi2 of azimuth binned timestreams
    'max_az_chi2',   # maximum chi2 of azimuth binned single frequency timestreams
    'med_az_chi2',   # median chi2 of azimuth binned single frequency timestreams
    'fbit',          # featurebit of scan (indicates scanning mode)
    'az_amp',        # average amplitude of fitted azimuth-template
    'el_amp',        # average amplitude of fitted elevation-template
    'n_spikes',      # number of spikes
    'n_jumps',       # number of jumps
    'n_anomalies',   # number of anomalies
    'n_nan',         # number of nan-samples
    'tsys',          # average Tsys value of scan
    'pca1',          # average variance of removed pca mode 1
    'pca2',          # average variance of removed pca mode 2
    'pca3',          # average variance of removed pca mode 3
    'pca4',          # average variance of removed pca mode 4
    'weather',       # probability of bad weather
    'kurtosis',      # kurtosis of timestreams
    'skewness',      # skewness of timestreams
    'scan_length',   # length of scan in minutes
    'saddlebag',     # saddlebag number (1-4)
    'sigma_poly0',   # sigma of mean in poly filter
    'fknee_poly0',   # fknee of mean in poly filter
    'alpha_poly0',   # alpha of mean in poly filter
    'sigma_poly1',   # sigma of slope in poly filter
    'fknee_poly1',   # fknee of slope in poly filter
    'alpha_poly1',   # alpha of slope in poly filter
    'power_mean',    # mean of sideband mean
    'sigma_mean',    # sigma of sideband mean
    'fknee_mean',    # fknee of sideband mean
    'alpha_mean',    # alpha of sideband mean
    'airtemp',       # hk: airtemp, C
    'dewtemp',       # hk: dewpoint temp, C
    'humidity',      # hk: relative humidity, (0-1)
    'pressure',      # hk: pressure, millibars
    'rain',          # hk: rain today, mm
    'winddir',       # hk: azimuth from where wind is blowing, deg 
    'windspeed',     # hk: windspeed m/s
    'moon_dist',     # distance to the moon in deg
    'moon_angle',    # the azimutal angle of the moon relative to pointing
    'moon_cent_sl',  # is the moon close to the central sidelobe
    'moon_outer_sl', # is the moon close to the outer sidelobes (from feedlegs)
    'sun_dist',      # distance to the sun in deg
    'sun_angle',     # the azimutal angle of the sun relative to pointing
    'sun_cent_sl',   # is the sun close to the central sidelobe
    'sun_outer_sl',  # is the msun close to the outer sidelobes (from feedlegs)
    'sun_el',        # elevation of sun
    'ps_chi2',       # the "old" ps_chi2 algorith, should be ~ ps_s_sb_chi2
    'ps_s_sb_chi2',  # ps_chi2 for single sb single scan (previously ps_chi2)
    'ps_s_feed_chi2',# ps_chi2 for single feed single scan 
    'ps_s_chi2',     # ps_chi2 combining all feeds for single scan 
    'ps_o_sb_chi2',  # ps_chi2 for single sb full obsid 
    'ps_o_feed_chi2',# ps_chi2 for single feed full obsid  
    'ps_o_chi2',     # ps_chi2 combining all feeds for full obsid
    'ps_z_s_sb_chi2',# ps_chi2 for average of z-direction 1D ps, single sb single scan
    'ps_xy_s_sb_chi2',# ps_chi2 for average of xy-direction 2D ps, single sb single scan
    'sw_01',           # standing wave magnitude (unnormalized)
    'sw_02',           # standing wave magnitude (unnormalized)
    'sw_03',           # standing wave magnitude (unnormalized)
    'sw_04',           # standing wave magnitude (unnormalized)
    'sw_05',           # standing wave magnitude (unnormalized)
    'sw_06',           # standing wave magnitude (unnormalized)
    'sw_07',           # standing wave magnitude (unnormalized)
    'sw_08',           # standing wave magnitude (unnormalized)
    'sw_09',           # standing wave magnitude (unnormalized)
    'sw_10',          # standing wave magnitude (unnormalized)
    'sw_11',          # standing wave magnitude (unnormalized)
    'sw_12',          # standing wave magnitude (unnormalized)
    'sw_13',          # standing wave magnitude (unnormalized)
    'sw_14'           # standing wave magnitude (unnormalized)
 ]
