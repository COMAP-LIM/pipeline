stats_cut = {
    ### Non-data derived statistics ###
    # High-level info
    'obsid': [float("nan"), float("nan")],
    'scanid': [float("nan"), float("nan")],
    'mjd': [float("nan"), float("nan")],
    'fbit': [float("nan"), float("nan")],
    'scan_length': [3.0, 8.0],
    'saddlebag': [0.0, 5.0],
    'acceptmod_error': [0.0, 0.0],

    # Time and pointing related info
    'night': [0.0, 12.0],
    'sidereal': [0.0, 360.0],
    'az': [0.0, 360.0],
    'el': [35.0, 65.0],
    'moon_dist': [10, 180],
    'moon_angle': [-180, 360], 
    'moon_cent_sl': [0, 0.5],
    'moon_outer_sl': [0, 0.5],
    'sun_dist': [40, 180],
    'sun_angle': [-180, 360], 
    'sun_cent_sl': [0, 0.5],
    'sun_outer_sl': [0, 0.5],
    'sun_el': [-90, 90],

    # Weather related info
    'weather': [0.0, 0.35],
    'airtemp': [-10, 40],
    'dewtemp': [-30, 20],
    'humidity': [0, 1],
    'pressure': [860, 900],
    'rain': [0.0, 1e-6],
    'winddir': [0, 360],
    'windspeed': [0, 9],

    ### Data derived statistics ###
    'acceptrate': [0.5, 1.0],
    'acceptrate_specific': [0.8, 1.0],

    # High-level sanity checks
    'tsys': [25.0, 65.0],
    'power_mean': [0.0, 1e8],
    'sigma_mean': [20, 600],
    'fknee_mean': [5, 12],
    'alpha_mean': [-1.3, -0.75],
    'n_spikes': [0.0, 20.0],
    'n_jumps': [0.0, 0.99],
    'n_anomalies': [0.0, 2.0],
    'n_nan': [float("nan"), float("nan")],

    # Data statistics
    'chi2': [-3, 3],
    'az_chi2': [-6.0, 2.0],
    'max_az_chi2': [0.0, 5.0],
    'med_az_chi2': [-1.0, 0.1],
    'az_amp': [-0.0005, 0.0005],
    'el_amp': [-1.0, 1.0],
    'kurtosis': [-0.015, 0.015],
    'skewness': [-0.005, 0.01],

    # Tests on L2gen filter parameters
    'npca' : [0.0, 6.5],
    'npcaf' : [0.0, 3.5],
    'pca1': [0.0, 12.0],
    'pca2': [0.0, 5.0],
    'pca3': [0.0, 4.0],
    'pca4': [0.0, 3.5],
    'pcf1' : [0.0, 10.0],
    'pcf2' : [0.0, 7.5],
    'pcsm' : [0, 30.0],
    'sigma_poly0': [0, 0.0015],  #[float("nan"), float("nan")], #[1e-7, 1e-1],
    'fknee_poly0': [5, 11],  #[float("nan"), float("nan")], #[1e-4, 1e2],
    'alpha_poly0': [-1.2, -0.7],  #[float("nan"), float("nan")], #[-4.0, 2.0],
    'sigma_poly1': [1e-7, 5e-4],  #[float("nan"), float("nan")], #[1e-5, 1e-2],
    'fknee_poly1': [1e-8, 0.9],  #[float("nan"), float("nan")], #[1e-4, 1e2],
    'alpha_poly1': [-2, -0.1],  #[float("nan"), float("nan")], #[-5.0, 3.0],

    # PS chi2 statistics
    'ps_chi2': [float("nan"), float("nan")],
    'ps_s_sb_chi2': [float("nan"), float("nan")],
    'ps_s_feed_chi2': [float("nan"), float("nan")],
    'ps_s_chi2': [float("nan"), float("nan")],
    'ps_o_sb_chi2': [float("nan"), float("nan")],
    'ps_o_feed_chi2': [float("nan"), float("nan")],
    'ps_o_chi2': [float("nan"), float("nan")],
    'ps_z_s_sb_chi2': [float("nan"), float("nan")],
    'ps_xy_s_sb_chi2': [float("nan"), float("nan")],
    
    # Standing waves (not used)
    'sw_01': [float("nan"), float("nan")],
    'sw_02': [float("nan"), float("nan")],
    'sw_03': [float("nan"), float("nan")],
    'sw_04': [float("nan"), float("nan")],
    'sw_05': [float("nan"), float("nan")],
    'sw_06': [float("nan"), float("nan")],
    'sw_07': [float("nan"), float("nan")],
    'sw_08': [float("nan"), float("nan")],
    'sw_09': [float("nan"), float("nan")],
    'sw_10': [float("nan"), float("nan")],
    'sw_11': [float("nan"), float("nan")],
    'sw_12': [float("nan"), float("nan")],
    'sw_13': [float("nan"), float("nan")],
    'sw_14': [float("nan"), float("nan")],
}
