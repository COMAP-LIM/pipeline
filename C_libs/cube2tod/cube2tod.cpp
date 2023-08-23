// Compile as
// g++  -shared cube2tod.cpp -o cube2tod.so.1 -fPIC -fopenmp -Ofast -std=c++11 -g

#include <stdio.h>
#include <omp.h>
#include <cmath>
#include <iostream>


extern "C" void cube2tod(
    float *tod,
    double *Tsys,
    float *simdata,
    double *ra,
    double *dec,
    float dra,
    float ddec,
    float ra0,
    float dec0,
    int nra,
    int ndec,
    long int nfreq,
    long int nfeed,
    long int nsamp,
    int nthread
    )
    /**
     * Inject signal from temperature cube simulation into
     * TOD for all feeds, frequencies and time samples.
     * 
     * @param 
     *      - tod -- Time stream to inject signal into. Has dimension {detector, frequencies, times} and be normalized and unitless.
     *      - Tsys -- System temperature. Has dimension {detector, frequencies}.
     *      - simdata -- Simulation cube of brighness temperatures. Has dimension {frequencies, DEC, RA}
     *      - ra -- Right ascention telescope pointing coordinates. Has dimension {detectors, times}
     *      - dec -- Declination telescope pointing coordinates. Has dimension {detectors, times}
     *      - dra -- Right ascention simulation grid resolution.
     *      - ddec -- Declination simulation grid resolution.
     *      - dra -- Right Ascention simulation grid size.
     *      - ddec -- Declination simulation grid size.
     *      - nfreq -- Number of frequency channels in all sidebands.
     *      - nfeed -- Number of detectors.
     *      - nsamp -- Number of time samples in TOD.
     *      - nthread -- Number of OpenMP threads to use for parallelization.
    */
{
    // Number of pixels in simulation box grid
    int npix = nra * ndec;

    // Looping through data and binning up into maps
    omp_set_num_threads(nthread);
    #pragma omp parallel for
    for (long int d = 0; d < nfeed; d++)
    {   
        // std::cout << "Detector " << d << std::endl;
        for (long int t = 0; t < nsamp; t++)
        {     
            // std::cout << "Detector" << d << "Time sample" << t << std::endl;

            long int time_feed_idx = nsamp * d + t;

            // Define pointing index
            float x = (dec[time_feed_idx] - dec0) / ddec;
            float y = (ra[time_feed_idx] - ra0) / dra;

            // Previous grid points
            int x0 = std::floor(x);
            int y0 = std::floor(y);
            
            // Next grid points
            int x1 = x0 + 1;
            int y1 = y0 + 1;
            
            // Skip if any pointing index is outside of valid bounding box
            if (x0 < 0 || x0 >= ndec)
            {
                continue;
            }
            else if (x1 < 0 || x1 >= ndec)
            {
                continue;
            }
            else if (y0 < 0 || y0 >= nra)
            {
                continue;
            }
            else if (y1 < 0 || y1 >= nra)
            {
                continue;
            }

            // Distance from previous grid points
            float dx = x - x0;
            float dy = y - y0;
            
            // Defining flattened pixel grid indices
            int x00 = nra * x0 + y0; 
            int x11 = nra * x1 + y1; 
            int x01 = nra * x0 + y1; 
            int x10 = nra * x1 + y0; 
            
            for (long int f = 0; f < nfreq; f++)
            {   
                // Defining flattened frequency-grid indices
                int idx_00 = f * npix + x00; 
                int idx_11 = f * npix + x11; 
                int idx_01 = f * npix + x01; 
                int idx_10 = f * npix + x10; 

                // Interpolate in x-direction along to constant y-grid points
                double signal_xy1 = (1 - dx) * simdata[idx_00] + dx * simdata[idx_10];  
                double signal_xy2 = (1 - dx) * simdata[idx_01] + dx * simdata[idx_11]; 
                
                // Interpolate from previous x-interpolation in y-direction
                float signal  = (1 - dy) * signal_xy1 + dy * signal_xy2;

                // Defining frequency-detector-time index for indexing TOD
                long int idx_tod = nsamp * (d * nfreq + f) + t;
                
                // Injecting TOD with signal
                tod[idx_tod] *= (1 + signal / Tsys[d * nfreq + f]);
            }
        }
    }
}



extern "C" void replace_tod_with_nearest_neighbor_signal(
    float *tod,
    float *simdata,
    double *ra,
    double *dec,
    float dra,
    float ddec,
    float ra0,
    float dec0,
    int nra,
    int ndec,
    long int nfreq,
    long int nfeed,
    long int nsamp,
    int nthread
    )
    /**
     * Inject signal from temperature cube simulation into
     * TOD for all feeds, frequencies and time samples.
     * 
     * @param 
     *      - tod -- Time stream to inject signal into. Has dimension {detector, frequencies, times} and be normalized and unitless.
     *      - simdata -- Simulation cube of brighness temperatures. Has dimension {frequencies, DEC, RA}
     *      - ra -- Right ascention telescope pointing coordinates. Has dimension {detectors, times}
     *      - dec -- Declination telescope pointing coordinates. Has dimension {detectors, times}
     *      - dra -- Right ascention simulation grid resolution.
     *      - ddec -- Declination simulation grid resolution.
     *      - dra -- Right Ascention simulation grid size.
     *      - ddec -- Declination simulation grid size.
     *      - nfreq -- Number of frequency channels in all sidebands.
     *      - nfeed -- Number of detectors.
     *      - nsamp -- Number of time samples in TOD.
     *      - nthread -- Number of OpenMP threads to use for parallelization.
    */
{

    // Number of pixels in simulation box grid
    int npix = nra * ndec;

    // Looping through data and binning up into maps
    omp_set_num_threads(nthread);
    #pragma omp parallel for
    for (int d = 0; d < nfeed; d++)
    {
        for (int t = 0; t < nsamp; t++)
        {     
            long int time_feed_idx = nsamp * d + t;

            // Define pointing index
            float x = (dec[time_feed_idx] - dec0) / ddec;
            float y = (ra[time_feed_idx] - ra0) / dra;
            
            int x_idx = std::round(x);
            int y_idx = std::round(y);
            
            // Skip if any pointing index is outside of valid bounding box
            if (x_idx < 0 || x_idx >= ndec)
            {   
                continue;
            }
            else if (y_idx < 0 || y_idx >= nra)
            {
                continue;
            }

            // Distance from previous grid points
            
            // Defining flattened pixel grid indices
            int idx = nra * x_idx + y_idx; 
            
            for (int f = 0; f < nfreq; f++)
            {   
                // Defining flattened frequency-grid indices
                int idx_full = f * npix + idx; 

                // Interpolate from previous x-interpolation in y-direction
                float signal  = simdata[idx_full];
                
                // Defining frequency-detector-time index for indexing TOD
                long int idx_tod = nsamp * (d * nfreq + f) + t;

                // Injecting TOD with signal
                tod[idx_tod] = signal;
            }
        }
    }
}


extern "C" void replace_tod_with_bilinear_interp_signal(
    float *tod,
    float *simdata,
    double *ra,
    double *dec,
    float dra,
    float ddec,
    float ra0,
    float dec0,
    int nra,
    int ndec,
    long int nfreq,
    long int nfeed,
    long int nsamp,
    int nthread
    )
    /**
     * Inject signal from temperature cube simulation into
     * TOD for all feeds, frequencies and time samples.
     * 
     * @param 
     *      - tod -- Time stream to inject signal into. Has dimension {detector, frequencies, times} and be normalized and unitless.
     *      - simdata -- Simulation cube of brighness temperatures. Has dimension {frequencies, DEC, RA}
     *      - ra -- Right ascention telescope pointing coordinates. Has dimension {detectors, times}
     *      - dec -- Declination telescope pointing coordinates. Has dimension {detectors, times}
     *      - dra -- Right ascention simulation grid resolution.
     *      - ddec -- Declination simulation grid resolution.
     *      - dra -- Right Ascention simulation grid size.
     *      - ddec -- Declination simulation grid size.
     *      - nfreq -- Number of frequency channels in all sidebands.
     *      - nfeed -- Number of detectors.
     *      - nsamp -- Number of time samples in TOD.
     *      - nthread -- Number of OpenMP threads to use for parallelization.
    */
{



    // Number of pixels in simulation box grid
    int npix = nra * ndec;

    // Looping through data and binning up into maps
    omp_set_num_threads(nthread);
    #pragma omp parallel for
    for (int d = 0; d < nfeed; d++)
    {
        for (int t = 0; t < nsamp; t++)
        {     
            long int time_feed_idx = nsamp * d + t;

            // Define pointing index
            double x = (dec[time_feed_idx] - dec0) / ddec;
            double y = (ra[time_feed_idx] - ra0) / dra;

            // Previous grid points
            int x0 = std::floor(x);
            int y0 = std::floor(y);
            
            // Next grid points
            int x1 = x0 + 1;
            int y1 = y0 + 1;
            
            // Skip if any pointing index is outside of valid bounding box
            if (x0 < 0 || x0 >= ndec)
            {
                continue;
            }
            else if (x1 < 0 || x1 >= ndec)
            {
                continue;
            }
            else if (y0 < 0 || y0 >= nra)
            {
                continue;
            }
            else if (y1 < 0 || y1 >= nra)
            {
                continue;
            }

            // Distance from previous grid points
            double dx = x - x0;
            double dy = y - y0;
            
            // Defining flattened pixel grid indices
            int x00 = nra * x0 + y0; 
            int x11 = nra * x1 + y1; 
            int x01 = nra * x0 + y1; 
            int x10 = nra * x1 + y0; 
            
            for (int f = 0; f < nfreq; f++)
            {   
                // Defining flattened frequency-grid indices
                int idx_00 = f * npix + x00; 
                int idx_11 = f * npix + x11; 
                int idx_01 = f * npix + x01; 
                int idx_10 = f * npix + x10; 

                // Interpolate in x-direction along to constant y-grid points
                double signal_xy1 = (1 - dx) * simdata[idx_00] + dx * simdata[idx_10];  
                double signal_xy2 = (1 - dx) * simdata[idx_01] + dx * simdata[idx_11];  

                // Interpolate from previous x-interpolation in y-direction
                double signal  = (1 - dy) * signal_xy1 + dy * signal_xy2;

                // Defining frequency-detector-time index for indexing TOD
                long int idx_tod = nsamp * (d * nfreq + f) + t;

                // Injecting TOD with signal
                tod[idx_tod] = signal;
            }
        }
    }
}
