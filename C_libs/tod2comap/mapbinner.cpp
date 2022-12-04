// Compile as
// g++  -shared mapbinner.cpp -o mapbinner.so.1 -fPIC -fopenmp -Ofast -std=c++11 -g

#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <iostream>

// ##########################################################################
// Functions (OMP threads):   Runtimes:
// -----------------------------------------------------
// bin_freq_map (1):        0.22084399335086347 s ± 0.00023269443926871565 s
// bin_freq_map (24):       0.13131949849426747 s ± 0.011892204762269517 s
//
// add_tod2map (1):         0.09907585203647613 s ± 0.007195123036765524 s
// add_tod2map (24):        0.04028610795736313 s ± 0.0017081607784004726 s
//
// add_tod (1):             0.042744452953338626 s ± 0.007279209212815392 s
// add_tod (24):            0.012062790542840958 s ± 0.001080302766340433
// ###########################################################################

// Coadd functions
extern "C" void bin_map(
    float *tod,
    float *sigma,
    int *idx_ra_pix,
    int *idx_dec_pix,
    float *numerator,
    float *denominator,
    int nfreq,
    int nfeed,
    int nsamp,
    int nside_ra,
    int nside_dec,
    int nthread)
{
    int npix = nside_ra * nside_dec;

    // Looping through data and binning up into maps
    omp_set_num_threads(nthread);
#pragma omp parallel for
    for (int d = 0; d < nfeed; d++)
    {
        for (int t = 0; t < nsamp; t++)
        {
            int ra = idx_ra_pix[nfeed * t + d];
            int dec = idx_dec_pix[nfeed * t + d];
            if (ra < 0 || ra >= nside_ra)
            {
                continue;
            }
            else if (dec < 0 || dec >= nside_dec)
            {
                continue;
            }

            int pixel_index = idx_dec_pix[nfeed * t + d] * nside_ra + idx_ra_pix[nfeed * t + d];
            int feed_px_idx = npix * d + pixel_index;
            int time_det_idx = nsamp * d + t;
            for (int f = 0; f < nfreq; f++)
            {
                int freq_feed_idx = nfreq * d + f;

                float inv_var = sigma[freq_feed_idx];

                int idx_map = nfreq * feed_px_idx + f;

                int idx_tod = nfreq * time_det_idx + f;

                numerator[idx_map] += tod[idx_tod] * inv_var;
                denominator[idx_map] += inv_var;
            }
        }
    }
}

// Coadd functions
extern "C" void bin_nhit_and_map(
    float *tod,
    float *sigma,
    int *freqmask,
    int *idx_ra_pix,
    int *idx_dec_pix,
    int *nhit,
    float *numerator,
    float *denominator,
    int nfreq,
    int nfeed,
    int nsamp,
    int nside_ra,
    int nside_dec,
    int nthread,
    int scanid)
{
    int npix = nside_ra * nside_dec;

    // Looping through data and binning up into maps
    omp_set_num_threads(nthread);

#pragma omp parallel for
    for (int d = 0; d < nfeed; d++)
    {

        for (int t = 0; t < nsamp; t++)
        {
            int ra = idx_ra_pix[nfeed * t + d];
            int dec = idx_dec_pix[nfeed * t + d];
            if (ra < 0 || ra >= nside_ra)
            {
                continue;
            }
            else if (dec < 0 || dec >= nside_dec)
            {
                continue;
            }
            int pixel_index = idx_dec_pix[nfeed * t + d] * nside_ra + idx_ra_pix[nfeed * t + d];
            int feed_px_idx = npix * d + pixel_index;
            int time_det_idx = nsamp * d + t;

            for (int f = 0; f < nfreq; f++)
            {
                int freq_feed_idx = nfreq * d + f;

                float inv_var = sigma[freq_feed_idx];

                int idx_map = nfreq * feed_px_idx + f;

                int idx_tod = nfreq * time_det_idx + f;

                numerator[idx_map] += tod[idx_tod] * inv_var;
                denominator[idx_map] += inv_var;

                nhit[idx_map] += freqmask[freq_feed_idx];
            }
        }
    }
}

// Coadd functions
extern "C" void add_tod2map(
    float *tod,
    float *sigma,
    float *freqmask,
    int *idx_pix,
    int *nhit,
    float *numerator,
    float *denominator,
    int nfreq,
    int nfeed,
    int nsamp,
    int nside_ra,
    int nside_dec,
    int nthread)
{
    int npix = nside_ra * nside_dec * nfeed;
    printf("%d %d", nside_ra, nside_dec);

    // Looping through data and binning up into maps
    omp_set_num_threads(nthread);

#pragma omp parallel for
    for (int d = 0; d < nfeed; d++)
    {
        for (int t = 0; t < nsamp; t++)
        {
            int feed_px_idx = idx_pix[nsamp * d + t];
            int time_det_idx = nsamp * d + t;
            for (int f = 0; f < nfreq; f++)
            {
                int idx_map = nfreq * feed_px_idx + f;

                int idx_tod = nfreq * time_det_idx + f;

                numerator[idx_map] += tod[idx_tod];
            }
        }
    }
}

// Coadd functions
extern "C" void add_tod(
    float *tod,
    float *sigma,
    float *freqmask,
    int *idx_pix,
    int *nhit,
    float *numerator,
    float *denominator,
    int nfreq,
    int nfeed,
    int nsamp,
    int nside,
    int nthread)
{
    int npix = nside * nside * nfeed;
    float a = 0;
    // Looping through data and binning up into maps

    omp_set_num_threads(nthread);

#pragma omp parallel for reduction(+ \
                                   : a)
    for (int d = 0; d < nfeed; d++)
    {
        for (int t = 0; t < nsamp; t++)
        {
            int time_det_idx = nsamp * d + t;
            for (int f = 0; f < nfreq; f++)
            {
                int idx_tod = nfreq * time_det_idx + f;
                a += tod[idx_tod];
            }
        }
    }
}