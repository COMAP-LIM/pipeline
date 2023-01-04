// Compile as
// g++  -shared mapbinner.cpp -o mapbinner.so.1 -fPIC -fopenmp -Ofast -std=c++11 -g

#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <iostream>

// Coadd functions
extern "C" void bin_map(
    float *tod,
    float *sigma,
    int *freqmask,
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

                numerator[idx_map] += tod[idx_tod] * inv_var * freqmask[freq_feed_idx];
                denominator[idx_map] += inv_var * freqmask[freq_feed_idx];
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

                numerator[idx_map] += tod[idx_tod] * inv_var * freqmask[freq_feed_idx];
                denominator[idx_map] += inv_var * freqmask[freq_feed_idx];

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