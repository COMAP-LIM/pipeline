// Compile as
// gcc  -shared mapbinner.c -o mapbinner.so.1 -fPIC -fopenmp -Ofast -std=c99 -g

#include <stdio.h>
#include <math.h>
#include <omp.h>

// // Coadd functions
// void bin_freq_map(
//     float *tod,
//     float *sigma,
//     float *freqmask,
//     int *idx_pix,
//     int *nhit,
//     float *numerator,
//     float *denominator,
//     int nfreq,
//     int nfeed,
//     int nsamp,
//     int nside,
//     int nthread)
// {
//     int t;
//     int d;
//     int f;
//     int feed_px_idx;
//     int freq_feed_idx;
//     int idx_map;
//     int idx_tod;
//     float inv_var;
//     int npix;

//     // Looping through data and binning up into maps
//     omp_set_num_threads(nthread);
// #pragma omp parallel private(idx_map, idx_tod, freq_feed_idx, feed_px_idx, f, d, t, inv_var) shared(tod, sigma, freqmask, idx_pix, nhit, numerator, denominator, nfreq, nsamp, nfeed, nside, npix)
//     {
//         npix = nside * nside * nfeed;

//         for (d = 0; d < nfeed; d++)
//         {
// #pragma omp for
//             for (f = 0; f < nfreq; f++)
//             {
//                 freq_feed_idx = nfreq * d + f;
//                 if (freqmask[freq_feed_idx] == 0)
//                 {
//                     continue;
//                 }
//                 else
//                 {
//                     inv_var = sigma[freq_feed_idx];
//                     inv_var = inv_var * inv_var;
//                     inv_var = freqmask[freq_feed_idx] / inv_var;
//                 }
//                 {
//                     for (t = 0; t < nsamp; t++)
//                     {
//                         feed_px_idx = idx_pix[nfeed * t + d];

//                         idx_map = nfreq * feed_px_idx + f;

//                         idx_tod = nfeed * t + d;
//                         idx_tod = nfreq * idx_tod + f;

//                         if (feed_px_idx > 0 && feed_px_idx < npix)
//                         {
//                             nhit[idx_map]++;

//                             if (!isinf(inv_var))
//                                 if (!isinf(inv_var))
//                                 {
//                                     numerator[idx_map] = numerator[idx_map] + tod[idx_tod] * inv_var;
//                                     denominator[idx_map] = denominator[idx_map] + inv_var;
//                                 }
//                         }
//                         else
//                         {
//                             continue;
//                         }
//                     }
//                 }
//             }
//         }
//     }
// }

// #####################################################
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
// #####################################################

// Coadd functions
void bin_freq_map(
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

    // Looping through data and binning up into maps
    omp_set_num_threads(nthread);

#pragma omp parallel for
    for (int d = 0; d < nfeed; d++)
    {
        for (int t = 0; t < nsamp; t++)
        {
            int feed_px_idx = idx_pix[nfeed * t + d];
            int time_det_idx = nsamp * d + t;
            for (int f = 0; f < nfreq; f++)
            {
                int freq_feed_idx = nfeed * f + d;

                float inv_var = sigma[freq_feed_idx];

                int idx_map = nfreq * feed_px_idx + f;

                int idx_tod = nfreq * time_det_idx + f;

                // nhit[idx_map]++;

                numerator[idx_map] += tod[idx_tod] * inv_var;
                denominator[idx_map] += inv_var;
            }
        }
    }
}

// Coadd functions
void add_tod2map(
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

    // Looping through data and binning up into maps
    omp_set_num_threads(nthread);

#pragma omp parallel for
    for (int d = 0; d < nfeed; d++)
    {
        for (int t = 0; t < nsamp; t++)
        {
            int feed_px_idx = idx_pix[nfeed * t + d];
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
void add_tod(
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