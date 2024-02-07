// Compile as:
// g++ -shared -lm -fPIC -o mPCAlib.so.1 mPCAlib.cpp -std=c++11 -O3 -fopenmp -lpthread

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <vector>

using namespace std;


extern "C"
void PCA_experimental(float *map_signal, float *map_signal_T, float *inv_rms_2, float *inv_rms_2_T, double *angvec, double *freqvec, int Nfreq, int Npix, int max_iter, double err_tol){
    double *angvec_new = new double[Npix];
    double *freqvec_new = new double[Nfreq];
    double *angvec_old = new double[Npix];
    double *freqvec_old = new double[Nfreq];
    double freqvec_diff_new = 0;
    double freqvec_diff_old = 0;
    double angvec_diff_new = 0;
    double angvec_diff_old = 0;

    double *freqvec_momentum = new double[Nfreq];
    double *angvec_momentum = new double[Npix];

    for(int ifreq=0; ifreq<Nfreq; ifreq++){
        freqvec_old[ifreq] = freqvec[ifreq];
        freqvec_momentum[ifreq] = 0.0;
    }
    for(int ipix=0; ipix<Npix; ipix++){
        angvec_old[ipix] = angvec[ipix];
        angvec_momentum[ipix] = 0.0;
    }


    for(int iter=0; iter<max_iter; iter++){
        
        #pragma omp parallel for
        for(int ifreq=0; ifreq<Nfreq; ifreq++){
            freqvec_new[ifreq] = 0.0;
            for(int ipix=0; ipix<Npix; ipix++){
                int idx = ifreq*Npix + ipix;
                freqvec_new[ifreq] += map_signal[idx]*angvec[ipix]*inv_rms_2[idx];
            }
            double freqvec_denominator = 0.0;
            for(int ipix=0; ipix<Npix; ipix++){
                int idx = ifreq*Npix + ipix;
                freqvec_denominator += angvec[ipix]*angvec[ipix]*inv_rms_2[idx];
            }
            if(freqvec_denominator != 0){
                freqvec_new[ifreq] /= freqvec_denominator;
            }
            else{
                freqvec_new[ifreq] = 0.0;
            }
        }


        for(int ifreq=0; ifreq<Nfreq; ifreq++){
            freqvec_diff_old = freqvec[ifreq] - freqvec_old[ifreq];
            freqvec_diff_new = freqvec_new[ifreq] - freqvec[ifreq];
            freqvec_momentum[ifreq] += 0.03;
            if(freqvec_diff_new*freqvec_diff_old < 0){
                freqvec_momentum[ifreq] = 0.0;
            }
            if(freqvec_momentum[ifreq] > 0.9){
                freqvec_momentum[ifreq] = 0.9;
            }
        }

        for(int ifreq=0; ifreq<Nfreq; ifreq++){
            freqvec_new[ifreq] += freqvec_momentum[ifreq]*(freqvec_new[ifreq] - freqvec[ifreq]);
        }


        #pragma omp parallel for
        for(int ipix=0; ipix<Npix; ipix++){
            angvec_new[ipix] = 0.0;
            for(int ifreq=0; ifreq<Nfreq; ifreq++){
                int idx = ipix*Nfreq + ifreq;
                angvec_new[ipix] += map_signal_T[idx]*freqvec_new[ifreq]*inv_rms_2_T[idx];
            }
            double angvec_denominator = 0.0;
            for(int ifreq=0; ifreq<Nfreq; ifreq++){
                int idx = ipix*Nfreq + ifreq;
                angvec_denominator += freqvec_new[ifreq]*freqvec_new[ifreq]*inv_rms_2_T[idx];
            }
            if(angvec_denominator != 0.0){
                angvec_new[ipix] /= angvec_denominator;
            }
            else{
                angvec_new[ipix] = 0.0;
            }
        }

        for(int ipix=0; ipix<Npix; ipix++){
            angvec_diff_old = angvec[ipix] - angvec_old[ipix];
            angvec_diff_new = angvec_new[ipix] - angvec[ipix];
            angvec_momentum[ipix] += 0.03;
            if(angvec_diff_new*angvec_diff_old < 0){
                angvec_momentum[ipix] = 0.0;
            }
            if(angvec_momentum[ipix] > 0.9){
                angvec_momentum[ipix] = 0.9;
            }
        }

        for(int ipix=0; ipix<Npix; ipix++){
            angvec_new[ipix] += angvec_momentum[ipix]*(angvec_new[ipix] - angvec[ipix]);
        }


        double diff_angvec = 0.0;
        double diff_freqvec = 0.0;
        for(int ipix=0; ipix<Npix; ipix++){
            diff_angvec += fabs(angvec[ipix] - angvec_new[ipix]);
            angvec_old[ipix] = angvec[ipix];
            angvec[ipix] = angvec_new[ipix];
        }
        for(int ifreq=0; ifreq<Nfreq; ifreq++){
            diff_freqvec += fabs(freqvec[ifreq] - freqvec_new[ifreq]);
            freqvec_old[ifreq] = freqvec[ifreq];
            freqvec[ifreq] = freqvec_new[ifreq];
        }
        diff_freqvec /= Nfreq;
        diff_angvec /= Npix;
        if((diff_angvec < err_tol) && (diff_freqvec < err_tol)){
            return;
        }
    }

    free(angvec_new);
    free(angvec_old);
    free(freqvec_new);
    free(freqvec_old);
}


int main(){

    return 0;
}