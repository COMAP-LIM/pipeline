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

        double diff_angvec = 0.0;
        double diff_freqvec = 0.0;
        for(int ipix=0; ipix<Npix; ipix++){
            diff_angvec += fabs(angvec[ipix] - angvec_new[ipix]);
            angvec[ipix] = angvec_new[ipix];
        }
        for(int ifreq=0; ifreq<Nfreq; ifreq++){
            diff_freqvec += fabs(freqvec[ifreq] - freqvec_new[ifreq]);
            freqvec[ifreq] = freqvec_new[ifreq];
        }
        // cout << diff_angvec/Npix << " " << diff_freqvec/Nfreq << endl;
    }
}


int main(){

    return 0;
}