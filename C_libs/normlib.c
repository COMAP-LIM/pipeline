#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <fftw3.h>
#include <math.h>

void normalize(float* tod, int _Nfeed, int _Nsb, int _Nfreq, int _Ntod, int Nfast){
    printf("Entering ctypes...\n");
    unsigned long Ntod, Nouter, idx, iouter, itod;
    Ntod  = _Ntod;
    Nouter = _Nfeed*_Nsb*_Nfreq;

    double samprate = 50;
    double freq_knee = 0.01;
    double alpha = 4.0;

    unsigned long N = Nfast;
    double freqs[N];
    double W[N];

    for(int i=0; i<N/2; i++){
        freqs[i] = samprate*((double) i / (double) N);
    }
    for(int i=N/2; i<N; i++){
        freqs[i] = samprate*(-1.0 + (double) i / (double) N);
    }
    for(int i=0; i<N; i++){
        W[i] = 1.0/(1.0 + pow(freqs[i]/freq_knee, alpha));
    }

    fftw_init_threads();
    fftw_plan_with_nthreads(144);

    fftw_plan p, q;
    fftw_complex in[N];
    fftw_complex out[N];

    #pragma omp parallel for private(iouter, itod, idx, p, q, in, out)
    for(iouter=0; iouter<Nouter; iouter++){
        printf("%d start  ", iouter);
        // printf("1  ");
        double tod_endmean = 0;
        for(itod=0; itod<400; itod++){
            idx = iouter*Ntod + itod;
            tod_endmean += tod[idx];
        }
        // printf("2  ");
        tod_endmean /= 400;
        for(itod=0; itod<Ntod; itod++){
            idx = iouter*Ntod + itod;
            in[itod][0] = tod[idx];
            in[itod][1] = 0;
            in[2*Ntod-itod-1][0] = tod[idx];
            in[2*Ntod-itod-1][1] = 0;
        }
        for(itod=Ntod*2; itod<N; itod++){
            in[itod][0] = tod_endmean;
            in[itod][1] = 0;
        }
        
        // printf("3  ");
        #pragma omp critical
        p = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
        fftw_execute(p);
        // printf("4  ");
        #pragma omp critical
        fftw_destroy_plan(p);

        // printf("5  ");
        for(int i=0; i<N; i++){
            out[i][0] = out[i][0]*W[i];
            out[i][1] = out[i][1]*W[i];
        }

        #pragma omp critical
        q = fftw_plan_dft_1d(N, out, in, FFTW_BACKWARD, FFTW_ESTIMATE);
        // printf("6  ");
        fftw_execute(q);
        // printf("7  ");
        #pragma omp critical
        fftw_destroy_plan(q);
        // printf("8  ");

        for(int itod=0; itod<N; itod++){
            in[itod][0] = in[itod][0]/N;
            in[itod][1] = in[itod][1]/N;
        }
        // printf("9  ");

        for(int itod=0; itod<Ntod; itod++){
            idx = iouter*Ntod + itod;
            tod[idx] = tod[idx]/in[itod][0] - 1.0;
        }
        printf("%d stop  ", iouter);
    }
    fftw_cleanup();
    printf("Finished");
}

int main(){
    // float tod[1474560000];
    float* tod;
    tod = (float*) calloc(147456000, sizeof(float));
    for(int i=0; i<147456000; i++){
        // tod[i] = 0.1*i*(i-500) - i*0.5 - 10;
        tod[i] = 100;
    }

    normalize(tod, 18, 4, 1024, 2000, 4400);

    for(int i=0; i<100; i++){
        printf("%.2e   ", tod[i]);
    }


    return 0;
}