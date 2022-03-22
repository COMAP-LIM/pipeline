// This code is about equally fast as numpy when run from terminal, but much slower in ctypes??
// Not exactly sure if everything is needed:
// g++ -shared -std=c++11 -fopenmp -lpthread -fPIC -o normlib.so.1 normlib.cpp -lfftw3_omp -lfftw3 -lm -O3
// g++ -std=c++11 -fopenmp -lpthread -o norm normlib.cpp -lfftw3_omp -lfftw3 -lm -O3
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <fftw3.h>
// #include <fftw_threads.h>
#include <math.h>

extern "C"
void normalize(float* tod, unsigned long Nouter, unsigned long Ntod)
{
    double samprate = 50;
    double freq_knee = 0.01;
    double alpha = 4.0;

    double *freqs = new double[Ntod];
    double *W = new double[Ntod];

    for(int i=0; i<Ntod/2; i++){
        freqs[i] = samprate*((double) i / (double) Ntod);
    }
    for(int i=Ntod/2; i<Ntod; i++){
        freqs[i] = samprate*(-1.0 + (double) i / (double) Ntod);
    }
    for(int i=0; i<Ntod; i++){
        W[i] = 1.0/(1.0 + pow(freqs[i]/freq_knee, alpha));
    }

    omp_set_num_threads(48);
    fftw_plan_with_nthreads(48);

    fftw_plan p, q;
    fftw_complex *in, *out;
    in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Ntod*Nouter);
    out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Ntod*Nouter);

    #pragma omp parallel for
    for(int iouter=0; iouter<Nouter; iouter++)
    {
        for(int itod=0; itod<Ntod; itod++)
        {
            in[iouter*Ntod + itod][0] = tod[iouter*Ntod + itod];
            in[iouter*Ntod + itod][1] = 0.0;
        }
    }

    p = fftw_plan_dft_2d(Nouter, Ntod, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(p);
    fftw_destroy_plan(p);

    #pragma omp parallel for
    for(int iouter=0; iouter<Nouter; iouter++){
        for(int itod=0; itod<Ntod; itod++){
            out[iouter*Ntod + itod][0] = out[iouter*Ntod + itod][0]*W[itod];
            out[iouter*Ntod + itod][1] = out[iouter*Ntod + itod][1]*W[itod];
        }
    }

    q = fftw_plan_dft_2d(Nouter, Ntod, out, in, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(q);
    fftw_destroy_plan(q);

    #pragma omp parallel for
    for(int i=0; i<Ntod*Nouter; i++){
        in[i][0] = in[i][0]/(Ntod*Nouter);
        in[i][1] = in[i][1]/(Ntod*Nouter);
    }

    #pragma omp parallel for
    for(int iouter=0; iouter<Nouter; iouter++){
        for(int itod=0; itod<Ntod; itod++){
            int idx = iouter*Ntod + itod;
            tod[idx] = tod[idx]/in[idx][0] - 1.0;
        }
    }

    fftw_free(in);
    fftw_free(out);
    fftw_cleanup();
}

int main(){
    unsigned long Nouter, Ntod, Ntot;
    Nouter = 1024*4*18;
    Ntod = 40960;
    Ntot = Nouter*Ntod;

    float *tod = new float[Ntot];
    for(int i=0; i<Nouter; i++){
        for(int j=0; j<Ntod; j++){
            tod[i*Ntod + j] = (1.0+0.1*i)*sin(0.1*j) + 50.0;
        }
    }

    printf("START!\n");
    normalize(tod, Nouter, Ntod);
    printf("STOP!\n");

    // for(int i=0; i<200; i++){
    //     printf("%.4f\n", tod[i+2000]);
    // }


    return 0;
}