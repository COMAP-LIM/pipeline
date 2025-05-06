// g++ -std=c++11 -lm -lgsl -fopenmp -o test_polyfit test_polyfit.cpp
// g++ -shared -std=c++11 -O2 -lm -lgsl -fopenmp -fPIC -o test_polyfit.so test_polyfit.cpp
#include <stdio.h>
#include <stdlib.h>
#include <gsl/gsl_fit.h>
#include <omp.h>
#include <chrono>

using namespace std;
using namespace std::chrono;


extern "C"
void polyfit(float *x, float *y, double *w, int Nfreq, int Ntod, double *c0, double *c1)
{
    double x_i[Nfreq];
    for(int i=0; i<Nfreq; i++)
    {
        x_i[i] = x[i];
    }
    #pragma omp parallel for
    for(int i=0; i<Ntod; i++)
    {
        double y_i[Nfreq];
        for(int j=0; j<Nfreq; j++)
        {
            y_i[j] = y[i+j*Ntod];
        }
        double cov00, cov01, cov11, sumsq;
        gsl_fit_wlinear(x_i, 1, w, 1, y_i, 1, Nfreq, &c0[i], &c1[i], &cov00, &cov01, &cov11, &sumsq);
    }
}

extern "C"
void polyfit_float64(double *x, double *y, double *w, int Nfreq, int Ntod, double *c0, double *c1)
{

    #pragma omp parallel for
    for(int i=0; i<Ntod; i++)
    {
        double y_i[Nfreq];
        for(int j=0; j<Nfreq; j++)
        {
            y_i[j] = y[i+j*Ntod];
        }
        double cov00, cov01, cov11, sumsq;
        gsl_fit_wlinear(x, 1, w, 1, y_i, 1, Nfreq, &c0[i], &c1[i], &cov00, &cov01, &cov11, &sumsq);
    }
}




int main(void){

    int Nfreq = 1024;
    int Ntod = 20000;
    float *x = new float[Nfreq];
    float *y = new float[Nfreq*Ntod];
    double *w = new double[Nfreq];
    for(int i=0; i<Nfreq; i++)
    {
        for (int j=0; j<Ntod; j++)
        {
            y[i*Ntod+j] = 1.5*j + 0.5 + (rand()%1000)/1000.0;
        }
    }
    for(int i=0; i<Nfreq; i++)
    {
        x[i] = i;
        w[i] = 1.0;
    }


    double *c0 = new double[Ntod];
    double *c1 = new double[Ntod];

    auto start = high_resolution_clock::now();
    polyfit(x, y, w, Nfreq, Ntod, c0, c1);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    printf("Parallel time taken: %lld milliseconds\n", duration.count());


    return 0;
}