// g++ -std=c++11 -lm -lgsl -fopenmp -o pointing pointinglib.cpp
// g++ -shared -std=c++11 -O2 -lm -lgsl -fopenmp -fPIC -o pointinglib.so.1 pointinglib.cpp
#include <stdio.h>
#include <stdlib.h>
#include <gsl/gsl_fit.h>
#include <gsl/gsl_multifit.h>
#include <omp.h>
#include <chrono>

using namespace std;
using namespace std::chrono;


extern "C"
void az_fit(double *az, float *tod, int Nouter, int Ntod, double *c, double *d)
// Performs linear fit "tod = az*d + c".
// tod consists of Nouter sequences of tods, each of length Ntod.
// az needs to be length Ntod, and same for all tods.
{
    #pragma omp parallel for
    for(int i=0; i<Nouter; i++)
    {
        double y_i[Ntod];
        for(int j=0; j<Ntod; j++)
        {
            y_i[j] = tod[j + i*Ntod];
        }
        double cov00, cov01, cov11, sumsq;
        gsl_fit_linear(az, 1, y_i, 1, Ntod, &c[i], &d[i], &cov00, &cov01, &cov11, &sumsq);
    }
}

extern "C"
void az_el_fit(double *az_term, double *el_term, float *tod, int Nouter, int Ntod, double *c, double *d, double *g)
// Same as above, but for the model "tod = g*el_term + d*az_term + c".
{
    #pragma omp parallel for
    for(int i=0; i<Nouter; i++)
    {
        gsl_matrix *X, *cov;
        gsl_vector *y, *coeffs;
        double chisq;

        X = gsl_matrix_alloc(Ntod, 3);
        y = gsl_vector_alloc(Ntod);
        coeffs = gsl_vector_alloc(3);
        cov = gsl_matrix_alloc(3, 3);
        gsl_multifit_linear_workspace *work = gsl_multifit_linear_alloc(Ntod, 3);

        for(int i=0; i<Ntod; i++)
        {
            gsl_matrix_set(X, i, 0, 1.0);
            gsl_matrix_set(X, i, 1, az_term[i]);
            gsl_matrix_set(X, i, 2, el_term[i]);
        }
        for(int j=0; j<Ntod; j++)
        {
            gsl_vector_set(y, j, tod[j + i*Ntod]);
        }
        gsl_multifit_linear(X, y, coeffs, cov, &chisq, work);
        c[i] = gsl_vector_get(coeffs, 0);
        d[i] = gsl_vector_get(coeffs, 1);
        g[i] = gsl_vector_get(coeffs, 2);

        gsl_multifit_linear_free (work);
        gsl_matrix_free(X);
        gsl_vector_free(y);
        gsl_vector_free(coeffs);
        gsl_matrix_free(cov);
    }
}





int main(void){

    int Nfreq = 1024;
    int Ntod = 20000;
    double *az_term = new double[Ntod];
    double *el_term = new double[Ntod];
    float *tod = new float[Nfreq*Ntod];
    for(int i=0; i<Ntod; i++)
    {
        az_term[i] = i;
        el_term[i] = i;
    }

    for(int i=0; i<Nfreq; i++)
    {
        for (int j=0; j<Ntod; j++)
        {
            tod[i*Ntod+j] = 1.5*j + 0.5 + (rand()%1000)/1000.0;
        }
    }
    double *c0 = new double[Nfreq];
    double *c1 = new double[Nfreq];
    double *c2 = new double[Nfreq];

    auto start = high_resolution_clock::now();
    az_el_fit(az_term, el_term, tod, Nfreq, Ntod, c0, c1, c2);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    printf("Parallel time taken: %lld milliseconds\n", duration.count());


    return 0;
}