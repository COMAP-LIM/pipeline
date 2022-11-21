// Compile as:
// g++ -shared -lm -fPIC -o PCAlib.so.1 PCAlib.cpp -O3 -std=c++11 -fopenmp -lpthread
// Make sure you have a new enough version of gcc to have openmp >4.5 (which support array reduction). Use a Sigurd module if needed :)

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <vector>

using namespace std;

inline double get_norm(double *vec, int size){
    double norm = 0.0;
    for(int i=0; i<size; i++)
        norm += vec[i]*vec[i];
    norm = sqrt(norm);
    return norm;
}


inline void normalize_vector(double *vec, int size){
    double norm = get_norm(vec, size);
    for(int i=0; i<size; i++)
        vec[i] = vec[i]/norm;
}


inline double dot_product(double *x, double *y, int size){
    double dot = 0.0;
    for(int i=0; i<size; i++){
        dot += x[i]*y[i];
    }
    return dot;
}

inline double dot_product_float_double(float *x, double *y, int size){
    double dot = 0.0;
    for(int i=0; i<size; i++){
        dot += x[i]*y[i];
    }
    return dot;
}


extern "C"
void PCA(float *X, double *r, double *err, int n, int p, int max_iter, double err_tol){
    double *s = new double[p];
    double lambda = 0.0;


    for(int iter=0; iter<max_iter; iter++){
        for(int i=0; i<p; i++){
            s[i] = 0.0;
        }
        #pragma omp parallel for reduction(+:s[:p])
        for(int row=0; row<n; row++){
            double xr_dot = dot_product_float_double(&X[row*p], r, p);
            for(int col=0; col<p; col++){
                s[col] += X[row*p + col]*xr_dot;
            }
        }

        lambda = dot_product(r, s, p);
        for(int col=0; col<p; col++){
            err[iter] += fabs(r[col] - s[col]/lambda)/n;
        }
        normalize_vector(s, p);
        for(int col=0; col<p; col++){
            r[col] = s[col];
        }
        cout << iter << " ";
        cout << lambda << " ";
        cout << err[iter] << endl;
        if(err[iter] < err_tol){
            for(int i=iter; i<max_iter; i++){
                err[i] = 0.0;
            }
            break;
        }
    }
}


int main(){
    int n = 10;
    int p = 5;
    int max_iter = 10;
	float *X = new float[n*p];
    double *r = new double[p];
    double *err = new double[max_iter];

    vector <double> X_vec;
    X_vec = {6, 3, 7, 4, 6, 9, 2, 6, 7, 4, 3, 7, 7, 2, 5, 4, 1, 7, 5, 1, 4, 0, 9, 5, 8, 0, 9, 2, 6, 3, 8, 2, 4, 2, 6, 4, 8, 6, 1, 3, 8, 1, 9, 8, 9, 4, 1, 3, 6, 7};

    for(int i=0; i<n*p; i++)
        X[i] = X_vec[i];

    for(int i=0; i<p; i++)
        r[i] = 1.0;
    normalize_vector(r, p);

    for(int y=0; y<n; y++){
        for(int x=0; x<p; x++){
            cout << X[y*p + x] << " ";
        }
        cout << endl;
    }
    for(int i=0; i<p; i++)
        cout << r[i] << endl;

    PCA(X, r, err, n, p, max_iter, 1e-12);

    for(int i=0; i<p; i++)
        cout << r[i] << endl;


    return 0;
}