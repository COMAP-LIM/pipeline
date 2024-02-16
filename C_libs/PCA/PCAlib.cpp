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
void PCA(float *X, bool *mask, double *r, double *lambda, double *err, int n, int p, int max_iter, double err_tol){
    double *s = new double[p];

    for(int iter=0; iter<max_iter; iter++){
        for(int i=0; i<p; i++){
            s[i] = 0.0;
        }
        //#pragma omp parallel for reduction(+:s[:p])
        for(int row=0; row<n; row++){
            if(mask[row]){
                double xr_dot = dot_product_float_double(&X[row*p], r, p);
                for(int col=0; col<p; col++){
                    s[col] += X[row*p + col]*xr_dot;
                }
            }
        }

        lambda[iter] = dot_product(r, s, p);
        for(int col=0; col<p; col++){
            err[iter] += fabs(r[col] - s[col]/lambda[iter])/n;
        }
        normalize_vector(s, p);
        for(int col=0; col<p; col++){
            r[col] = s[col];
        }
        if(err[iter] < err_tol){
            for(int i=iter; i<max_iter; i++){
                err[i] = 0.0;
                lambda[i] = lambda[iter];
            }
            break;
        }
    }
}


extern "C"
void eigensolver(double *X, double *eigvecs, int p, int max_iter, double err_tol){
    double *s = new double[p];
    double *r = new double[p];
    double lambda = 0.0;

    for(int eigvec_num=0; eigvec_num<p; eigvec_num++){

        for(int i=0; i<p; i++){
            r[i] = 1.0;
        }
        double norm = get_norm(r, p);
        for(int i=0; i<p; i++){
            r[i] /= norm;
        }

        for(int iter=0; iter<max_iter; iter++){
            double err = 0.0;
            for(int i=0; i<p; i++){
                s[i] = 0.0;
            }
            for(int row=0; row<p; row++){
                s[row] = dot_product(&X[row*p], r, p);
            }

            lambda = dot_product(r, s, p);
            for(int col=0; col<p; col++){
                err += fabs(r[col] - s[col]/lambda)/p;
            }
            normalize_vector(s, p);
            for(int col=0; col<p; col++){
                r[col] = s[col];
            }
            if(err < err_tol){
                break;
            }
        }

        for(int i=0; i<p; i++){
            eigvecs[eigvec_num*p + i] = r[i];
        }

        for(int row=0; row<p; row++){
            for(int col=0; col<p; col++){
                X[row*p + col] -= lambda*r[row]*r[col];
            }
        }

    }
}


extern "C"
void PCA_lanczos(float *X, double *V, double *alphas, double *betas, int n, int p, int m){

    double *w = new double[p];
    double alpha = 0.0;
    double beta = 0.0;

    for(int row=0; row<n; row++){
        double xv_dot = dot_product_float_double(&X[row*p], V, p);
        for(int col=0; col<p; col++){
            w[col] += X[row*p + col]*xv_dot;
        }
    }

    alpha = dot_product(w, V, p);
    alphas[0] = alpha;
    for(int col=0; col<p; col++){
        w[col] -= alpha*V[col];
    }


    for(int iter=1; iter<m; iter++){
        beta = get_norm(w, p);
        betas[iter-1] = beta;

        for(int col=0; col<p; col++){
            V[iter*p + col] = w[col]/beta;
        }

        for(int col=0; col<p; col++){
            w[col] = 0.0;
        }
        for(int row=0; row<n; row++){
            double xv_dot = dot_product_float_double(&X[row*p], &V[iter*p], p);
            for(int col=0; col<p; col++){
                w[col] += X[row*p + col]*xv_dot;
            }
        }

        alpha = dot_product(w, &V[iter*p], p);
        alphas[iter] = alpha;

        for(int col=0; col<p; col++){
            w[col] = w[col] - alpha*V[iter*p + col] - beta*V[(iter-1)*p + col];
        }

        // Double re-orthogonalization
        for(int prev_iter=0; prev_iter<iter; prev_iter++){
            double inner_prod = dot_product(&V[iter*p], &V[prev_iter*p], p);
            for(int col=0; col<p; col++){
                V[iter*p + col] -= inner_prod*V[prev_iter*p + col];
            }
        }
        double norm = get_norm(&V[iter*p], p);
        for(int col=0; col<p; col++){
            V[iter*p + col] /= norm;
        }

        for(int prev_iter=0; prev_iter<iter; prev_iter++){
            double inner_prod = dot_product(&V[iter*p], &V[prev_iter*p], p);
            for(int col=0; col<p; col++){
                V[iter*p + col] -= inner_prod*V[prev_iter*p + col];
            }
        }
        norm = get_norm(&V[iter*p], p);
        for(int col=0; col<p; col++){
            V[iter*p + col] /= norm;
        }


        // Tried to implement the full calculate T and the eigenvectors thing, but gave up:
        // if(iter >= 5){
        //     double *T = new double[iter*iter];
        //     double *T_eigvecs = new double[iter*iter];

        //     for(int i=0; i<iter*iter; i++){
        //         T[i] = 0.0;
        //         T_eigvecs[i] = 0.0;
        //     }

        //     T[0] = alphas[0];
        //     T[1] = betas[0];
        //     for(int i=1; i<iter-1; i++){
        //         T[iter*i+i-1] = betas[i-1];
        //         T[iter*i+i] = alphas[i];
        //         T[iter*i+i+1] = betas[i];
        //     }

        //     eigensolver(T, T_eigvecs, iter, 200, 1e-12);

        // }


    }


}




extern "C"
void subtract_eigcomp(float *tod, double *r, double *ak, long Nouter, long Ntod){
    // for(int ifeed=0; ifeed<Nfeeds; ifeed++){
    //     for(int isb=0; isb<Nsb; isb++){
    //         for(int ifreq=0; ifreq<Nfreqs; ifreq++){
    //             int outer_idx = ifeed*Nfreqs*Nsb + isb*Nfreqs + ifreq;
    #pragma omp parallel for
    for(long outer_idx=0; outer_idx<Nouter; outer_idx++){
        ak[outer_idx] = dot_product_float_double(&tod[outer_idx*Ntod], r, Ntod);

        for(int itod=0; itod<Ntod; itod++){
            tod[outer_idx*Ntod + itod] -= r[itod]*ak[outer_idx];
        }
    }
}








int main(){
    // int n = 20000;
    // int p = 20*4*1024;

    // cout << p << " " << n << endl;

    // int max_iter = 10;
    // double error_tol = 1e-100;

	// float *X = new float[n*p];
    // double *r = new double[p];
    // double *err = new double[max_iter];

    // for(int i=0; i<n*p; i++){
    //     X[i] = i*0.001 - 1000;  // rand();
    // }

    // for(int i=0; i<p; i++)
    //     r[i] = rand();
    // normalize_vector(r, p);

    // cout << "Starting PCA" << endl;

    // PCA(X, r, err, n, p, max_iter, error_tol);

    // for(int i=0; i<max_iter; i++){
    //     cout << err[i] << endl;
    // }

    return 0;
}
