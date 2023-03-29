g++ -shared -std=c++11 -fopenmp -lpthread -fPIC -o norm/normlib.so.1 norm/normlib.cpp -lfftw3_omp -lfftw3 -lm -O3
g++ -shared -std=c++11 -O3 -lm -lgsl -fopenmp -fPIC -o pointing/pointinglib.so.1 pointing/pointinglib.cpp
g++ -shared -std=c++11 -O3 -lm -lgsl -fopenmp -fPIC -o polyfit/polyfit.so polyfit/polyfit.cpp
g++ -shared -lm -fPIC -o PCA/PCAlib.so.1 PCA/PCAlib.cpp -O3 -std=c++11 -fopenmp -lpthread
g++ -shared tod2comap/mapbinner.cpp -o tod2comap/mapbinner.so.1 -fPIC -fopenmp -Ofast -std=c++11