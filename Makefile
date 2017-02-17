# Makefile for near-field SAR
# algorithms: bp and wk
# fft libs used in wk: armadillo, fftw and cufttw

# CUFFTW paths. Change them according to the location of cufft on your machine
CUFFTW_INC_PATH = -I/usr/local/cuda/include
CUFFTW_LIB_PATH = -L/usr/local/cuda/lib64

# HDF5 paths.
HDF5_INC_PATH = -I/usr/include/hdf5/serial
HDF5_LIB_PATH = -L/usr/lib

all: ubuntu_WK_3D_imaging_experiment.cpp ubuntu_fftw_WK_3D_imaging_experiment.cpp ubuntu_cufftw_WK_3D_imaging_experiment.cpp

wk: ubuntu_WK_3D_imaging_experiment.cpp
	g++ ubuntu_WK_3D_imaging_experiment.cpp -o $@ -O2 -DARMA_DONT_USE_WRAPPER -DARMA_USE_BLAS -DARMA_USE_LAPACK -DARMA_USE_HDF5 -lblas -llapack -larmadillo -lhdf5 $(HDF5_INC_PATH) $(HDF5_LIB_PATH)

wk_fftw: ubuntu_fftw_WK_3D_imaging_experiment.cpp
	g++ ubuntu_fftw_WK_3D_imaging_experiment.cpp -o $@ -O2 -DARMA_DONT_USE_WRAPPER -DARMA_USE_BLAS -DARMA_USE_LAPACK -DARMA_USE_HDF5 -lblas -llapack -larmadillo -lhdf5 -lfftw3 -lm $(HDF5_INC_PATH) $(HDF5_LIB_PATH)

wk_cufftw: ubuntu_cufftw_WK_3D_imaging_experiment.cpp
	nvcc ubuntu_cufftw_WK_3D_imaging_experiment.cpp -o $@ -O2 -DARMA_DONT_USE_WRAPPER -DARMA_USE_BLAS -DARMA_USE_LAPACK -DARMA_USE_HDF5 -lblas -llapack -larmadillo -lhdf5 -lcufft -lcufftw -lm $(CUFFTW_INC_PATH) $(CUFFTW_LIB_PATH) $(HDF5_INC_PATH) $(HDF5_LIB_PATH)

clean:
	rm -rf *.o wk wk_fftw wk_cufftw
