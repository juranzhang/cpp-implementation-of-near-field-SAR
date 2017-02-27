#ifndef BP_KERNEL_H
#define BP_KERNEL_H

#include <math.h>
#include <armadillo>
using namespace arma;
#include <malloc.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuComplex.h>

void bp_kernel(float*& bp_real_host,float*& bp_imag_host,fcube& bp_real, fcube& bp_imag,fcube y_bp_real,fcube y_bp_imag,int ybpw,int ybph,int ybpd,fvec x,fvec y,fvec z,fvec x_array,fvec y_array,fmat R0_xy1,int Nx,int Ny,int ixn,int iyn,float k,float rs,float rstart,float rstop,float R0);

#endif
