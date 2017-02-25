#ifndef BP_KERNEL_H
#define BP_KERNEL_H

#include <math.h>
#include <armadillo>
using namespace arma;
#include <malloc.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuComplex.h>

extern "C" cx_fcube bp_kernel(cx_fcube y_bp,fvec x,fvec y,fvec z,fvec x_array,fvec y_array,fmat R0_xy1,int Nx,int Ny,int ixn,int iyn,float k,float rs,float rstart,float rstop,float R0);

#endif
