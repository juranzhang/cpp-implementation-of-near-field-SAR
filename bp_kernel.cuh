#ifndef BP_KERNEL_H
#define BP_KERNEL_H

#include <math.h>
#include <armadillo>
using namespace arma;

#include <cuComplex.h>

cx_cube bp_kernel(cx_cube y_bp,vec x,vec y,vec z,vec x_array,vec y_array,mat R0_xy1,int Nx,int Ny,int ixn,int iyn,double k,double rs,double rstart,double rstop,double R0);

#endif
