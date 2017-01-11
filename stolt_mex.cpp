#define _USE_MATH_DEFINES
#include "mex.h"
#include <math.h> 
using namespace std;

mxArray* sinc(mxArray *x_IN){
	int M = mxGetM(x_IN);
	int N = mxGetN(x_IN);
	mxArray *res_OUT = mxCreateNumericMatrix(M,N,mxDOUBLE_CLASS,mxREAL);
	double *res = mxGetPr(res_OUT);
	double *x = mxGetPr(x_IN);
	for(int i=0;i<M*N;i++){
		if(x[i] == 0){
			res[i] = 1;
		}
		else{
			res[i] = sin(M_PI * x[i]) / (M_PI * x[i]);
		}
	}
	return res_OUT;
}
/*
	3D stolt interrupt
*/
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	#define Stolt_OUT plhs[0]
	#define S_matched_IN prhs[0]
	#define k_IN prhs[1]
	#define kx_IN prhs[2]
	#define ky_IN prhs[3]
	#define kz_interp_IN prhs[4]
	#define deltkr_IN prhs[5]
	#define Nx_IN prhs[6]
	#define Ny_IN prhs[7]
	#define Nf_IN prhs[8]
	#define kz_dim_IN prhs[9]
	#define kx_max_IN prhs[10]
	#define ky_max_IN prhs[11]
	#define p_IN prhs[12]
	
	double *S_matchedr = mxGetPr(S_matched_IN);
	double *S_matchedi = mxGetPi(S_matched_IN);
	double *k = mxGetPr(k_IN);
	double *kx = mxGetPr(kx_IN);
	double *ky = mxGetPr(ky_IN);
	double *kz_interp = mxGetPr(kz_interp_IN);
	double *deltkr = mxGetPr(deltkr_IN);
	double *Nx = mxGetPr(Nx_IN);
	double *Ny = mxGetPr(Ny_IN);
	double *kz_dim = mxGetPr(kz_dim_IN);
	double *kx_max = mxGetPr(kx_max_IN);
	double *ky_max = mxGetPr(ky_max_IN);
	double *p = mxGetPr(p_IN);

	size_t K = mxGetNumberOfDimensions(S_matched_IN);
	const mwSize *Ni = mxGetDimensions(S_matched_IN);
	// create identity cube
	mwSize *N;
	N[0] = Ny[0];
	N[1] = Nx[0];
	N[2] = kz_dim[0];
	mxArray *identity_IN = mxCreateNumericArray(K, N, mxDOUBLE_CLASS, mxREAL);
	double *identity = mxGetPr(identity_IN);
	for(int i=0;i<N[0]*N[1]*N[2];i++){
		identity[i] = k[0];
	}

	// create kz_interp cube
	mxArray *kz_interp_cub = mxCreateNumericArray(K, N, mxDOUBLE_CLASS, mxREAL);
	double *kzcub =  mxGetPr(kz_interp_cub);
	for(int i=0;i<N[0]*N[1]*N[2];i++){
		kzcub[i] = kz_interp[i%N[2]];
	}

	// create kx cube
	mxArray *kx_cub = mxCreateNumericArray(K, N, mxDOUBLE_CLASS, mxREAL);
	double *kxcub =  mxGetPr(kx_cub);
	for(int i=0;i<N[0]*N[1]*N[2];i++){
		kxcub[i] = kx[i%N[1]];
	}

	// create ky cube
	mxArray *ky_cub = mxCreateNumericArray(K, N, mxDOUBLE_CLASS, mxREAL);
	double *kycub =  mxGetPr(ky_cub);
	for(int i=0;i<N[0]*N[1]*N[2];i++){
		kycub[i] = ky[i%N[0]];
	}

	// create DKZ
	mxArray *DKZ_IN = mxDuplicateArray(identity_IN);
	double *DKZ =  mxGetPr(DKZ_IN);
	for(int i=0;i<N[0]*N[1]*N[2];i++){
		DKZ[i] = 0.5*sqrt(pow(kxcub[i],2)+pow(kycub[i],2)+pow(kzcub[i],2)) - identity[i];
	}

	// create NDKZ, find min and max at the same time
	mxArray *NDKZ_IN = mxDuplicateArray(identity_IN);
	double *NDKZ = mxGetPr(NDKZ_IN);
	double NDKZ_min = floor(DKZ[0]/deltkr[0]);
	double NDKZ_max = NDKZ_min;
	for(int i=0;i<N[0]*N[1]*N[2];i++){
		NDKZ[i] = floor(DKZ[i]/deltkr[0]);
		if(NDKZ[i] < NDKZ_min){
			NDKZ_min = NDKZ[i];
		}
		if(NDKZ[i] > NDKZ_max){
			NDKZ_max = NDKZ[i];
		}
	}
	
	// create B1
	mwSize *N_B1;
	N_B1[0] = Ny[0];
	N_B1[1] = Nx[0];
	N_B1[2] = NDKZ_max - NDKZ_min + p[0] + 1;
	mxArray *B1 = mxCreateNumericArray(K, N_B1, mxDOUBLE_CLASS, mxCOMPLEX);
	double *B1r = mxGetPr(B1);
	double *B1i = mxGetPi(B1);
	for(int i=0;i<N_B1[0]*N_B1[1]*N_B1[2];i++){
		B1r[i] = 0;
		B1i[i] = 0;
	}

	for(int i=0;i<Ni[0]*Ni[1]*N[2];i++){
		B1r[int(i+(-NDKZ_min)*Ni[0]*Ni[1])] = S_matchedr[i];
		B1i[int(i+(-NDKZ_min)*Ni[0]*Ni[1])] = S_matchedi[i];
	}

	// preprocessing
	mxArray *win_interp_IN = mxCreateNumericMatrix(1,2*p[0],mxDOUBLE_CLASS, mxREAL);
	double *win_interp = mxGetPr(win_interp_IN);
	for(int i=0;i<2*p[0];i++){
		win_interp[i] = 0.53836 - 0.46164*cos(2*M_PI*i/(2*p[0]-1));
	}
	mxArray *NN_IN = mxCreateNumericMatrix(1,2*p[0],mxDOUBLE_CLASS,mxREAL);
	double *NN = mxGetPr(NN_IN);
	mxArray *be4sinc_IN = mxCreateNumericMatrix(1,2*p[0],mxDOUBLE_CLASS,mxREAL);
	double *be4sinc = mxGetPr(be4sinc_IN);
	mxArray *B2 = mxCreateNumericMatrix(2*p[0],1,mxDOUBLE_CLASS,mxCOMPLEX);
	double *B2r = mxGetPr(B2);
	double *B2i = mxGetPi(B2);
	
	Stolt_OUT = mxCreateNumericArray(K, N, mxDOUBLE_CLASS, mxCOMPLEX);
	double *Stolt_OUTr = mxGetPr(Stolt_OUT);
	double *Stolt_OUTi = mxGetPi(Stolt_OUT);
	for(int i=0;i<Ny[0];i++){
		for(int j=0;j<Nx[0];j++){
			for(int nn=0;nn<2*p[0];nn++){
				NN[nn] = 0;
				be4sinc[nn] = 0;
				B2r[nn] = 0;
				B2i[nn] = 0;
			}
			for(int q = p[0] -1;q<kz_dim[0];q++){
				for(int k=0;k<2*p[0];k++){
					NN[k] = NDKZ[i + N[0]*(j + N[1]*q)]+k+1-p[0];
					be4sinc[k] = DKZ[i+ N[0]*(j + N[1]*q)]/deltkr[0] - NN[k];
					B2r[k] = B1r[int(i + N_B1[0]*(j + N_B1[1]*(NN[k]-NDKZ_min)))];
					B2i[k] = B1i[int(i + N_B1[0]*(j + N_B1[1]*(NN[k]-NDKZ_min)))];
				}

				// Stolt(i,j,q)=win_interp .* sinc( DKZ(i,j,q)/deltkr - NN )*B2; 
				mxArray *be4stolt = mxCreateNumericMatrix(1,2*p[0],mxDOUBLE_CLASS,mxCOMPLEX);
				double *be4stoltr = mxGetPr(be4stolt);
				double *be4stolti = mxGetPi(be4stolt);
				for(int ii=0;ii<2*p[0];ii++){
					mxArray *res_OUT = sinc(be4sinc_IN);
					double *res = mxGetPr(res_OUT);
					be4stoltr[ii] = win_interp[ii] * res[ii];
					be4stolti[ii] = 0;
				}
				double sumr = 0;
				double sumi = 0;
				for(int iii=0;iii<2*p[0];iii++){
					sumr += be4stoltr[iii]*B2r[iii];
					sumi += be4stolti[iii]*B2i[iii];
				}
				Stolt_OUTr[i+ N[0]*(j + N[1]*q)] = sumr;
				Stolt_OUTi[i+ N[0]*(j + N[1]*q)] = sumi;

				if(!((abs(kxcub[i+ N[0]*(j + N[1]*q)]) < kx_max[0]) && (kycub[i+ N[0]*(j + N[1]*q)] < ky_max[0]))) {
					Stolt_OUTr[i+ N[0]*(j + N[1]*q)] = 0;
					Stolt_OUTi[i+ N[0]*(j + N[1]*q)] = 0;
				}
			}
		}
	}
	return;
}
