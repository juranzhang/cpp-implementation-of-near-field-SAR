#define _USE_MATH_DEFINES
#include "mex.h"
#include <math.h> 
#include <ctime>
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
    time_t tstart = time(0);
    
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
    
	mexPrintf("Hello, world41!\n");
	double *S_matchedr = mxGetPr(S_matched_IN);
	double *S_matchedi = mxGetPi(S_matched_IN);
    mexPrintf("%f\n",S_matchedr[0]);
    mexPrintf("%f\n",S_matchedr[1]);
    mexPrintf("%f\n",S_matchedr[2]);
    mexPrintf("%f\n",S_matchedr[3]);
    mexPrintf("%f\n",S_matchedr[4]);
    mexPrintf("%f\n",S_matchedr[120]);
    mexPrintf("%f\n",S_matchedr[121]);
	double *k = mxGetPr(k_IN);
	double *kx = mxGetPr(kx_IN);
	double *ky = mxGetPr(ky_IN);
	double *kz_interp = mxGetPr(kz_interp_IN);
    //double Nx = 120;
	int Nx = mxGetScalar(Nx_IN);
    double deltkr = mxGetScalar(deltkr_IN);
	int Ny = mxGetScalar(Ny_IN);
	int kz_dim = mxGetScalar(kz_dim_IN);
	double kx_max = mxGetScalar(kx_max_IN);
	double ky_max = mxGetScalar(ky_max_IN);
	int p = mxGetScalar(p_IN);
    mexPrintf("Hello, world55!\n");
	size_t K = mxGetNumberOfDimensions(S_matched_IN);
	const mwSize *Ni = mxGetDimensions(S_matched_IN);
    mexPrintf("%d %d %d\n",Ni[0],Ni[1],Ni[2]);
	// create identity cube
    mexPrintf("%d\n",Ny);
    mexPrintf("%d\n",Nx);
    mexPrintf("%d\n",kz_dim);
    mwSize *N = new int [3];
    N[0] = Ny;
    N[1] = Nx;
    N[2] = kz_dim;
    mexPrintf("Hello, world63!\n");
	mxArray *identity_IN = mxCreateNumericArray(K, N, mxDOUBLE_CLASS, mxREAL);
	double *identity = mxGetPr(identity_IN);
	for(int i=0;i<N[0]*N[1]*N[2];i++){
		identity[i] = k[0];
	}
    mexPrintf("%f\n",k[0]);
    mexPrintf("Hello, world68!\n");
	// create kz_interp cube
	mxArray *kz_interp_cub = mxCreateNumericArray(K, N, mxDOUBLE_CLASS, mxREAL);
	double *kzcub =  mxGetPr(kz_interp_cub);
	for(int i=0;i<N[0]*N[1]*N[2];i++){
		kzcub[i] = kz_interp[i/(N[0]*N[1])];
	}
    mexPrintf("Hello, world75!\n");
	// create kx cube
	mxArray *kx_cub = mxCreateNumericArray(K, N, mxDOUBLE_CLASS, mxREAL);
	double *kxcub =  mxGetPr(kx_cub);
	for(int i=0;i<N[0]*N[1]*N[2];i++){
		kxcub[i] = kx[i%(N[0]*N[1])/N[1]];
	}
    mexPrintf("Hello, world89!\n");
	// create ky cube
	mxArray *ky_cub = mxCreateNumericArray(K, N, mxDOUBLE_CLASS, mxREAL);
	double *kycub =  mxGetPr(ky_cub);
	for(int i=0;i<N[0]*N[1]*N[2];i++){
		kycub[i] = ky[i%(N[0]*N[1])%N[1]];
	}
    
    mexPrintf("Hello, world96!\n");
	// create DKZ
	mxArray *DKZ_IN = mxDuplicateArray(identity_IN);
	double *DKZ =  mxGetPr(DKZ_IN);
	for(int i=0;i<N[0]*N[1]*N[2];i++){
		DKZ[i] = 0.5*sqrt(pow(kxcub[i],2)+pow(kycub[i],2)+pow(kzcub[i],2)) - identity[i];
	}
    mexPrintf("%f\n",DKZ[0]);
    mexPrintf("%f\n",DKZ[1]);
    mexPrintf("%f\n",DKZ[2]);
    mexPrintf("%f\n",DKZ[3]);
    mexPrintf("%f\n",DKZ[4]);
    mexPrintf("%f\n",DKZ[120]);
    mexPrintf("%f\n",DKZ[121]);
	// create NDKZ, find min and max at the same time
	mxArray *NDKZ_IN = mxDuplicateArray(identity_IN);
	double *NDKZ = mxGetPr(NDKZ_IN);
	double NDKZ_min = floor(DKZ[0]/deltkr);
	double NDKZ_max = NDKZ_min;
	for(int i=0;i<N[0]*N[1]*N[2];i++){
		NDKZ[i] = floor(DKZ[i]/deltkr);
		if(NDKZ[i] < NDKZ_min){
			NDKZ_min = NDKZ[i];
		}
		if(NDKZ[i] > NDKZ_max){
			NDKZ_max = NDKZ[i];
		}
	}
	mexPrintf("NDKZ_max%f\n",NDKZ_max);
    mexPrintf("NDKZ_min%f\n",NDKZ_min);
	// create B1
	mwSize *N_B1 = new int [3];
	N_B1[0] = Ny;
	N_B1[1] = Nx;
	//N_B1[2] = NDKZ_max - NDKZ_min + p + 1;
    N_B1[2] = 189;
	mxArray *B1 = mxCreateNumericArray(K, N_B1, mxDOUBLE_CLASS, mxCOMPLEX);
	double *B1r = mxGetPr(B1);
	double *B1i = mxGetPi(B1);
	for(int i=0;i<N_B1[0]*N_B1[1]*N_B1[2];i++){
		B1r[i] = 0;
		B1i[i] = 0;
	}
    mexPrintf("Hello, world131!\n");
	for(int i=0;i<Ni[0]*Ni[1]*Ni[2];i++){
		B1r[int(i+(-NDKZ_min)*Ni[0]*Ni[1])] = S_matchedr[i];
		B1i[int(i+(-NDKZ_min)*Ni[0]*Ni[1])] = S_matchedi[i];
	}
    mexPrintf("%f\n",S_matchedr[14400]);
    mexPrintf("%f\n",B1r[1080000]);
	// preprocessing
	mxArray *win_interp_IN = mxCreateNumericMatrix(1,2*p,mxDOUBLE_CLASS, mxREAL);
	double *win_interp = mxGetPr(win_interp_IN);
	for(int i=0;i<2*p;i++){
		win_interp[i] = 0.53836 - 0.46164*cos(2*M_PI*i/(2*p-1));
	}
	mxArray *NN_IN = mxCreateNumericMatrix(1,2*p,mxDOUBLE_CLASS,mxREAL);
	double *NN = mxGetPr(NN_IN);
	mxArray *be4sinc_IN = mxCreateNumericMatrix(1,2*p,mxDOUBLE_CLASS,mxREAL);
	double *be4sinc = mxGetPr(be4sinc_IN);
	mxArray *B2 = mxCreateNumericMatrix(1,2*p,mxDOUBLE_CLASS,mxCOMPLEX);
	double *B2r = mxGetPr(B2);
	double *B2i = mxGetPi(B2);
	mexPrintf("Hello, world150!\n");
	Stolt_OUT = mxCreateNumericArray(K, N, mxDOUBLE_CLASS, mxCOMPLEX);
	double *Stolt_OUTr = mxGetPr(Stolt_OUT);
	double *Stolt_OUTi = mxGetPi(Stolt_OUT);
    
	for(int i=0;i<Ny;i++){
		for(int j=0;j<Nx;j++){
			for(int nn=0;nn<2*p;nn++){
				NN[nn] = 0;
				be4sinc[nn] = 0;
				B2r[nn] = 0;
				B2i[nn] = 0;
			}
			for(int q = p -1;q<kz_dim;q++){
				for(int k=0;k<2*p;k++){
					NN[k] = NDKZ[i + N[0]*(j + N[1]*q)]+k+1-p;
					be4sinc[k] = DKZ[i+ N[0]*(j + N[1]*q)]/deltkr - NN[k];
					B2r[k] = B1r[int(i + N_B1[0]*(j + N_B1[1]*(NN[k]-NDKZ_min)))];
					B2i[k] = B1i[int(i + N_B1[0]*(j + N_B1[1]*(NN[k]-NDKZ_min)))];
				}

				// Stolt(i,j,q)=win_interp .* sinc( DKZ(i,j,q)/deltkr - NN )*B2; 
				mxArray *be4stolt = mxCreateNumericMatrix(1,2*p,mxDOUBLE_CLASS,mxCOMPLEX);
				double *be4stoltr = mxGetPr(be4stolt);
				double *be4stolti = mxGetPi(be4stolt);
				for(int ii=0;ii<2*p;ii++){
					mxArray *res_OUT = sinc(be4sinc_IN);
					double *res = mxGetPr(res_OUT);
					be4stoltr[ii] = win_interp[ii] * res[ii];
					be4stolti[ii] = 0;
				}
				double sumr = 0;
				double sumi = 0;
				for(int iii=0;iii<2*p;iii++){
					sumr += be4stoltr[iii]*B2r[iii];
					sumi += be4stolti[iii]*B2i[iii];
				}
				Stolt_OUTr[i+ N[0]*(j + N[1]*q)] = sumr;
				Stolt_OUTi[i+ N[0]*(j + N[1]*q)] = sumi;

				if(!((abs(kxcub[i+ N[0]*(j + N[1]*q)]) < kx_max) && (kycub[i+ N[0]*(j + N[1]*q)] < ky_max))) {
					Stolt_OUTr[i+ N[0]*(j + N[1]*q)] = 0;
					Stolt_OUTi[i+ N[0]*(j + N[1]*q)] = 0;
				}
			}
		}
	}
    delete [] N;
    delete [] N_B1;
    
    time_t tend = time(0)-tstart;
    mexPrintf("Run time: %d\n",tend);
	return;
}
