#include <hdf5.h>
#include <armadillo>
using namespace arma;

#include <iostream>
#include <math.h>
#include <ctime>
#include <stdlib.h>
using namespace std;

#define _USE_MATH_DEFINES
#define MAX_ZERO_PADDING 5
#define TARGET_IMAGE_BACKGROUND -30

/*
	downsample a cube to a smaller cube. dim indicates the direction of downsampling.
	row 1, col 2, slice 3.
*/
cube downsample(cube S_echo,uword downsample_factor, int dim){
	cube res;
	uword new_slice;
	if(dim == 1) {

	}

	if(dim == 2) {

	}

	if(dim == 3) {
		new_slice = (S_echo.n_slices-1)/downsample_factor + 1;
		res.set_size(S_echo.n_rows,S_echo.n_cols,new_slice);
		for(uword i=0;i<S_echo.n_rows;i++){
			for(uword j=0;j<S_echo.n_cols;j++){
				for(uword k=0;k<new_slice;k++){
					res(i,j,k) = S_echo(i,j,k*downsample_factor);
				}
			}
		}
	}
	return res;

}

/*
	replicate a vector to a cube with dim x,y,z.
	z equals to the size of the vector.
*/
cube vec2cub_xy(vec z,uword x,uword y){
	cube res(x,y,z.n_elem);
	for(uword i=0;i<x;i++){
		for(uword j=0;j<y;j++){
			for(uword k=0;k<z.n_elem;k++){
				res(i,j,k) = z(k);
			}
		}
	}
	return res;
}

/*
	replicate a vector to a cube with dim x,y,z.
	x equals to the size of the vector.
*/
cube vec2cub_yz(vec x,uword y,uword z){
	cube res(x.n_elem,y,z);
	for(uword i=0;i<x.n_elem;i++){
		for(uword j=0;j<y;j++){
			for(uword k=0;k<z;k++){
				res(i,j,k) = x(i);
			}
		}
	}
	return res;
}

/*
	replicate a vector to a cube with dim x,y,z.
	y equals to the size of the vector.
*/
cube vec2cub_xz(vec y,uword x,uword z){
	cube res(x,y.n_elem,z);
	for(uword i=0;i<x;i++){
		for(uword j=0;j<y.n_elem;j++){
			for(uword k=0;k<z;k++){
				res(i,j,k) = y(j);
			}
		}
	}
	return res;
}

/*
	reshape cube from x-y-z to z-x-y
*/
template <class cubeType>
cubeType reshape_zxy(cubeType echo){
	cubeType res(echo.n_slices,echo.n_rows,echo.n_cols);
	for(uword i=0;i<res.n_rows;i++){
		for(uword j=0;j<res.n_cols;j++){
			for(uword k=0;k<res.n_slices;k++){
				res(i,j,k) = echo(j,k,i);
			}
		}
	}
	return res;
}

/*
	reshape cube from x-y-z to y-z-x
*/
template <class cubeType>
cubeType reshape_yzx(cubeType echo){
	cubeType res(echo.n_cols,echo.n_slices,echo.n_rows);
	for(uword i=0;i<res.n_rows;i++){
		for(uword j=0;j<res.n_cols;j++){
			for(uword k=0;k<res.n_slices;k++){
				res(i,j,k) = echo(k,i,j);
			}
		}
	}
	return res;
}

// a cpp implementation of matlab function floor()
cube floor_cube(cube x) {
	cube res(size(x));
	for (uword i =0;i<x.n_rows;i++){
		for(uword j=0;j<x.n_cols;j++){
			for(uword k=0;k<x.n_slices;k++){
				res(i,j,k) = floor(x(i,j,k));
			}
		}
	}
	return res;
}

// convert 1*1 complex matrix to complex double
cx_double mat2cx_double(cx_mat x){
	cx_double res(x(0,0).real(),x(0,0).imag());
	return res;
}

// sinc function, both input and output are 1*N
mat sinc(mat x){
	mat res(size(x));
	for(uword i =0;i<x.n_elem;i++){
		if(x(0,i) == 0){
			res(0,i) = 1;
		}
		else{
			res(0,i) = sin(M_PI * x(0,i)) / (M_PI * x(0,i));
		}
	}
	return res;
}

// hamming window function, return a mat(1,L)
mat hamming(uword L){
	mat res(1,L);
	for(uword i=0;i<L;i++){
		res(0,i) = 0.53836 - 0.46164*cos(2*M_PI*i/(L-1));
	}
	return res;
}

/*
	a cpp implementation of matlab function fftshift
	dim == 1 means row operation
	dim == 2 means col operation
*/
cx_mat fftshift(cx_mat x,int dim){
	cx_mat res(size(x));
	if(dim==1){
		uword mid = x.n_rows/2;
		res.rows(0,mid-1) = x.rows(x.n_rows-mid,x.n_rows-1);
		res.rows(mid,x.n_rows-1) = x.rows(0,x.n_rows-mid-1);
	}
	else{
		uword mid = x.n_cols/2;
		res.cols(0,mid-1) = x.cols(x.n_cols-mid,x.n_cols-1);
		res.cols(mid,x.n_cols-1) = x.cols(0,x.n_cols-mid-1);
	}
	return res;
}

/*
	a cpp implementation of matlab function ifftshift
	dim == 1 means row operation
	dim == 2 means col operation
*/
cx_mat ifftshift(cx_mat x,int dim){
	cx_mat res(size(x));
	if(dim==1){
		uword mid = x.n_rows/2;
		res.rows(0,x.n_rows-mid-1) = x.rows(mid,x.n_rows-1);
		res.rows(x.n_rows-mid,x.n_rows-1) = x.rows(0,mid-1);
	}
	else{
		uword mid = x.n_cols/2;
		res.cols(0,x.n_cols-mid-1) = x.cols(mid,x.n_cols-1);
		res.cols(x.n_cols-mid,x.n_cols-1) = x.cols(0,mid-1);
	}
	return res;
}

/*
	3D match filtering
*/
cx_cube match_filter_3D(cx_cube S_kxy,vec k,vec kx,vec ky,double R0){
	uword Nx = S_kxy.n_cols;
	uword Ny = S_kxy.n_rows;
	uword Nf = S_kxy.n_slices;

	cube k_cub = vec2cub_xy(2*k,Ny,Nx);
	cube kx_cub = vec2cub_xz(kx,Ny,Nf);
	cube ky_cub = vec2cub_yz(ky,Nx,Nf);
	cube kz_sq = pow(k_cub,2)-pow(kx_cub,2)-pow(ky_cub,2);

	cube zero_cub(size(kz_sq),fill::zeros);
	kz_sq = arma::max(kz_sq,zero_cub);
	cube Fmf = (R0+0)*sqrt(kz_sq);
	cx_cube Fmf_exp(cos(Fmf),sin(Fmf));

	cx_cube S_matched = S_kxy % Fmf_exp;

	return S_matched;
}

/*
	3D stolt interrupt
*/
cx_cube stolt_interrupt(cx_cube S_matched,vec k,vec kx,vec ky,vec kz_interp,double deltkr,double kx_max,double ky_max,uword p){
	uword Nx = S_matched.n_cols;
	uword Ny = S_matched.n_rows;
	uword Nf = S_matched.n_slices;
	uword kz_dim = kz_interp.n_elem;

	cube kz_interp_cub = vec2cub_xy(kz_interp,Ny,Nx);
	cube kx_cub = vec2cub_xz(kx,Ny,kz_dim);
	cube ky_cub = vec2cub_yz(ky,Nx,kz_dim);

	cube identity = k(0) * ones<cube>(Ny, Nx, kz_dim);

	cube DKZ = 0.5*sqrt(pow(kx_cub,2) + pow(ky_cub,2) + pow(kz_interp_cub,2)) - identity;
	cube NDKZ = floor_cube(DKZ/deltkr);

	double NDKZ_min = NDKZ.min();
	double NDKZ_max = NDKZ.max();

	cx_cube B1(Ny,Nx,NDKZ_max-NDKZ_min+p+1,fill::zeros);
	B1(0,0,-NDKZ_min,size(Ny,Nx,Nf)) = S_matched;

	mat win_interp = hamming(2*p);
	mat NN(1,2*p);
	mat be4sinc(size(NN));
	cx_mat B2(NN.n_elem,1);
	cx_cube Stolt(Ny,Nx,kz_dim,fill::zeros);

	for(uword i=0;i<Ny;i++) {
		for(uword j=0;j<Nx;j++){
			NN.zeros();
			be4sinc.zeros();
			B2.zeros();
			for(uword q=p-1;q<kz_dim;q++){
				for(uword k=0;k<NN.n_elem;k++){
					NN(0,k) = NDKZ(i,j,q)+k+1-p;
					be4sinc(0,k) = DKZ(i,j,q)/deltkr - NN(0,k);
					B2(k,0) = B1(i,j,NN(0,k)-NDKZ_min);
				}
				mat be4stolt_real = win_interp % sinc(be4sinc);
				mat be4stolt_imag(size(be4stolt_real), fill::zeros);
				cx_mat be4stolt(be4stolt_real,be4stolt_imag);
				Stolt(i,j,q) = mat2cx_double(be4stolt * B2);

				if(!((abs(kx_cub(i,j,q)) < kx_max) && (ky_cub(i,j,q) < ky_max))) {
					Stolt(i,j,q).real(0.0);
					Stolt(i,j,q).imag(0.0);
				}

			}

		}
	}
	return Stolt;
}

int main(int argc, char* argv[]) {
	// time start
	time_t tstart, tend;
	tstart = time(0);

	// load data from .hdf5 files
	cube secho_real;
	secho_real.load("secho_real.h5",hdf5_binary);
	cube secho_imag;
	secho_imag.load("secho_imag.h5",hdf5_binary);

	tend = time(0);
	cout << "Data load took "<< difftime(tend, tstart) <<" second(s)."<< endl;

	// reshape from x-y-z to z-x-y
	cube S_echo_real = reshape_zxy<cube>(secho_real);
	cube S_echo_imag = reshape_zxy<cube>(secho_imag);

	// downsampleing to reduce dim
	uword Nf_downsample_factor = 12;
	S_echo_real = downsample(S_echo_real,Nf_downsample_factor,3);
	S_echo_imag = downsample(S_echo_imag,Nf_downsample_factor,3);
	S_echo_real = S_echo_real(span(29,148),span(59,178),span::all);
	S_echo_imag = S_echo_imag(span(29,148),span(59,178),span::all);

	tend = time(0);
	cout << "Reshaping and downsampling took "<< difftime(tend, tstart) <<" second(s)."<< endl;

	// pre-processing and system delay
	cx_cube S_echo(S_echo_real,S_echo_imag);
	uword Nx = S_echo.n_cols;
	uword Ny = S_echo.n_rows;
	uword Nf = S_echo.n_slices;

	double c = 299792458;
	double Theta_antenna = 40*M_PI/180;
	double f_start = 92000000000;
	double f_stop = 93993750000;
	double deltf = Nf_downsample_factor * 6250000;

	double B = f_stop - f_start;
	Nf = floor(B/deltf)+1;
	f_stop = f_start + (Nf-1)*deltf;
	B = f_stop - f_start;
	vec freq = linspace<vec>(f_start,f_stop,Nf);
	vec k = 2 * M_PI * freq/c;
	double deltkr=k(1)-k(0);
	double deltkz = 2 * deltkr;

	double R0 = 1;
	double dx = 0.003;
	double dy = 0.003;

	double system_delay = 0.4;

	cube freq_cub = vec2cub_xy(freq,Ny,Nx);
	cx_cube delay(cos(2*M_PI*freq_cub*2*system_delay/c),sin(2*M_PI*freq_cub*2*system_delay/c));

	S_echo = S_echo % delay;

	vec kx = linspace(-M_PI/dx, M_PI/dx - 2*M_PI/dx/Nx, Nx);
	vec ky = linspace(-M_PI/dy, M_PI/dy - 2*M_PI/dy/Ny, Ny);

	cx_cube S_kxy(size(S_echo));

	tend = time(0);
	cout << "Pre-processing took "<< difftime(tend, tstart) <<" second(s)."<< endl;

	// fftshift and fft2 for each slice
	for(uword k=0;k<S_echo.n_slices;k++){
		S_echo.slice(k) = fftshift(S_echo.slice(k),1);
		S_echo.slice(k) = fftshift(S_echo.slice(k),2);
		S_kxy.slice(k) = fft2(S_echo.slice(k));
		S_kxy.slice(k) = fftshift(S_kxy.slice(k),1);
		S_kxy.slice(k) = fftshift(S_kxy.slice(k),2);
	}

	cx_cube S_matched = match_filter_3D(S_kxy,k,kx,ky,R0);

	tend = time(0);
	cout << "Match filter took "<< difftime(tend, tstart) <<" second(s)."<< endl;

	// Stolt interrupt
	uword p = 4;
	double kz_interp_min = 2*k(0)*cos(Theta_antenna/2);
	double kz_interp_max = 2*k(k.n_elem-1);
	vec kz_interp = linspace(kz_interp_min,kz_interp_max);
	uword kz_dim = kz_interp.n_elem;
	double kx_max = 2*k(0)*sin(Theta_antenna/2);
	double ky_max = 2*k(0)*sin(Theta_antenna/2);

	cx_cube Stolt = stolt_interrupt(S_matched,k,kx,ky,kz_interp,deltkr,kx_max,ky_max,p);

	tend = time(0);
	cout << "Stolt interrupt took "<< difftime(tend, tstart) <<" second(s)."<< endl;

	// pad zeros to increase dimmension of Stolt result
	int zeroPadding;
	if(argc == 1){
		cout<<"Zero padding is not specified, using default 3."<<endl;
		zeroPadding = 3;
	}
	else if(atoi(argv[1]) > MAX_ZERO_PADDING){
		cout<<"Please specify smaller number, using default 3."<<endl;
		zeroPadding = 3;
	}
	else{
		cout<<"Zero padding is "<<argv[1]<<"."<<endl;
		zeroPadding = atoi(argv[1]);
	}
	uword point_number = max(max(Nx,Ny),kz_dim) * zeroPadding;

	cx_cube complex_image_cx(point_number,point_number,point_number,fill::zeros);
	complex_image_cx(0,0,0,size(Ny,Nx,kz_dim)) = Stolt;

	// ifftn
	for(uword k=0;k<complex_image_cx.n_slices;k++){
		complex_image_cx.slice(k) = ifft2(complex_image_cx.slice(k));
	}

	tend = time(0);
	cout << "ifftn first 2 dims took "<< difftime(tend, tstart) <<" second(s)."<< endl;

	// third dimension ifft of ifftn
	cx_mat x_slice(complex_image_cx.n_slices,complex_image_cx.n_cols);
	for(uword i=0;i<complex_image_cx.n_rows;i++){
		for(uword j=0;j<complex_image_cx.n_slices;j++){
			for(uword k=0;k<complex_image_cx.n_cols;k++){
				x_slice(j,k) = complex_image_cx(i,k,j);
			}
		}
		x_slice = ifft(x_slice);
		for(uword j=0;j<complex_image_cx.n_slices;j++){
			for(uword k=0;k<complex_image_cx.n_cols;k++){
				complex_image_cx(i,k,j) = x_slice(j,k);
			}
		}
	}

	tend = time(0);
	cout << "ifftn last dim took "<< difftime(tend, tstart) <<" second(s)."<< endl;

	// ifftshift 3D
	for(uword k=0;k<complex_image_cx.n_slices;k++){
		complex_image_cx.slice(k) = ifftshift(complex_image_cx.slice(k),1);
		complex_image_cx.slice(k) = ifftshift(complex_image_cx.slice(k),2);
	}

	for(uword i=0;i<complex_image_cx.n_rows;i++){
		for(uword j=0;j<complex_image_cx.n_slices;j++){
			for(uword k=0;k<complex_image_cx.n_cols;k++){
				x_slice(j,k) = complex_image_cx(i,k,j);
			}
		}
		x_slice = ifftshift(x_slice,1);
		for(uword j=0;j<complex_image_cx.n_slices;j++){
			for(uword k=0;k<complex_image_cx.n_cols;k++){
				complex_image_cx(i,k,j) = x_slice(j,k);
			}
		}
	}

	tend = time(0);
	cout << "ifftshift 3D took "<< difftime(tend, tstart) <<" second(s)."<< endl;

	// Target image generation
	cube complex_image = abs(complex_image_cx);
	complex_image = complex_image/complex_image.max();
	complex_image = 20*log10(complex_image);

	tend = time(0);
	cout << "log operation took "<< difftime(tend, tstart) <<" second(s)."<< endl;

	for(uword i=0;i<complex_image.n_rows;i++){
		for(uword j=0;j<complex_image.n_cols;j++){
			for(uword k=0;k<complex_image.n_slices;k++){
				if(complex_image(i,j,k) < TARGET_IMAGE_BACKGROUND){
					complex_image(i,j,k) = TARGET_IMAGE_BACKGROUND;
				}
			}
		}
	}

	tend = time(0);
	cout << "Set background took "<< difftime(tend, tstart) <<" second(s)."<< endl;

	uword index = 0;
	double max_cube = TARGET_IMAGE_BACKGROUND;
	for(uword k=0;k<complex_image.n_slices;k++){
		if (complex_image.slice(k).max() > max_cube){
			max_cube = complex_image.slice(k).max();
			index = k;
		}
	}

	tend = time(0);
	cout << "Find max index took "<< difftime(tend, tstart) <<" second(s)."<< endl;

	mat resulting_image(complex_image.n_cols,complex_image.n_slices);
	for(uword j=0;j<complex_image.n_rows;j++){
		for(uword k=0;k<complex_image.n_cols;k++){
			resulting_image(j,k) = complex_image(j,k,index);
		}
	}

	resulting_image.save("resulting_image.txt",arma_ascii);

	tend = time(0);
	cout << "Data store took "<< difftime(tend, tstart) <<" second(s)."<< endl;
	return 0;
}
