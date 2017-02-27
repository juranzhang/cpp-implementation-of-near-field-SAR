#define _USE_MATH_DEFINES
#include <hdf5.h>
#include <armadillo>
using namespace arma;

#include <iostream>
#include <math.h>
#include <climits>
#include <ctime>
#include <stdlib.h>
using namespace std;

//#include <omp.h>

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
	dim == 1 means row operation (along each column vector)
	dim == 2 means col operation (along each row vector)
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
	cpp implementation of matlab function nextpow2
*/
uword nextpow2(uword Nf){
	int i=0;
	while(pow(2,i)<INT_MAX && pow(2,i) < Nf){
		i++;
	}
	return i;
}

int main() {
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
	cout<<S_echo(0)<<S_echo(130)<<endl;
	cout<<Nx<<Ny<<Nf<<endl;
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

	vec x_array = linspace<vec>(-60,59,Nx) * dx;
	vec y_array = linspace<vec>(60,-59,Ny) * dy;

	double system_delay = 0.4;

	cube freq_cub = vec2cub_xy(freq,Ny,Nx);
	cx_cube delay(cos(2*M_PI*freq_cub*2*system_delay/c),sin(2*M_PI*freq_cub*2*system_delay/c));

	S_echo = S_echo % delay;
	cout<<S_echo(0)<<S_echo(130)<<endl;
	vec kx = linspace(-M_PI/dx, M_PI/dx - 2*M_PI/dx/Nx, Nx);
	vec ky = linspace(-M_PI/dy, M_PI/dy - 2*M_PI/dy/Ny, Ny);

	tend = time(0);
	cout << "Pre-processing took "<< difftime(tend, tstart) <<" second(s)."<< endl;

	// RCMC
	uword FNf = pow(2,nextpow2(Nf)) * 1;
	mat R0_xy1(Ny,Nx,fill::zeros);
	for(uword i=0;i<Ny;i++){
		for(uword j=0;j<Nx;j++){
			R0_xy1(i,j) = sqrt(pow(y_array(i),2)+pow(x_array(j),2)+pow(R0,2));
		}
	}
  cube R0_xy1_cub(Ny,Nx,Nf);
  R0_xy1_cub.each_slice() = R0_xy1;

  cube K0(Ny,Nx,Nf);

  for(uword l=0;l<Nf;l++){
  	for(uword i=0;i<Ny;i++){
  		for(uword j=0;j<Nx;j++){
  			K0(i,j,l) = k(l)-k(0);
  		}
  	}
  }

  cube tmp = 2*K0 % R0_xy1_cub;
  cx_cube tmp_exp(cos(tmp),sin(tmp));
  S_echo = S_echo % tmp_exp;
  cout<<S_echo(0)<<S_echo(130)<<endl;
  /*
  	IFFT along the z-dimension
  	reshape to do ifft on x-dimension
  	then reshape back
  	this might be faster than extracting each slice along z-axis
  	TODO: verify the above statement
  */
  cx_cube S_echo_ifft = reshape_zxy<cx_cube>(S_echo);
  cx_cube y_bp_ifft(FNf,Ny,Nx);

  for(uword i=0;i<S_echo_ifft.n_slices;i++){
  	y_bp_ifft.slice(i) = ifft(S_echo_ifft.slice(i),FNf)*FNf/Nf;
  	y_bp_ifft.slice(i) = fftshift(y_bp_ifft.slice(i),1);
  }

  cx_cube y_bp = reshape_yzx<cx_cube>(y_bp_ifft);
  cout<<y_bp(0)<<y_bp(130)<<endl;
	tend = time(0);
  cout << "ifft and fftshift took "<< difftime(tend, tstart) <<" second(s)."<< endl;

  // range
  double maxr = c / (2*deltf);
  double rs = maxr / (FNf -1);
  double rstart = -maxr/2;
  double rstop = maxr/2;

  double x_image_zone = 2 * x_array.max();
  cout<<x_image_zone<<endl;
  uword ixn = floor(x_image_zone/dx * 2);
  if(ixn%2 == 0){
  	ixn=ixn+1;
  }
  uword iyn = ixn;
  cout<<ixn<<endl;
  vec x = linspace<vec>(-x_image_zone/2,x_image_zone/2,ixn);
	vec y = linspace<vec>(-x_image_zone/2,x_image_zone/2,iyn);
	vec z = linspace<vec>(-x_image_zone/2,x_image_zone/2,iyn);
	cout<<x(0)<<x(5)<<endl;
	cx_cube bp_image(iyn,ixn,iyn);
	double Ri,lr,yi_be4_exp;
	cx_double yi,yi_cx;
	uword l1,l2;
	// int nthreads;
	for(uword i=0;i<ixn;i++){
		cout<<"x_i: "<<i<<endl;
		for(uword j=0;j<iyn;j++){
			cout<<"y_i: "<<j<<endl;
			for(uword l=0;l<iyn;l++){
				//cout<<"z_i: "<<l<<endl;
				//#pragma omp parallel for collapse(2)
				for(uword m=0;m<Ny;m++){
					//cout<<"m: "<<m<<endl;
					for(uword n=0;n<Nx;n++){
						// nthreads = omp_get_num_threads();
	      		// printf("Number of threads = %d\n", nthreads);
						//cout<<"n: "<<n<<endl;
						Ri = sqrt(pow(x(i)-x_array(n),2) + pow(y(j) - y_array(m),2) + pow(z(l)+R0,2)) - R0_xy1(m,n);
						l1 = floor((Ri-rstart)/rs);
						l2 = ceil((Ri-rstart)/rs);
						lr = (Ri-rstart)/rs;
						//cout<<l1<<l2<<lr<<endl;
						yi = y_bp(m,n,l1) + (lr-l1)*(y_bp(m,n,l2) - y_bp(m,n,l1));
						// cout<<y_bp(m,n,l2)<<y_bp(m,n,l1)<<endl;
						// cout<<lr-l1<<endl;
						// cout<<yi<<endl;
						yi_be4_exp = k(0)*2*(Ri+R0_xy1(m,n));
						yi_cx.real(cos(yi_be4_exp));
						yi_cx.imag(sin(yi_be4_exp));
						yi = yi*yi_cx;
						bp_image(i,j,l) = bp_image(i,j,l) + yi;
						// cout<<yi<<endl;
						// cout<<bp_image(0)<<bp_image(400)<<endl;
					}


				}
			}
		}
	}
	cout<<iyn<<" "<<ixn<<endl;
	tend = time(0);
	cout << "range took "<< difftime(tend, tstart) <<" second(s)."<< endl;

	double dynamic_range = 30;
	cube bp_image_abs = abs(bp_image);
	cube image_r_x = bp_image_abs/bp_image_abs.max();
	image_r_x = 20*log10(image_r_x);
	double max_image = image_r_x.max();

	tend = time(0);
	cout << "log operation took "<< difftime(tend, tstart) <<" second(s)."<< endl;

	for(uword i=0;i<image_r_x.n_rows;i++){
		for(uword j=0;j<image_r_x.n_cols;j++){
			for(uword k=0;k<image_r_x.n_slices;k++){
				if(image_r_x(i,j,k) < max_image - dynamic_range){
					image_r_x(i,j,k) = max_image - dynamic_range;
				}
			}
		}
	}
	cout<<size(image_r_x)<<endl;
	tend = time(0);
	cout << "Set background took "<< difftime(tend, tstart) <<" second(s)."<< endl;

	uword index = 141;
	mat resulting_image = image_r_x.slice(index);
	resulting_image.save("resulting_image.txt",arma_ascii);

	// time end
	tend = time(0);
	cout << "Data store took "<< difftime(tend, tstart) <<" second(s)."<< endl;
	return 0;
}
