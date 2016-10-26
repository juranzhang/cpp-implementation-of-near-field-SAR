#define _USE_MATH_DEFINES
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;

#include <armadillo>
using namespace arma;

#include <iostream>
#include <math.h> 
using namespace std;

cx_vec vec2cx_vec(vec a, double imaginary_part) {
	cx_vec res(a.n_elem);
	uword i=0;
	while(i<a.n_elem){
		res(i) = cx_double(a(i),imaginary_part);
		i++;
	}
	return res;
}

cx_mat simulated_2Ddechirped_data (cx_vec pos, vec freq, cx_vec tar, cx_vec tar_amp) {
	int speed_light = 299792458;
	uword tar_dim = tar.n_elem;
	uword freq_dim = freq.n_elem;
	uword pos_dim = pos.n_elem;
	cout<<pos_dim<<endl;
	//reshape and repmat
	cx_cube tar_cube(freq_dim, pos_dim, tar_dim);
	cx_cube pos_cube(freq_dim, pos_dim, tar_dim);
	cube freq_cube(freq_dim, pos_dim, tar_dim);
	cx_cube tar_amp_cube(freq_dim, pos_dim, tar_dim);
	
	cout<<freq(0)<<" "<<tar(0)<<" "<<pos(0)<<endl;
	for(uword i = 0; i<freq_dim;++i){
		for(uword j = 0;j<pos_dim;++j){
			for(uword k=0;k<tar_dim;++k){
				tar_cube(i,j,k) = tar(k);
				pos_cube(i,j,k) = pos(j);
				freq_cube(i,j,k) = freq(i);
				tar_amp_cube(i,j,k) = tar_amp(k);
			}
		}
	}

	// for(uword i=0;i<freq_dim;i++){
	// 	for(uword j=0;j<pos_dim;j++){
	// 		for(uword k=0;k<tar_dim;k++){
	// 			//cout<<rm(i,j)<<" ";
	// 			cout<<pos_cube(i,j,k)<<" ";
	// 		}
	// 	}
	// }

	cout<<"hahah6haha"<<endl;
	cube D = abs(tar_cube - pos_cube);
	cout<<D(0,0,0)<<endl;
	// wavenumber
	cube wavenumber = 2 * M_PI/ speed_light * freq_cube ;
	cout<<wavenumber(0,0,0)<<endl;

	// for(uword i=0;i<freq_dim;i++){
	// 	for(uword j=0;j<pos_dim;j++){
	// 		for(uword k=0;k<tar_dim;k++){
	// 			//cout<<rm(i,j)<<" ";
	// 			cout<<wavenumber(i,j,k)<<" ";
	// 		}
	// 	}
	// }

	// SISO response
	cube be4exp_realpart = -2* (wavenumber % D);
	cout<<be4exp_realpart(0,0,0)<<endl;
	// eular formula
	cube afterexp_realpart = cos(be4exp_realpart);
	cube afterexp_imgpart = sin(be4exp_realpart);

	cx_cube afterexp(afterexp_realpart,afterexp_imgpart);
	cout<<afterexp(0,0,0)<<endl;
	cx_cube resp = tar_amp_cube % afterexp;
	cout<<resp(0,0,0)<<endl;
	// sum
	cx_mat siso_resp(freq_dim, pos_dim);
	cx_double z_sum;
	for(uword i = 0; i<freq_dim;++i){
		for(uword j = 0;j<pos_dim;++j){
			z_sum.real(0);
			z_sum.imag(0);
			for(uword k=0;k<tar_dim;++k){
				z_sum = z_sum+resp(i,j,k);
			}
			siso_resp(i,j) = z_sum;
		}
	}
	cout<<siso_resp(0,0)<<endl;
	cout<<"yusda"<<endl;
	return siso_resp;
}

int main() {
	int dynamic_range = 50;         
	double c = 299792458;
	double f_start = 28000000000;
	double f_stop = 33000000000;
	double deltf = 100000000;
	double B = f_stop-f_start;

	// number of frequency
	double Nf = floor(B/deltf)+1;
	f_stop = f_start + (Nf-1)*deltf;
	B = f_stop-f_start;

	// generate a vector of Nf freq
	vec freq = linspace<vec>(f_start,f_stop,Nf);
	freq.t();

	vec k = 2 * M_PI * freq/c; 
	double deltkr=k(1)-k(0);
	double deltkz = 2 * deltkr; 
	double f_mid = f_start+B/2;
	double lam_max = c/f_start;
	double lam_mid = c/f_mid;
	double lam_min = c/f_stop;     
	double resolution_y = c/2/B;
	printf("resolution_y: %e \n", resolution_y);
	double R0 = 5 ; 
	double array_length = 2;
	double dx = 0.01;
	double Nx = round(array_length/dx);  
	if (Nx/2-round(Nx/2)!=0) {
		Nx = Nx+1;
	}                   
	printf("Nx: %e \n", Nx);

	vec x_array = linspace<vec> (-Nx * dx/2, (Nx/2 - 1)*dx, Nx);
	cout<<"x_array length: "<<x_array.n_elem<<endl;
	cout<<x_array(Nx-1)<<" "<<x_array(0)<<endl;
	array_length = x_array(Nx-1) - x_array(0);
	printf("array_length: %e \n", array_length);

	double Theta_antenna = 2*atan((x_array(Nx-1)-x_array(0))/2/R0);
	double resolution_x_theory = lam_max/4/sin(Theta_antenna/2);
	double antenna_space = lam_min/4/sin(Theta_antenna/2);
	printf("resolution_x_theory: %e \n", resolution_x_theory);
	printf("antenna_space: %e \n", antenna_space);

	// initialize target.
	cx_vec tar(5);
	tar(0) = cx_double(0,0);
	tar(1) = cx_double(0,0.3);
	tar(2) = cx_double(0.3,0.3);
	tar(3) = cx_double(-0.3,0.3);
	tar(4) = cx_double(0.1,-0.3);
	cout<<"ha3haha"<<endl;
	// fill cube of the same size with ones. size(1,1,tar_dim)
	vec tar_amp_vec(size(tar),fill::ones);
	cx_vec tar_amp = vec2cx_vec(tar_amp_vec,0);

	// initialize pos
	cx_vec X_array = vec2cx_vec(x_array,R0);
	
	cx_mat S_echo = simulated_2Ddechirped_data(X_array,freq,tar,tar_amp);
	cout<<"S_echo"<<S_echo(0,0)<<endl;
	double FFT_NUM_Multi_x = 2;
	double FFT_NUM_Multi_z = 2;

	double NLx = 8;
	double NLz = 8;

	double FNx = Nx; 
	double FNf = Nf;

	vec kx = linspace(-M_PI/dx, M_PI/dx - 2*M_PI/dx/FNx, FNx);
	double deltkx = kx(1)-kx(0);

	//mat Kr(k,FNx,fill::zeros);
	cx_mat be4abs = ifft(S_echo);
	cout<<"be4abs"<<be4abs(0,0)<<endl;
	cout<<"be4abs"<<be4abs(50,199)<<endl;
	cout<<"be4abs"<<be4abs(0,199)<<endl;
	cout<<"be4abs"<<be4abs(50,0)<<endl;
	mat rm = abs(ifft2(S_echo));
	for(uword i=0;i<rm.n_rows;i++){
		for(uword j=0;j<rm.n_cols;j++){
			if(rm(i,j)>1){
				cout<<rm(i,j)<<" "<<i<<" "<<j;
			//cout<<S_echo(i,j)<<" ";
			}
			
		}
		cout<<"\n";
	}

	cout<<rm.n_rows<<" "<<rm.n_cols<<endl;
	cv::Mat img0( rm.n_rows, rm.n_cols, CV_8UC1, rm.memptr());
	cv::Mat image;
	applyColorMap(img0, image, COLORMAP_HOT);
	namedWindow( "Display window", WINDOW_NORMAL);
	imshow("Display window", image);
	waitKey(0);

	return 0;
	
}