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

// turn kx vector into matrix of the same col dim and row dim of Nf
mat vec2mat_y(vec kx, uword Nf){
	mat kx_mat(kx.n_elem,Nf);
	for(uword i=0;i<kx.n_elem;i++){
		for(uword j=0;j<Nf;j++){
			kx_mat(i,j) = kx(i);
		}
	}
	return kx_mat;
}

// turn ky vector into matrix of the same row dim and col dim of Nf
mat vec2mat_x(vec ky, uword Nf){
	mat kx_mat(Nf,ky.n_elem);
	for(uword i=0;i<Nf;i++){
		for(uword j=0;j<ky.n_elem;j++){
			kx_mat(i,j) = ky(j);
		}
	}
	return kx_mat;
}

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

	//reshape and repmat
	cx_cube tar_cube(freq_dim, pos_dim, tar_dim);
	cx_cube pos_cube(freq_dim, pos_dim, tar_dim);
	cube freq_cube(freq_dim, pos_dim, tar_dim);
	cx_cube tar_amp_cube(freq_dim, pos_dim, tar_dim);
	
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

	cube D = abs(tar_cube - pos_cube);
	
	// wavenumber
	cube wavenumber = 2 * M_PI/ speed_light * freq_cube ;
	
	// SISO response
	cube be4exp_realpart = -2* (wavenumber % D);

	// euler's formula
	cube afterexp_realpart = cos(be4exp_realpart);
	cube afterexp_imgpart = sin(be4exp_realpart);
	cx_cube afterexp(afterexp_realpart,afterexp_imgpart);
	cx_cube resp = tar_amp_cube % afterexp;
	
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
	cout<<"Nf:"<<Nf<<endl;
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
	
	// fill cube of the same size with ones. size(1,1,tar_dim)
	vec tar_amp_vec(size(tar),fill::ones);
	cx_vec tar_amp = vec2cx_vec(tar_amp_vec,0);

	// initialize pos
	cx_vec X_array = vec2cx_vec(x_array,R0);
	
	cx_mat S_echo = simulated_2Ddechirped_data(X_array,freq,tar,tar_amp);

	cx_mat be4abs = ifft(S_echo);
	mat rm = abs(ifft(S_echo));
	// for(uword i=0;i<rm.n_rows;i++){
	// 	cout<<be4abs(i,0)<<endl;
	// }

	// cv::Mat img0( rm.n_rows, rm.n_cols, CV_8UC1, rm.memptr());
	// cv::Mat image;
	// applyColorMap(img0, image, COLORMAP_HOT);
	// namedWindow( "Display window", WINDOW_NORMAL);
	// imshow("Display window", image);
	// waitKey(0);

/*

rm = ifft(S_echo);                                      %¾àÀëáã¶¯Í¼Ïñ
figure(204)
imagesc(x_array,[],abs(rm))
title('¾àÀëáã¶¯Ê¾ÒâÍ¼')
figure(205)
imagesc(kx,[],abs(fftshift(fft(rm,[],2),2)))
title('¾àÀë¶àÆÕÀÕÓò')

S_echo = fftshift(S_echo,2);                             %½«S_echo½øÐÐ¸µÀïÒ¶Æ½ÒÆ£¬£¨ÑØ×ÅµÚ¶þÎ¬£¿£©
S_kx = fft(S_echo,FNx,2);                                %½«S_echoµÄµÚ¶þÎ¬£¨ÌìÏßÔªÊýÄ¿£©µÄ·½Ïò½øÐÐ¸µÀïÒ¶±ä»»£¬Æä·µ»Ø½á¹ûÊÇS_kx
S_kx = fftshift(S_kx,2);                                 %¶ÔS_kxÑØµÚ¶þÎ¬½øÐÐÒ»´Î¸µÀïÒ¶Æ½ÒÆ£¬

figure(1);                                               %Éú³ÉÍ¼Ïñ1
imagesc(kx,k,abs(S_kx));                                 %½«S_kxµÄ¾ø¶ÔÖµÏÔÊ¾ÎªÍ¼Ïñ£¬ºáÖáÎªkx,×ÝÖáÎªk,    imagesc(x,y,c)½«ÊäÈë±äÁ¿cÏÔÊ¾ÎªÍ¼Ïñ£¬ÓÃx,y±äÁ¿È·¶¨x,yÖáµÄ±ß½ç
axis xy;                                                 %Éè¶¨MatlabÎªÄ¬ÈÏÏÔÊ¾Ä£Ê½,xÖáË®Æ½ÇÒ×óÐ¡ÓÒ´ó£¬yÖá´¹Ö±ÇÒµ×Ð¡¶¥´ó¡£ÈôÃ»ÓÐÕâ¸öÉè¶¨£¬ÒòÎªkÖµÊÇ¶¥²¿Ð¡µ×²¿´ó£¬Í¼Ïñ½«ÉÏÏÂµßµ¹
title('spatial frequency');                              %Éè¶¨±êÌâ   ¡°¿Õ¼äÆµÂÊ¡±
xlabel('kx');ylabel('k'); 

*/
	double FFT_NUM_Multi_x = 2;
	double FFT_NUM_Multi_z = 2;

	double NLx = 8;
	double NLz = 8;

	double FNx = Nx; 
	double FNf = Nf;

	vec kx = linspace(-M_PI/dx, M_PI/dx - 2*M_PI/dx/FNx, FNx);
	double deltkx = kx(1)-kx(0);

	// match filter
	// KX is copied from kx alone y-axis for Nf times
	// KX:200*51
	mat KX = vec2mat_y(kx,Nf);	
	cx_mat matched_filter(FNx,Nf);

	// KZ is copied from k alone x-axis for FNx times. Note k is a vec on the y-axis.
	// KZ:200*51
	mat Kr = vec2mat_x(k,FNx);
	mat KZ = 4 * Kr; 
	KZ = sqrt(pow(KZ,2)-pow(KX,2));

	/*
	Matched_filter = exp(1i*KZ*R0).* (abs(KX) <= 2*(k*ones(1,FNx)));
	*/
	for(uword i=0;i<FNx;i++){
		for(uword j=0;j<Nf;j++){
			if(abs(KX(i,j))<=2*KZ(i,j)) {
				matched_filter(i,j).real(cos(R0*KZ(i,j)));
				matched_filter(i,j).imag(sin(R0*KZ(i,j)));
			}
			else{
				matched_filter(i,j).real(0.0);
				matched_filter(i,j).imag(0.0);
			}
		}
	}

	Kr = 2 * Kr;
	mat Kz_sq = pow(Kr,2) - pow(KX, 2);

	// S_kx is generated by fftshift
	mat S_kx(FNx,Nf);
	for(uword i=0;i<FNx;i++){
		for(uword j=0;j<Nf;j++){
			if(Kz_sq(i,j)<0){
				Kz_sq(i,j)=0;
				S_kx(i,j)=0;
			}
		}
	}

	mat fmf = R0 * sqrt(Kz_sq);
	mat matched_real = S_kx % cos(fmf);
	mat matched_imag = S_kx % sin(fmf);
	cx_mat S_matched(matched_real,matched_imag);

	// matlab angle function
	mat S_matched_angle = atan(matched_imag / matched_real);
	
	/*
	figure(2)                                                %»æÖÆµÚ¶þ¸öÍ¼Ïñ  
subplot(331)
plot(kx,abs(S_kx(1,:)))                                 %ÓÃS_kxºÍS_matchedµÄÊý¾ÝÈ¡abs¿´·ùÆµ½á¹ûÒ»Ñù
xlabel('kx');ylabel('real part');
title('×îµÍÆµÂÊ·ùÆµÍ¼')
subplot(332)
plot(kx,angle(S_kx(1,:)))
xlabel('kx');ylabel('mag')
title('×îµÍÆµÂÊÆ¥ÅäÂË²¨Ç°ÏàÆµÍ¼')                 %¿ÉÒÔ¿´³ö¸ßÆµÆ¥ÅäÂË²¨Ö®ºóµÄ¿Õ¼äÆµÂÊ´ø¿íÒª´ó£¬ËùÒÔ¸ßÆµµÄ¿Õ¼ä·Ö±æÂÊ¸ü¸ß
subplot(333)
plot(kx,angle(S_matched(1,:)))
xlabel('kx');ylabel('mag')
title('×îµÍÆµÂÊÆ¥ÅäÂË²¨ºóÏàÆµÍ¼')
subplot(334)                                             %»æÖÆµÚ¶þ¸öÍ¼ÏñÖÐµÄµÚÒ»²¿·Ö        subplot£¨m n p£©±íÊ¾»æÖÆÒ»¸öm*nµÄÍ¼£¬pÊÇÖ¸ÄãÏÖÔÚÒª°ÑÇúÏß»­µ½figureÖÐÄÄ¸öÍ¼ÉÏ£¬
plot(kx,abs(S_kx(round(Nf/2),:)));                       %ÒÔkxÎªxÖá£¬S_kxÖÐ¼äµÄÒ»ÐÐÎªyÖá½øÐÐ»æÖÆ  
xlabel('kx');ylabel('real part')
title('ÖÐÐÄÆµÂÊ·ùÆµÍ¼')                        %±êÌâ£ºÆ¥ÅäÂË²¨Ö®Ç°µÄ¿Õ¼äÆµÂÊµÄÊµ²¿  
subplot(335)                                             % %»æÖÆµÚ¶þ¸öÍ¼ÏñÖÐµÄµÚ¶þ²¿·Ö  
plot(kx,angle(S_kx(round(Nf/2),:)));      %ÒÔkxÎªxÖá£¬S_matchedÖÐ¼äµÄÒ»ÐÐÎªyÖá½øÐÐ»æÖÆ
xlabel('kx');ylabel('mag');
title('ÖÐÐÄÆµÂÊÆ¥ÅäÂË²¨Ç°ÏàÆµÍ¼')
subplot(336)                                             % %»æÖÆµÚ¶þ¸öÍ¼ÏñÖÐµÄµÚ¶þ²¿·Ö  
plot(kx,angle(S_matched(round(Nf/2),:)));      %ÒÔkxÎªxÖá£¬S_matchedÖÐ¼äµÄÒ»ÐÐÎªyÖá½øÐÐ»æÖÆ
xlabel('kx');ylabel('mag');
title('ÖÐÐÄÆµÂÊÆ¥ÅäÂË²¨ºóÏàÆµÍ¼')                %±êÌâ£ºÆ¥ÅäÂË²¨Ö®ºóµÄ¿Õ¼äÆµÂÊµÄ·ùÆµÍ¼
subplot(337);                                             
plot(kx,abs(S_kx(Nf,:)))
title('×î¸ßÆµÂÊ·ùÆµÍ¼')
subplot(338);                                              
plot(kx,angle(S_kx(Nf,:)))
xlabel('kx');ylabel('mag')
title('×î¸ßÆµÂÊÆ¥ÅäÂË²¨Ç°ÏàÆµÍ¼')
subplot(339);                                              
plot(kx,angle(S_matched(Nf,:)))
xlabel('kx');ylabel('mag')
title('×î¸ßÆµÂÊÆ¥ÅäÂË²¨ºóÏàÆµÍ¼')

figure(3);                                                %»æÖÆµÚÈý¸öÍ¼Ïñ  
imagesc(kx,(k),angle(S_matched))                          %½«S_matchedµÄ½Ç¶È»æÖÆ³ÉÍ¼Ïñ£¬ÒÔkxÎªxÖá£¬kÎªyÖá  
colorbar();axis xy;xlabel('kx');ylabel('k');              %
title('Phases after matched filtering');                  %±êÌâ£ºÆ¥ÅäÂË²¨ºóµÄÏàÎ»

figure(4);                                
	*/

	uword line_num = S_matched.n_rows;
	uword col_num = S_matched.n_cols;

	cout<<"n_rows:"<<line_num<<" n_cols:"<<col_num<<endl;
	double mid_line= round(line_num/2); 

	/*
lot(kx,mod(unwrap(angle(S_matched(mid_line,:))),2*pi));  %    £¿£¿£¿£¿
xlabel('kx');ylabel('rad');
title('Phases after matched filtering');
	*/

	// stolt interpolation
	// line 212
	return 0;
	
}