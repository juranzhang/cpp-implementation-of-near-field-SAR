#define _USE_MATH_DEFINES

#include <armadillo>
using namespace arma;

#include <iostream>
#include <math.h> 
#include <ctime>
using namespace std;


int main() {
	// cube S_echo_real;
	// S_echo_real.load("secho_real.txt",arma_ascii);
	// cube S_echo_imag;
	// S_echo_imag.load("secho_imag.txt",arma_ascii);

	// cx_cube S_echo(S_echo_real,S_echo_imag);
	// cout<<S_echo(0,0,0)<<endl;

	cube complex_image;
	complex_image.load("complex_image.txt",arma_ascii);
	cout<<"0,0,0: "<<complex_image(0,0,0)<<endl;
	cout<<"0,0,178: "<<complex_image(0,0,178)<<endl;
	cout<<"0,178,0: "<<complex_image(0,178,0)<<endl;
	cout<<"178,0,0: "<<complex_image(178,0,0)<<endl;
	cout<<"178,179,107: "<<complex_image(178,179,107)<<endl;
	cout<<"178,107,179: "<<complex_image(178,107,179)<<endl;
	cout<<size(complex_image)<<endl;

	//cout<<max(complex_image)<<endl;
	//uword index = max(max(complex_image)).index_max();
	// uword index = complex_image.index_max();
	//cout<<index<<endl;
	mat resulting_image(complex_image.n_cols,complex_image.n_slices);
	for(uword j=0;j<complex_image.n_cols;j++){
		for(uword k=0;k<complex_image.n_slices;k++){
			resulting_image(j,k) = complex_image(178,j,k);
		}
	}
	
	resulting_image.save("resulting_image.txt",arma_ascii);
	return 0;
}
