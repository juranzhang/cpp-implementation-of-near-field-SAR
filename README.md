# cpp implementation of near-field SAR
A cpp implementation of near-field SAR. The original code is written in MATLAB.

## How to setup

Using armadillo to do matrix/cube transformation and opencv to plot.

Armadillo can be downloaded here http://arma.sourceforge.net/download.html.

OpenCV can be downloaded from https://github.com/opencv/opencv. This is the most recent update.

An installation guide can be found here: http://stackoverflow.com/questions/19671827/opencv-installation-on-mac-os-x.


## The files

### azimuth.cpp
Azimuth.cpp creates a simulated target that consists of 5 spots and processes the target into received signal (S_echo in the .cpp file). The received signal is then computed to reconstruct the target. This .cpp file aims to test and give a simple demo of the construction algorithm.

### generalized_azimuth.cpp 
This is the main program that reconstruct real 2D signal. The signal is stored in 'real2d.txt' and 'imag2d.txt'.

### plotdata.py 
A plotting tool in python as an alternative to opencv. It loads data from 'resulting_image.txt'.

### WK_3D_imaging_experiment.cpp
The main program using WK algorithm to reconstruct 3D signal. The received signal is stored separately in 'secho_real.txt' and 'secho_imag.txt".

### BP_3D_imaging.cpp
The main program using BP algorithm to reconstruct 3D signal. This method is really slow compared to WK that takes 48 hours to run in Matlab.

### stolt_mex.cpp
This is the MEX file that is called in Matlab to improve speed of the stolt_interrupt function.


## How to run Azimuth.cpp

1. `g++ azimuth.cpp -o azimuth -O2 -larmadillo` and run `./azimuth` to save the image data into ‘resulting_image.txt’. 

2. Delete the header of ‘resulting_image.txt’ that was generated by the armadillo save function (you can refer to the armadillo docs for detail).

3. ‘python plotdata.py’ to see the resulting image which has 5 bright dots.


## How to run Generalized_azimuth.cpp

1. (If you have custom received signal stored in a .mat file) use Matlab to get the dimensions of the input data. Then add ‘ARMA_MAT_TXT_FN008 num_row num_col’, as shown in ‘real2d.txt’ and ‘imag2d.txt’. The .cpp program will read ascii from the text files.

2. `g++ generalized_azimuth.cpp -o generalized_azimuth -O2 -larmadillo` and run `./generalized_azimuth` to save the image data into ‘resulting_image.txt’.

3. Delete the header of ‘resulting_image.txt’ that was generated by the armadillo save function (you can refer to the armadillo docs for detail).

4. `python plotdata.py` to see the resulting image.


## How to run WK_3D_imaging_experiment.cpp

1. (If you have custom received signal stored in a .mat file) use Matlab to load the data and write into a hdf5 file. We are preserving data into hdf5 format because the data is pretty large with over 10 million complex doubles. This amount of data will take armadillo 20 seconds to read if stored in ascii text format.
  1. (In matlab) `h5create('secho_real.h5','/DS1',[200,320,200])`
  2. `dreal = real(secho);`
  3. `h5write('secho_real.h5','/DS1',dreal)`
  4. Repeat the above steps for secho_imag. Feel free to check usage of these functions on Matlab documentation page.

2. In order to enable hdf5 file load/save functionalities: Go to your armadillo folder, inside /include/armadillo_bits/config.hpp, uncomment the line `#define ARMA_USE_HDF5` as specified in http://arma.sourceforge.net/docs.html#config_hpp.

3. You'll also need to install hdf5 library by `brew install homebrew/science/hdf5`

4. `g++ WK_3D_imaging_experiment.cpp -o WK_3D_imaging_experiment -O2 -DARMA_DONT_USE_WRAPPER -DARMA_USE_BLAS -DARMA_USE_LAPACK -DARMA_USE_HDF5 -lblas -llapack -larmadillo -lhdf5` and run `./WK_3D_imaging_experiment` to save reconstructed imaging data into ‘resulting_image.txt’.

5. Delete the header of ‘resulting_image.txt’ that was generated by the armadillo save function (you can refer to the armadillo docs for detail: http://arma.sourceforge.net/docs.html#save_load_mat).

6. `python plotdata.py` to see the resulting image.


## Write Matlab MEX code to speed up stolt interrupt:

In WK_3D_imaging_experiment.m, stolt_interrupt consumes most of the time (27s out of 30s in total). While if you run WK_3D_imaging_experiment.cpp by any chance, you will find out that stolt_interrupt takes only 2 seconds out of 20s in total (most of the time is consumed by fft and ifft operations). In order to improve the run time of either Matlab code or c++ code, it is either stolt_interrupt or fft operations that need to be optimized. One of the question is, can we optimize stolt_interrupt in Matlab? Thus, can we implement this function in c++ and call it in Matlab?

MEX files are c++/c files that can be called in Matlab as functions. It needs to be compiled before being called along with other matlab instructions. Below is a quick tutorial for you to start with:

Ref: https://classes.soe.ucsc.edu/ee264/Fall11/cmex.pdf

#### How to run stolt_mex.cpp

1. Write the command `mex stolt_mex.cpp` in Matlab (make sure it is in the same working space with WK_3D_imaging_experiment.m and data). This will compile the file using g++.

2. Change the callee function from stolt_interrupt (a matlab file) to stolt_mex (name of the compiled MEX file). Input parameters remain the same. 

3. Run WK_3D_imaging_experiment.m as usual and see the run time change.


## How to run BP_3D_imaging.cpp

1. Follow step 1.2.3 in WK_3D_imaging_experiment.cpp if you haven't run that program first.

2. `g++ BP_3D_imaging.cpp -o BP_3D_imaging -O2 -DARMA_DONT_USE_WRAPPER -DARMA_USE_BLAS -DARMA_USE_LAPACK -DARMA_USE_HDF5 -lblas -llapack -larmadillo -lhdf5` and run `./BP_3D_imaging` to save reconstructed imaging data into 'resulting_image.txt'.

3. Same as the previous programs.


## How to compile if you are using opencv:

`g++ azimuth.cpp 'pkg-config --libs --cflags opencv' -o example1 -O2 -larmadillo`

You'll need to install pkg-config before compiling

