# cpp implementation of near-field SAR
A cpp implementation of near-field SAR. The original code is written in MATLAB.

## How to setup

Using armadillo to do matrix/cube transformation and opencv to plot. HDF5 is used to store data.

Armadillo can be downloaded here http://arma.sourceforge.net/download.html. Although personally I strongly recommend use homebrew to install everything. Checkout this: http://braumeister.org/repos/Homebrew/homebrew-science/formula/armadillo. The page also includes other prerequisite of armadillo. 

On both Mac and Linux:

To start with, install OpenBlas and LAPACK before installing Armadillo. 
`brew install openblas`. Other libraries can be installed in the similiar way.

Deprecated: OpenCV can be downloaded from https://github.com/opencv/opencv. This is the most recent update. An installation guide of OpenCV on Mac can be found here: http://stackoverflow.com/questions/19671827/opencv-installation-on-mac-os-x. OpenCV is no longer used. 


## The files

### test.cpp
Test.cpp creates a simulated target that consists of 5 spots and processes the target into received signal (S_echo in the .cpp file). The received signal is then computed to reconstruct the target. This .cpp file aims to test and give a simple demo of the reconstruction algorithm. `make test` and `./test` to build and run the program.

### test2d.cpp 
This is the main program that reconstruct real 2D signal. The signal is stored in 'real2d.txt' and 'imag2d.txt'.

### plotdata.py 
A plotting tool in python as an alternative to opencv. It loads data from 'resulting_image.txt'.

### WK_3D_imaging_experiment.cpp
The main program using WK algorithm to reconstruct 3D signal. The received signal is stored separately in 'secho_real.h5' and 'secho_imag.h5". These files are not here, you need to use your own data.

On Linux, be aware of the location of hdf5.h. If ld can't find libhdf5, try to create a symbolic link. `sudo ln -s /usr/lib/x86_64-linux-gnu/hdf5/serial/libhdf5.so /usr/lib/libhdf5.so`. The first path can be found by `locate libhdf5.so`.

### fftw_WK_3D_imaging_experiment.cpp
This file utilizes fftw to speed up fft in WK_3D_imaging_experiment.cpp. This will eventually be merged into WK_3D_imaging_experiment.cpp

### cufftw_WK_3D_imaging_experiment.cpp
This program uses cufftw with a GPU to speed up fft. It performs a litte better than fftw.

### BP_3D_imaging.cpp
The main program using BP algorithm to reconstruct 3D signal. This method is really slow that takes 48 hours to run in Matlab. I am working on cuda version of the algorithm. 

### stolt_mex.cpp
This is the MEX file that is called in Matlab to improve speed of the stolt_interrupt function. So, the file has nothing to do with computing on Mac or Linux.


## How to build and run these files

1. `make test.cpp` and run `./test` to save the image data into ‘resulting_image.txt’. 

2. Delete the header of ‘resulting_image.txt’ that was generated by the armadillo save function (you can refer to the armadillo docs for detail).

3. ‘python plotdata.py’ to see the resulting image.


## How to create your own .h5 files

1. (If you have custom received signal stored in a .mat file) use Matlab to load the data and write into a hdf5 file. We are preserving data into hdf5 format because the data is pretty large with over 10 million complex doubles. This amount of data will take armadillo 20 seconds to read if stored in ascii text format.
  1. (In matlab) `h5create('secho_real.h5','/DS1',[200,320,200])`
  2. `dreal = real(secho);`
  3. `h5write('secho_real.h5','/DS1',dreal)`
  4. Repeat the above steps for secho_imag. Feel free to check usage of these functions on Matlab documentation page.

2. In order to enable hdf5 file load/save functionalities: On Mac or Linux, go to your armadillo folder, inside /include/armadillo_bits/config.hpp, uncomment the line `#define ARMA_USE_HDF5` as specified in http://arma.sourceforge.net/docs.html#config_hpp.

3. You'll also need to install hdf5 library by `brew install homebrew/science/hdf5`.

4. Follow the instructions above to build and run.


## (Deprecated) How to compile if you are using opencv:

`g++ azimuth.cpp 'pkg-config --libs --cflags opencv' -o example1 -O2 -larmadillo`

You'll need to install pkg-config before compiling (On Mac).

##-------------Matlab optimization notes below---------------

## Write Matlab MEX code to speed up stolt interrupt:

In WK_3D_imaging_experiment.m, stolt_interrupt consumes most of the time (27s out of 30s in total). While if you run WK_3D_imaging_experiment.cpp by any chance, you will find out that stolt_interrupt takes only 2 seconds out of 20s in total (most of the time is consumed by fft and ifft operations). In order to improve the run time of either Matlab code or c++ code, it is either stolt_interrupt or fft operations that need to be optimized. One of the question is, can we optimize stolt_interrupt in Matlab? Thus, can we implement this function in c++ and call it in Matlab?

MEX files are c++/c files that can be called in Matlab as functions. It needs to be compiled before being called along with other matlab instructions. Below is a quick tutorial for you to start with:

Ref: https://classes.soe.ucsc.edu/ee264/Fall11/cmex.pdf


#### How to build and use stolt_mex.cpp in Matlab

1. Write the command `mex stolt_mex.cpp` in Matlab (make sure it is in the same working space with WK_3D_imaging_experiment.m and data). This will compile the file using g++.

2. Change the callee function from stolt_interrupt (a matlab file) to stolt_mex (name of the compiled MEX file). Input parameters remain the same. 

3. Run WK_3D_imaging_experiment.m as usual and see the run time change.
