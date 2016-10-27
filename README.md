# cpp implementation of near-field SAR
A cpp implementation of near-field SAR. The original code is written in MATLAB.

Using armadillo to do matrix/cube transformation and opencv to plot.

Armadillo can be downloaded here http://arma.sourceforge.net/download.html.

OpenCV can be downloaded from https://github.com/opencv/opencv. This is the most recent update.

An installation guide can be found here: http://stackoverflow.com/questions/19671827/opencv-installation-on-mac-os-x.

How to compile:

g++ azimuth.cpp \`pkg-config --libs --cflags opencv\` -o example1 -O2 -larmadillo

You'll need to install pkg-config before compiling
