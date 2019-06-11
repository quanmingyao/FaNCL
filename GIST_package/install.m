clear
clc


currentpath = cd;

addpath(genpath([currentpath,'/GIST']));

cd ./GIST

mex proximalRegC.c
mex funRegC.c

cd ..

