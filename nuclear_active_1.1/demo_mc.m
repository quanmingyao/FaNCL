clear;

maxNumCompThreads(1)
load movielens100kbrenorm
lambda = 14;
k = 100;

t = tic;
[U1, V1, obj, time]=mc_alt(Y',lambda, 1e-3, 1000, 150);
toc(t)


