% Before you run this example, make sure that you have run install.m
% to add the path and mex C files

clear;
clc;
close all;

% generate data 
n = 1000;
d = 500;
X = randn(n,d);
wt = randn(d,1);
y = X*wt + 0.02*randn(n,1);

% input parameters
lambda = 1e-3*abs(randn);
theta = 1e-2*lambda*abs(randn);

% optional parameter settings

regtype = 1; % nonconvex regularization type (default: 1 [capped L1]) 

w0 = randn(d,1); % starting point (default: zero vector)

stopcriterion = 0; % stopping criterion (default: 1)

maxiter = 1000; % number of maximum iteration (default: 1000)

tol = 1e-5; % stopping tolerance (default: 1e-5)

M = 5; % nonmonotone steps (default: 5)

t = 1; % initialization of t (default: 1)

tmin = 1e-20; % tmin parameter (default: 1e-20)

tmax = 1e20; % tmax parameter (default: 1e20)

sigma = 1e-5; % parameter in the line search (default: 1e-5)

eta = 2; % eta factor (default: 2)

stopnum = 3; % number of satisfying stopping criterion (default: 3)

maxinneriter = 20; % number of maximum inner iteration (line search) (default: 20)

% call the function
[w,fun,time,iter] = gistLeast(X,y,lambda,theta,...
                              'maxiteration',maxiter,...
                              'regtype',regtype,...
                              'stopcriterion', stopcriterion,...
                              'tolerance',tol,...
                              'startingpoint',w0,...
                              'nonmonotone',M,...
                              'tinitialization',t,...
                              'tmin',tmin,...
                              'tmax',tmax,...
                              'sigma',sigma,...
                              'eta',eta,...
                              'stopnum',stopnum,...
                              'maxinneriter',maxinneriter);
                          
% plot
figure
semilogy(time(1:iter+1),fun(1:iter+1),'r-','LineWidth', 2)
xlabel('CPU time (seconds)')
ylabel('Objective function value (log scaled)')
legend('GIST-Least')



