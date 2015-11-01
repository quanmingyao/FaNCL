clear; clc;

M = 500;
N = M;
K = 5;

U = randn(M, K);
V = randn(N, K);
O = U*V';
G = randn(M, N);
D = O + 0.1*G;
ratio = 2*M*K*log(M)/(M*N);
traD = D.*(rand(M, N) < ratio);
traD = sparse(traD);
tstD = (rand(M, N) < ratio);
tstD = O.*tstD;
tstD = sparse(tstD);
[trow, tcol, tval] = find(tstD);
para.test.row = trow;
para.test.col = tcol;
para.test.data = tval;
para.test.m = size(tstD, 1);
para.test.n = size(tstD, 2);

clear trow tcol tval tstD;

para.tol = 1e-4;
para.maxIter = 1000;
para.tau = 1.01;
para.decay = 0.9;

%% nuclear norm
lambda = 1.2;

% APG
method = 1;
t = tic;
[U, S, V, out{method}] = APGMatComp(full(traD), lambda, para );
Time(method) = toc(t);
X = U*S*V';

NMSE(method, 1) = norm((X - O), 'fro')/norm(X, 'fro');

% Soft-Impute
method = 2;
para.maxR = 80;
t = tic;
para.speedup = 1;
para.exact = 1;
[ U, S, V, out{method} ] = SoftImpute( traD, lambda, para);
Time(method) = toc(t);
X = U*S*V';

NMSE(method) = norm((X - O), 'fro')/norm(O, 'fro');

% Active
method = 3;
t = tic;
[ U, S, V, out{method} ] = mc_alt( traD, lambda, para );
Time(method) = toc(t);
X = U*S*V';

NMSE(method) = norm((X - O), 'fro')/norm(X, 'fro');

%% Non-Convex
para.regType = 1; % CAP 1, LSP 2, TNN 3
para.maxR = 5;

switch(para.regType)
    case 1
        lambda = 9.2;
        theta = 18.4;
    case 2
        lambda = 7.1;
        theta = sqrt(lambda);
    case 3
        lambda = 16.8;
        theta = 5;
end

% GIST
method = 4;
t = tic;
[U, S, V, out{method}] = GISTMatComp(full(traD), lambda, theta, para);
Time(method) = toc(t);
X = U*S*V';
NMSE(method) = norm((X - O), 'fro')/norm(O, 'fro');
clear X U S V;

% IRNN
method = 5;
t = tic;
[U, S, V, out{method}] = IRNNMatComp(full(traD), lambda, theta, para);
Time(method) = toc(t);
X = U*S*V';
NMSE(method) = norm((X - O), 'fro')/norm(O, 'fro');
clear X U S V;

% FaNCL
method = 6;
t = tic;
para.speedup = 1;
[U, S, V, out{method} ] = FastMatComp( traD, lambda, theta, para );
Time(method) = toc(t);
X = U*S*V';
NMSE(method) = norm((X - O), 'fro')/norm(O, 'fro');
clear X U S V;

clear D O U V t G traD tstD valD method para lambda i estRank K theta;

close all;
figure;

for i = 1:6
    semilogx(out{i}.Time, out{i}.RMSE);
    hold on;
end

legend('APG', 'Soft-Impute', 'Active', 'GIST', 'IRNN', 'FaNCL');

xlabel('time (seconds)');
ylabel('testing RMSE');

title('capped l1');