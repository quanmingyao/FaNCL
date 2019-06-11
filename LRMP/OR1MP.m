function [U, Theta, V, out ] = OR1MP(data, r, para )
%ORTHOGONAL-RANK-1-MATRIX-PURSUIT Infinit dimension matching pursuit for 
%low rank matrix
%
%For the problem           min    L(X) = ||X - Y||^2
%                          s.t.   rank(X) <= r
%                          X = Theta * M = sum_i theta_i * M_i = U Theta V'
%   Detail variable
%   Input:
%         m ---- row number
%         n ---- column number
%         r ---- number of basis
%         Known ---- index of the known spot in the matrix
%         data ---- the content of the known spot in the matrix 
%         opts ---- parameters for the algorithm
%   Output:
%         U ---- output matrix: U
%         Theta ---- output matrix: Theta
%         V ---- output matrix: V
%         numiter ---- number of iterations
%
%   Copyright Zheng Wang @ Arizona State University
%   $Date: 2013/01/29$

%addpath('PROPACK/');
%addpath('largescale_ops/');
%addpath('SLEP_package_4.1/');

[m, n] = size(data);

% initialization, 
[indm, indn, val] = find(data);
% [indm, indn] = ind2sub([m, n], Known);
% data( data == 0 )= eps;
res = sparse(indm, indn, val, m, n);
% [indm, indn, data] = find(res);
U = [];
V = [];
Msup = [];

i = 0;
W = 0;
oldresnorm = 0;
gresnorm = 1;
yy = [];
nnorm = norm(val, 'fro');
obj = zeros(1, r);
RMSE = zeros(1, r);
pwIter = zeros(1, r);
% main iteration
% In OR1MP, the stop criterion is small gradient of residual
while (i < r) && (gresnorm > 1e-2 )
    % 1. find the top singular pair of the residual and update the gresnorm
    resvec = val - W;
    sparse_update(res, resvec); % sparse update the res using resvec

    [u, ~, v, pwIter(i + 1)] = topsvd(res, 1000); % run our power method for 10 iterations
    %[u, ~, v] = topsvd(res, 1);
    %[u, ~, v] = lansvd(res, 1, 'L'); % fast sparse svd using PROPACK
    %[u, s, v] = svds(res, 1); % use matlab sparse top svd
    
    resnorm = sum(resvec.^2);
    gresnorm = abs(resnorm - oldresnorm)/nnorm;
    oldresnorm = resnorm;

    % 2. update the weight Theta, the pursuit basis is uv', its weight is s.
    Mi = sparse_inp(u', v', indm, indn)';

    % b) use incremental inverse to solve the least sqare problem
    if i~=0
        Minv = inverse_incremental(Minv, Msup'*Mi, Mi'*Mi);
    else
        Minv = 1/(Mi'*Mi);
    end
    yy = [yy; Mi'*val];
    Theta = Minv*yy;
    Msup = [Msup Mi];
    
    U = [U u];
    V = [V v];

    % 3. update the learned matrix W = U' * diag(Theta) * V;
    W = Msup * Theta;
    
    i = i + 1;
    obj(i) = resnorm;
    fprintf('iter: %d, obj %d, power: %d \n', i, resnorm, pwIter(i));
    if(isfield(para, 'test'))
        RMSE(i) = MatCompRMSE(U, V, diag(Theta), ...
            para.test.row, para.test.col, para.test.data);
        fprintf('RMSE %.2d \n', RMSE(i));
    end
end

out.obj = obj;
out.RMSE = RMSE;
out.pwIter = pwIter;

end
% V = diag(Theta)*V;
% fprintf( '\n OR1MP run %d rounds! \n', numiter);

% %* Sparse selection: for sharpe low rank problem, we may need the lasso 
% %fine selection. Use lasso to learn a more sparse weights.
% % Starting point
% opts.init=2;        % starting from a zero point
% % termination criterion
% opts.tFlag=5;       % run .maxIter iterations
% opts.maxIter=100;   % maximum number of iterations
% % normalization
% opts.nFlag=0;       % without normalization
% % regularization
% opts.rFlag=1;       % the input parameter 'rho' is a ratio in (0, 1)
% opts.mFlag=0;       % treating it as compositive function
% opts.lFlag=0;       % Nemirovski's line search
% % get the final sparse Theta by lasso in SLEP toolbox
% [Theta, ~, ~] = LeastR(Msup, data, 0.00001, opts); %  [W, funVal, ValueL] 00001

function Ninv = inverse_incremental(Minv, MMi, d)
% calculate the inverse of the blocked matrix of 
P = MMi' * Minv; % vector
q = 1/(d - P*MMi); % scaler
y = q*P;
Ninv = [Minv+P'*y, -y'; -y, q];

end

