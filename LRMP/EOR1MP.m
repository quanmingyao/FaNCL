function [U, Theta, V, out ] = EOR1MP(data, r, para )
%MP-MATRIX-COMPLETION Infinite dimension MP for matrix completion 
%
%For the problem           min  sum L(X),
%                          s.t. X = Theta * M = sum_i theta_i * M_i.
%   Detailed explanation goes here
%   Input:
%         m ---- row number
%         n ---- column number
%         r ---- number of basis
%         Known ---- index of the known elements in the matrix
%         data ---- the content of the known elements in the matrix 
%         opts ---- parameters for the algorithm
%   Output:
%         U ---- output matrix: U
%         V ---- output matrix: V'
%         Theta ---- output weights 
%         numiter ---- number of iterations
%
%   Copyright: Zheng Wang @ ASU
%   $Date: 2013/12/14$

% addpath('largescale_ops/');

[m, n] = size(data);

[indm, indn, data] = find(data);
% data(data == 0)= eps;
res = sparse(indm, indn, data, m, n);
% [indm, indn, data] = find(res);

% main iteration
U     = [];
V     = [];
Msup  = [];
i     = 0;
W     = 0;

% In regerssion pursuit, the stop criterion is small gradient of residual
obj = zeros(1, r);
RMSE = zeros(1, r);
Time = zeros(1, r);
Theta = [];
u = randn(m,1);
timeflag = tic;
while (i < r)
    % 1. find the top singular pair of the residual
    resvec = data - W; 
    obji = sum(resvec.^2);
    
    sparse_update(res, resvec);    
    [u, sv, v, pwIter] = topsvd(res, u, 50, 1e-5);

    % 2. update the weight Theta, the pursuit basis is uv', its weight is s.
    Mi = sparse_inp(u', v', indm, indn)';
    U    = [U u];
    V    = [V v];
    Msup = [Msup Mi];
    Sol  = inv(Msup'*Msup) * Msup' * data; % optimal line search by least square   
    if i == 0
        Theta = Sol;
    else
        Theta = [Theta * Sol(1) Sol(2)];
    end    
    W    = Msup * Sol; 
    Msup = W;
    
    i = i + 1;
    fprintf('iter: %d; obj: %.3d, pwIter: %d, sv: %d \n', i, obji, pwIter, sv);
    
    obj(i) = obji;
    Time(i) = toc(timeflag);
    if(isfield(para, 'test'))
        if(para.test.m ~= m)
            RMSE(i) = MatCompRMSE(V, U, diag(Theta), para.test.row, para.test.col, para.test.data);
        else
            RMSE(i) = MatCompRMSE(U, V, diag(Theta), para.test.row, para.test.col, para.test.data);
        end
        fprintf('RMSE %.2d \n', RMSE(i));
        
%         if(i > 1 && RMSE(i) > RMSE(i - 1))
%             break;
%         end
    end
end
numiter = i;
fprintf( '\n EOR1MP run %d rounds! \n', numiter);

out.obj = obj(1:i);
out.RMSE = RMSE(1:i);
out.Time = Time(1:i);

Theta = diag(Theta);

end

