function [ U, S, V ] = GSVT( Z, lambda, theta, regType, rnk )
%% ------------------------------------------------------------------------
% exact solve low rank proximal step
% (1/2)*|X - Z|_F^2 + lambda |X|_theta
%% ------------------------------------------------------------------------
%  regtype = 1: Capped L1 regularizer 
%            2: Log Sum Penalty
%            3: TNN
%% ------------------------------------------------------------------------

if(exist('rnk', 'var'))
    [U, S, V] = lansvd(Z, rnk, 'L');
else
    [U, S, V] = svd(Z, 'econ');
end

s = diag(S);

switch(regType)
    case 1 % CAP
        s = proximalRegC(s, length(s), lambda, theta, 1);
    case 2 % Logrithm
        s = proximalRegC(s, length(s), lambda, theta, 2);
    case 3 % TNN
        s = proximalRegC(s, length(s), lambda, theta, 3);
    otherwise
        assert(false);
end

svs = sum(s > 1e-10);

U = U(:,1:svs);
V = V(:,1:svs);
S = diag(s(1:svs));

end