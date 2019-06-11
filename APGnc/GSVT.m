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
        z = max(s - lambda, 0);
        s(theta:end) = z(theta:end);
    otherwise
        assert(false);
end

% switch(regType)
%     case 1
%         w = cappedl1_sg(s, theta, lambda);
%     case 2
%         disp('not support!\n');
%     case 3
%         disp('not support!\n');
%     case 4
%         disp('not support!\n');
%     case 5
%         w = lambda*ones(size(s));
%         w(1:theta) = 0;
%     case 6
%         disp('not support!\n');
%     otherwise
%         disp('not support!\n');
% end
% s = s - w;

svs = sum(s > 1e-10);

U = U(:,1:svs);
V = V(:,1:svs);
S = diag(s(1:svs));

end