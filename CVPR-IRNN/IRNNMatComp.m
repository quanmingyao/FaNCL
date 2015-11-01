function [ U, S, V, output ] = IRNNMatComp( O, lambda, theta, para )

if(isfield(para, 'decay'))
    decay = para.decay;
else
    decay = 0.96;
end

if(isfield(para, 'maxR'))
    maxR = para.maxR;
else
    maxR = floor(0.05*min(size(O)));
end

Omega = sparse(O ~= 0);

maxIter = para.maxIter;
tol = para.tol;
tau = para.tau;
regType = para.regType;

X = full(O);
s = svd(X);
lambdaMax = max(s(:));
s = 1e+6*ones(length(s), 1);

flagTime = tic;
obj = zeros(maxIter, 1);
Time = zeros(maxIter, 1);
RMSE = zeros(maxIter, 1);
for i = 1:maxIter
    switch(para.regType)
        case 1
            lambdai = abs(lambdaMax - lambda)*(decay^i) + lambda;
            thetai = theta + (decay^i)*lambdai;
        case 2
            lambdai = lambda;
            thetai = theta;
        case 3
            lambdai = abs(lambdaMax - lambda)*(decay^i) + lambda;
            thetai = theta;
    end
    
    % get weight
    [ w ] = getWeight( s, thetai, lambdai, regType );
    
    Z = X - (1/tau)*(X.*Omega - O.*Omega);
    [U, s, V] = svd(Z, 'econ');
    s = diag(s);
    s = max(s - w/tau, 0);
    nnzS = sum(s > 1e-6);
    
    nnzS = min(nnzS, maxR);
    
    U = U(:, 1:nnzS);
    V = V(:, 1:nnzS);
    X = U*diag(s(1:nnzS))*V';
    
    obj(i) = (1/2)*sumsqr((X.*Omega - O.*Omega));
    obj(i) = obj(i) + funRegC(s, length(s), lambda, theta, regType);
    
    if(i > 1)
        delta = abs(obj(i) - obj(i-1));
    else
        delta = inf;
    end
    
    fprintf('iter %d, (obj:%.3d, tol:%.3d), rank %d, lambda %.2d \n', ...
        i, obj(i), delta, nnz(s), lambdai);
    Time(i) = toc(flagTime);
    
    if(isfield(para, 'test'))
        S = diag(s(1:nnzS));
        RMSE(i) = MatCompRMSE(U, V, S, para.test.row, para.test.col, para.test.data);
        fprintf('RMSE %.2d \n', RMSE(i));
    end
    
    if(delta < tol)
        break;
    end
end

s = s(s > 0);
S = diag(s);

output.rank = nnz(s);
output.obj = obj(1:i);
output.RMSE = RMSE(1:i);
output.Time = Time(1:i);

end



