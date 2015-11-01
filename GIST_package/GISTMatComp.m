function [ U, S, V, output ] = GISTMatComp( O, lambda, theta, para)

if(isfield(para, 'decay'))
    decay = para.decay;
else
    decay = 0.95;
end

if(isfield(para, 'maxR'))
    maxR = para.maxR;
else
    maxR = floor(0.05*min(size(O)));
end

lambdaMax = topksvd(O, 1, 5);
Omega = sparse(O ~= 0);
X = full(O);

maxIter = para.maxIter;
tol = para.tol;
regType = para.regType;

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
    
    Z = X - (X.*Omega - O.*Omega);
    
    [ U, S, V ] = GSVT(Z, lambdai, thetai, regType);
    U = U(:, 1:min(size(U,2), maxR));
    V = V(:, 1:min(size(V,2), maxR));
    S = S(1:min(size(S,1), maxR), 1:min(size(S,2), maxR));
    X = U*S*V';
    
    obj(i) = (1/2)*sumsqr((X.*Omega - O.*Omega));
    obj(i) = obj(i) + funRegC(diag(S), nnz(S), lambda, theta, regType);
    
    if(i > 1)
        delta = abs(obj(i) - obj(i-1));
    else
        delta = inf;
    end
    
    fprintf('iter %d, (obj:%.3d, tol:%.3d), rank %d, lambda %.2d \n', ...
        i, obj(i), delta, nnz(S), lambdai);
    Time(i) = toc(flagTime);
    
    if(isfield(para, 'test'))
        RMSE(i) = MatCompRMSE(U, V, S, para.test.row, para.test.col, para.test.data);
        fprintf('RMSE %.2d \n', RMSE(i));
    end
    
    if(delta < tol)
        break;
    end
end

output.obj = obj(1:i);
output.RMSE = RMSE(1:i);
output.Time = Time(1:i);
output.rank = nnz(S);

end

