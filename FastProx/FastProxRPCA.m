function [ X, Y, output ] = FastProxRPCA( O, lambda, mu, theta1, theta2, para)

if(isfield(para, 'decay'))
    decay = para.decay;
else
    decay = 0.2;
end

if(isfield(para, 'maxR'))
    maxR = para.maxR;
else
    maxR = min(size(O));
end

tol = para.tol;
tau = para.tau;
regType = para.regType;
maxIter = para.maxIter;

Y = zeros(size(O));
X = O;

V = randn(size(O,2), 1);
V = powerMethod(O, V, 1, 1e-6);
[~, ~, V] = svd(V'*O, 'econ');
V0 = V;
V1 = V;

lambdaMax = topksvd(O, 1, 5);

obj  = zeros(maxIter, 1);
Time = zeros(maxIter, 1);
PSNR = zeros(maxIter, 1);
totalTime = 0;
for i = 1:maxIter
    timeFlag = tic;
    
    % setup loop parameter
    switch(regType)
        case 1 % CAP
            lambdai = abs(lambdaMax - lambda)*(decay^i) + lambda/tau;
            theta1i = theta1 + (decay^i)*lambdai;
        case 2 % Logrithm
            lambdai = lambda;
            theta1i = theta1;
        case 3 % TNN
            lambdai = abs(lambdaMax - lambda)*(decay^i) + lambda;
            theta1i = theta1;
        otherwise
            assert(false);
    end

    X = X - (1/tau)*(X + Y - O);
    
    [ R ] = filterBase( V1, V0, 1e-6);
    R = R(:,1:min(size(R,2), maxR));
    
    [Q, pwIter] = powerMethod( X, R, 3, 1e-5);
    hZ = Q'*X;
    [ U, S, V ] = GSVT( hZ, lambdai, theta1i, regType);
    if(nnz(S) > 0)
        X = (Q*U)*(S*V');
        V0 = V1;
        V1 = V;
    end
    
    Y = Y - (1/tau)*(X + Y - O);
    Y = reshape(Y, numel(Y), 1);
    Y = proximalRegC(Y, length(Y), mu/tau, theta2, 1);
    Y = reshape(Y, size(X,1), size(X,2));
    
    totalTime = totalTime + toc(timeFlag);
    Time(i) = totalTime;
    
    obj(i) = getObjRPCA(X, S, Y, O, lambda, theta1, regType, mu, theta2);   
    if(i == 1)
        deltaObj = inf;
    else
        deltaObj = obj(i - 1) - obj(i);
    end

    fprintf('iter %d, obj %.4d(dif %.2d), rank %d, lambda %.2d, power(rank %d, iter %d), nnz:%0.2f \n', ...
            i, obj(i), deltaObj, nnz(S), lambdai, size(R, 2), pwIter, nnz(Y)/numel(Y));
        
    if(isfield(para, 'test'))
        PSNR(i) = psnr(X + Y, para.test, 1);
        fprintf('PSNR %.3d \n', PSNR(i));
    end

    if(abs(deltaObj) < tol)
        break;
    end
end

output.S    = diag(S);
output.rank =  nnz(S);

output.obj  = obj(1:i);
output.Time = Time(1:i);
output.PSNR = PSNR(1:i);

end

