function [ X, Y, output ] = GISTRPCA( O, lambda, mu, theta1, theta2, para )

if(isfield(para, 'decay'))
    decay = para.decay;
else
    decay = 0.2;
end

if(isfield(para, 'maxR'))
    maxR = para.maxR;
else
    maxR = ceil(0.01*min(size(O)));
end

tol = para.tol;
tau = para.tau;
regType = para.regType;
maxIter = para.maxIter;

Y = zeros(size(O));
X = O;

r0 = 0;
r1 = maxR;

lambdaMax = topksvd(O, 1, 5);
obj = zeros(maxIter, 1);
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
    if(r0 + r1 > maxR)
        [ U, S, V ] = GSVT( X, lambdai, theta1i, regType, maxR);
    else
        [ U, S, V ] = GSVT( X, lambdai, theta1i, regType, r0 + r1);
    end
    X = U*S*V';
    
    r0 = max(r1, 1);
    r1 = max(nnz(S), 1); 
    
    Y = Y - (1/tau)*(X + Y - O);
    Y = reshape(Y, numel(Y), 1);
    Y = proximalRegC(Y, length(Y), mu/tau, theta2, 1);
    Y = reshape(Y, size(X,1), size(X,2));
    
    totalTime = totalTime + toc(timeFlag);
    Time(i) = totalTime;
    
    % check object and convergence
    obj(i) = getObjRPCA(X, S, Y, O, lambda, theta1, regType, mu, theta2);    
    fprintf('iter %d, obj %.4d, rank %d, lambda %.2d \n', ...
        i, obj(i), nnz(S), lambdai);
    
    if(isfield(para, 'test'))
        PSNR(i) = psnr(X + Y, para.test, 1);
        fprintf('PSNR %.3d \n', PSNR(i));
    end

    if(i > 1 && abs(obj(i) - obj(i - 1)) < tol)
        break;
    end
end

output.S    = diag(S);
output.rank =  nnz(S);

output.obj  = obj(1:i);
output.Time = Time(1:i);
output.PSNR = PSNR(1:i);

end

