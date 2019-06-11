function [ X, Y1, output ] = APGRPCA( O, lambda, mu, para)

if(isfield(para, 'decay'))
    decay = para.decay;
else
    decay = 0.2;
end

maxIter = para.maxIter;
tol = para.tol;

Y0 = O;
Y1 = O;

lambdaMax = topksvd( O, 1 );

maxR = floor(0.05*min(size(O)));
r0 = 0;
r1 = maxR;
t0 = 1;
t1 = 1;

obj  = zeros(maxIter, 1);
Time = zeros(maxIter, 1);
PSNR = zeros(maxIter, 1);
totalTime = 0;
for i = 1:maxIter
    timeFlag = tic;
    
    lambdai = abs(lambdaMax - lambda)*(decay^i) + lambda;
%     lambdai = lambda;

    Yi = Y1 + (t0 - 1)/t1 * (Y1 - Y0);
    
    X = O - Yi;
    if(r0 + r1 > maxR)
        [ U, S, V ] = GSVT( X, lambdai, 1e+8, 1);
    else
        [ U, S, V ] = GSVT( X, lambdai, 1e+8, 1, r0 + r1);
    end
    % [ U, S, V ] = proximalOperator( X, lambdai, theta, regType);
    X = U*S*V';
    
    r0 = r1;
    r1 = nnz(S); 
    
    Y0 = Y1;
    
    Y1 = O - X;
    Y1 = reshape(Y1, numel(Y1), 1);
    Y1 = proximalRegC(Y1, length(Y1), mu, 1e+8, 1);
    Y1 = reshape(Y1, size(X,1), size(X,2));
    
    ti = t1;
    t1 = 0.5*(1 + sqrt(1 + 4*t0^2));
    t0 = ti;
    
    totalTime = totalTime + toc(timeFlag);
    Time(i) = totalTime;
    
    % check object and convergence
    obj(i) = getObjRPCA(X, S, Y1, O, lambda, 1e+8, 1, mu, 1e+8);    
    fprintf('iter %d, obj %.4d, rank %d, lambda %.2d, nnz: %.2f \n', ...
            i, obj(i), nnz(S), lambdai, nnz(Y1)/numel(Y1));
        
    if(isfield(para, 'test'))
        PSNR(i) = psnr(X + Y1, para.test, 1);
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

