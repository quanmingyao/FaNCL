function [ U, S, V, output ] = APGMatComp( O, lambda, para )

if(isfield(para, 'maxTime'))
    maxTime = para.maxTime;
else
    maxTime = 1e+9;
end

if(isfield(para, 'decay'))
    decay = para.decay;
else
    decay = 0.9;
end

X0 = full(O);
X1 = X0;
Omega = (sparse(O) ~= 0);

lambdaMax = svds(O, 1);
maxIter = para.maxIter;
tol = para.tol;

maxR = floor(0.01*min(size(O)));
% maxR = 0;
r0 = 0;
t0 = 1;
t1 = 1;

flagTime = tic;
Time = zeros(maxIter, 1);
RMSE = zeros(maxIter, 1);
obj = zeros(maxIter, 1);
for i = 1:maxIter
    lambdai = abs(lambdaMax - lambda)*(decay^i) + lambda;
    
    Xi = X1 + ((t0 - 1)/t1)*(X1 - X0);
    Z = Xi - (Xi.*Omega - O.*Omega);
    % choose SVD type
    if(r0 > maxR)
        [ U, S, V ] = GSVT( Z, lambdai, 1e+8, 1);
    else
        [ U, S, V ] = GSVT( Z, lambdai, 1e+8, 1, r0);
    end
    if(r0 <= nnz(S))
        r0 = r0 + 5;
    else
        r0 = nnz(S) + 1;
    end
    
    Xi = U*S*V';
    X0 = X1;
    X1 = Xi;
    
    ti = (1 + sqrt(1 + 4*t0^2))/2;
    t0 = t1;
    t1 = ti;
    
    % check objective value    
    obj(i) = (1/2)*sumsqr((Xi.*Omega - O.*Omega));
    obj(i) = obj(i) + funRegC(diag(S), nnz(S), lambda, 1e+9, 1);
    
    if(i > 1)
        delta = abs(obj(i) - obj(i-1));
    else
        delta = inf;
    end
    
    Time(i) = toc(flagTime);
    fprintf('iter %d, (obj:%.3d, tol:%.3d), rank %d, lambda %.2d \n', ...
        i, obj(i), delta, nnz(S), lambdai);
    
    if(isfield(para, 'test'))
        RMSE(i) = MatCompRMSE(U, V, S, ...
            para.test.row, para.test.col, para.test.data);
        fprintf('RMSE %.2d \n', RMSE(i));
    end
    
    if(delta < tol)
        break;
    end
end

output.rank = nnz(S);
output.obj = obj(1:i);
output.RMSE = RMSE(1:i);
output.Time = Time(1:i);

end

