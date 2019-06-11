function [U, S, V, output ] = FastMatComp( D, lambda, theta, para )
% D: sparse observed matrix

if(isfield(para, 'maxR'))
    maxR = para.maxR;
else
    maxR = min(size(D));
end

if(isfield(para, 'decay'))
    decay = para.decay;
else
    decay = 0.96;
end

if(isfield(para, 'speedup'))
    speedup = para.speedup;
else
    speedup = 1;
end

maxIter = para.maxIter;
tol = para.tol;
regType = para.regType;

[row, col, data] = find(D);
[m, n] = size(D);

% U = randn(size(D, 1), 1);
% V0 = randn(size(D, 2), 1);
% V1 = V0;
% S = 1;

R = randn(n, 5);
U = powerMethod( D, R, 3, 1e-6);
[R, S, V0] = svd(U'*D, 'econ');
lambdaMax = max(S(:));
V1 = V0;
U = U*(R*S);

Z = sparse(row, col, data, m, n);

clear D;

obj = zeros(maxIter, 1);
rankIn = zeros(maxIter, 1);
rankOt = zeros(maxIter, 1);
RMSE = zeros(maxIter, 1);
Time = zeros(maxIter, 1);
flagTime = tic;
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
    
    % make up sparse term Z = U*V' +spa
    spa = partXY(U', V1', row, col, length(data));
    spa = data - spa';
    setSval(Z, spa, length(spa));
    
    R = filterBase( V1, V0, 1e-6*sqrt(i));
    rankIn(i) = min(size(R,2), maxR);
    R = R(:, 1:rankIn(i));

    if(speedup == 1)
        [Q, pwIter] = powerMethodMatComp( U, V1, Z, R, 3, 0);
    else
        if(isempty(R))
            pwIter = 0;
            Q = zeros(size(U,1), 1);
        else
            A = U*V1' + Z;
            [Q, pwIter] = powerMethod( A, R, 3, 0);
        end
    end
    hZ = (Q'*U)*V1' + Q'*Z;
    [ U, S, V ] = GSVT(hZ, lambdai, thetai, regType);
    rankOt(i) = size(V, 2);
    
    U = Q*(U*S);
    V0 = V1;
    V1 = V;
    
    obj(i) = (1/2)*sum(spa.^2);
    obj(i) = obj(i) + funRegC(diag(S), nnz(S), lambda, theta, regType);

    if(i > 1)
        delta = (obj(i-1) - obj(i))/obj(i);
    else
        delta = inf;
    end

    fprintf('iter:%d; (obj:%.3d, tol:%.3d); rank %d; lambda: %.1f; power(iter %d, rank %d) \n', ...
        i, obj(i), delta, nnz(S), lambdai, pwIter, size(R, 2))
    Time(i) = toc(flagTime);
    
    if(isfield(para, 'test'))
        tempS = eye(size(U,2), size(V,2));
        if(para.test.m ~= m)
            RMSE(i) = MatCompRMSE(V, U, tempS, para.test.row, para.test.col, para.test.data);
        else
            RMSE(i) = MatCompRMSE(U, V, tempS, para.test.row, para.test.col, para.test.data);
        end
        fprintf('RMSE %.2d \n', RMSE(i));
    end
    
    if( delta > 0 && delta < tol )
        break;
    end
end

output.obj = obj(1:i);
output.RMSE = RMSE(1:i);
output.Time = Time(1:i);
output.rankin = rankIn(1:i);
output.rankout = rankOt(1:i);

[U, S, V] = svd(U, 'econ');
V = V1*V;

end

%% --------------------------------------------------------------
