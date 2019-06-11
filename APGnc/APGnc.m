function [U0, S, V, output ] = APGnc( D, lambda, theta, para )
% D: sparse observed matrix

% work well

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

objstep = 1;

maxIter = para.maxIter;
tol = para.tol*objstep;

regType = para.regType;
[row, col, data] = find(D);
[m, n] = size(D);

% U = randn(size(D, 1), 1);
% V0 = randn(size(D, 2), 1);
% V1 = V0;
% S = 1;

R = randn(n, 5);
U0 = powerMethod( D, R, 3, 1e-6);
U1 = U0;

[~, ~, V0] = svd(U0'*D, 'econ');
V1 = V0;

% lambdaMax = max(S(:));
% clear S;

spa = sparse(row, col, data, m, n);

clear D;

obj = zeros(maxIter, 1);
RMSE = zeros(maxIter, 1);
Time = zeros(maxIter, 1);
rankIn = zeros(maxIter, 1);
rankOt = zeros(maxIter, 1);

part0 = partXY(U0', V0', row, col, length(data));
part1 = partXY(U1', V1', row, col, length(data));
c = 1;

for i = 1:maxIter
    tt = cputime;
    
%     switch(para.regType)
%         case 1
%             lambdai = abs(lambdaMax - lambda)*(decay^i) + lambda;
%             thetai = theta + (decay^i)*lambdai;
%         case 2
%             lambdai = lambda;
%             thetai = theta;
%         case 3
%             lambdai = abs(lambdaMax - lambda)*(decay^i) + lambda;
%             thetai = theta;
%     end
    
    bi = (c - 1)/(c + 2);
    
    % make up sparse term Z = U*V' +spa
    part0 = data - (1 + bi)*part1' + bi*part0';
    setSval(spa, part0, length(part0));
    
    rankIn(i) = size(V0, 2);
    
    R = filterBase( V1, V0, 1e-6*sqrt(i));
    [Q, pwIter] = powerMethodAccMatComp( U1, V1, U0, V0, spa, bi, R, 3, 0);
    hZ = ((1+bi)*(Q'*U1))*V1' - (bi*(Q'*U0))*V0' + Q'*spa;
    [ Ui, S, Vi ] = GSVT(hZ, lambda, theta, regType);
    Ui = Q*(Ui*S);
    
    rankOt(i) = size(Ui, 2);
   
    U0 = U1;
    U1 = Ui;

    V0 = V1;
    V1 = Vi;
    
    part0 = part1;
    part1 = sparse_inp(Ui', Vi', row, col);
    % part1 = partXY_blas(Ui', Vi', row, col, length(data));
    
    objVal = (1/2)*sum((data - part1').^2);
    objVal = objVal + funRegC(diag(S), nnz(S), lambda, theta, regType);
    
    if(i > 1 && objVal > getMaxOverk(obj(1:i - 1), 5))
        c = 1;
    else
        c = c + 1;
    end

    if(i > 1)
        delta = (obj(i - 1)- objVal)/objVal;
    else
        delta = inf;
    end
    
    fprintf('iter: %d; obj: %.3d (dif: %.3d); rank %d; lambda: %.1f; power(%d,%d); acc:%d \n', ...
        i, objVal, delta, nnz(S), lambda, pwIter, rankIn(i), c);
    
    Time(i) = cputime - tt;
    obj(i) = objVal;
    
    % testing performance
    if(isfield(para, 'test'))
        tempS = eye(size(U1,2), size(V1,2));
        if(para.test.m ~= m)
            RMSE(i) = MatCompRMSE(V1, U1, tempS, para.test.row, para.test.col, para.test.data);
        else
            RMSE(i) = MatCompRMSE(U1, V1, tempS, para.test.row, para.test.col, para.test.data);
        end
        fprintf('RMSE %.2d \n', RMSE(i));
    end
    
    if(i > 1 && abs(delta) < tol)
        break;
    end
end

output.obj = obj(1:i);
[U0, S, V] = svd(U1, 'econ');
V = V1*V;

output.RMSE = RMSE(1:i);
output.Time = cumsum( Time(1:i) );
output.rankin = rankIn(1:i);
output.rankout = rankOt(1:i);

end
