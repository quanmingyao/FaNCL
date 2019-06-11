function [U0, S, V, output ] = APGncext( D, lambda, theta, para )
% D: sparse observed matrix

% work well

if(isfield(para, 'maxR'))
    maxR = para.maxR;
else
    maxR = min(size(D));
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

R = randn(n, maxR);
U0 = powerMethod( D, R, 5, 1e-6);
U1 = U0;

[~, ~, V0] = svd(U0'*D, 'econ');
V1 = V0;

spa = sparse(row, col, data, m, n);

clear D;

obj = zeros(maxIter, 1);
RMSE = zeros(maxIter, 1);
Time = zeros(maxIter, 1);


c = 1;
part0 = partXY(U0', V0', row, col, length(data));
part1 = partXY(U1', V1', row, col, length(data));

for i = 1:maxIter
    tt = cputime;
    
    bi = (c - 1)/(c + 2);
    
    % make up sparse term Z = U*V' +spa
    part0 = data - (1 + bi)*part1' + bi*part0';
    setSval(spa, part0, length(part0));
    
    [Ui, S, Vi] = accExactSVD_APGnc( U1, V1, U0, V0, spa, bi, maxR);
    S = diag(S);
    S = proximalRegC(S, length(S), lambda, theta, regType);
    Ui = Ui*diag(S);

%     Q = powerMethodAccMatComp( U1, V1, U0, V0, spa, bi, V0, 15, 0);
%     hZ = ((1+bi)*(Q'*U1))*V1' - (bi*(Q'*U0))*V0' + Q'*spa;
%     [ Ui, S, Vi ] = GSVT(hZ, lambda, theta, regType);
%     Ui = Q*(Ui*S);

    U0 = U1;
    U1 = Ui;

    V0 = V1;
    V1 = Vi;
    
    part0 = part1;
    part1 = partXY_blas(Ui', Vi', row, col, length(data));
    
    objVal = (1/2)*sum((data - part1').^2);
    objVal = objVal + funRegC(S, length(S), lambda, theta, regType);
    
    if(i > 1 && objVal > getMaxOverk(obj(1:i - 1), 3))
        c = 1;
    else
        c = c + 1;
    end

    if(i > 1)
        delta = (obj(i - 1)- objVal)/objVal;
    else
        delta = inf;
    end
    
    fprintf('iter: %d; obj:(%d, %.2d); rank %d; lambda: %.1f; \n', ...
        i, objVal, delta, nnz(S), lambda );

    obj(i) = objVal;
    Time(i) = cputime - tt;
    
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
output.Rank = nnz(S);
output.RMSE = RMSE(1:i);
output.Time = cumsum(Time(1:i));

end

%% ------------------------------------------------------------------------
function [U, S, V] = accExactSVD_APGnc( U1, V1, U0, V0, spa, bi, k)

m = size(U1,1);
n = size(V1,1);
Afunc  = @(x) (spa*x + (1+bi)*(U1*(V1'*x)) - bi*(U0*(V0'*x)));
Atfunc = @(y) (spa'*y + (1+bi)*(V1*(U1'*y)) - bi*(V0*(U0'*y)));

rnk = min(min(m,n), ceil(1.25*k));
% rnk = k;
[U, S, V] = lansvd(Afunc, Atfunc, m, n, rnk, 'L');

U = U(:, 1:k);
V = V(:, 1:k);

S = diag(S);
S = S(1:k);
S = diag(S);

end

