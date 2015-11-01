function [output ] = FastMatCompFR( D, theta, para )
% D: sparse observed matrix

speedup = para.speedup;
maxIter = para.maxIter;
tol = para.tol;

[row, col, data] = find(D);
[m, n] = size(D);

R = randn(n, theta);
U = powerMethod( D, R, 3, 1e-6);
V = D'*U;
Z = sparse(row, col, data, m, n);

clear D;

obj = zeros(maxIter, 1);
for i = 1:maxIter
    % lambdai = (lambdaMax - lambda)*(0.95^i) + lambda/tau;
    
    % make up sparse term Z = U*V' +spa
    spa = partXY(U', V', row, col, length(data));
    spa = data - spa';
    objVal = (1/2)*sum(spa.^2);
    Z = setSval(Z, spa, length(spa));
    
    R = V;
    if(speedup == 1)
        [Q, pwIter] = powerMethodMatComp( U, V, Z, R, 3, 1e-8*sqrt(m*n));
    else
        A = U*V' + Z;
        [Q, pwIter] = powerMethod( A, R, 3, tol);
    end
    V = (Q'*U)*V' + Q'*Z;
    V = V';
    U = Q;
    
    fprintf('iter: %d; obj: %d; power(iter %d, rank %d) \n', ...
        i, objVal, pwIter, size(R, 2))
    obj(i) = objVal;
    if(i > 1 && abs(obj(i) - obj(i-1)) < tol)
        break;
    end
end

output.obj = obj(1:i);
output.U = U;
output.S = eye(size(U,2));
output.V = V;

end

%% --------------------------------------------------------------
