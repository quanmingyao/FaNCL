function [U, S, V, output ] = FastMatCompCheck( D, lambda, theta, para )
% D: sparse observed matrix

if(isfield(para, 'maxR'))
    maxR = para.maxR;
else
    maxR = min(size(D));
end

if(isfield(para, 'maxTime'))
    maxTime = para.maxTime;
else
    maxTime = 1e+9;
end

if(isfield(para, 'decay'))
    decay = para.decay;
else
    decay = 0.96;
end

speedup = para.speedup;
maxIter = para.maxIter;
tol = para.tol;
regType = para.regType;

[row, col, data] = find(D);
[m, n] = size(D);

R = randn(n, 5);
U = powerMethod( D, R, 3, 1e-6);
[R, S, V0] = svd(U'*D, 'econ');
lambdaMax = max(S(:));
V1 = V0;
U = U*(R*S);

Z = sparse(row, col, data, m, n);
i = 1;

clear D;

flagTime = tic;
obj = zeros(maxIter, 1);
rankIn = zeros(maxIter, 1);
rankOut = zeros(maxIter, 1);
objcheck = zeros(maxIter, 1);
objdiff = zeros(maxIter, 1);
iterq   = zeros(maxIter, 1);
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
    Z = setSval(Z, spa, length(spa));
    
    [ R ] = filterBase( V1, V0, 1e-7);
    rankIn(i) = min(size(R,2), maxR);
    R = R(:, 1:rankIn(i));
    pwTol = max(lambdaMax*0.9^i, 1e-4);
    for q = 1:100
        if(speedup == 1)
            [Q, pwIter] = powerMethodMatComp( U, V1, Z, R, 3, pwTol);
        else
            if(isempty(R))
                pwIter = 0;
                Q = zeros(size(U,1), 1);
            else
                A = U*V1' + Z;
                [Q, pwIter] = powerMethod( A, R, 3, 1e-8*sqrt(m*n));
            end
        end
        hZ = (Q'*U)*V1' + Q'*Z;
        [ Uq, Sq, Vq ] = proximalOperator(hZ, lambdai, thetai, regType);
        
        Uq = Q*(Uq*Sq);
        
        objnew = inf;
        if(i == 1)
            break;
        end
        
        objnew = partXY(Uq', Vq', row, col, length(data));
        objnew = data - objnew';
        objnew = 0.5*sum(objnew.^2);
        objnew = objnew + funRegC(diag(Sq), nnz(Sq), lambda, theta, regType);
        objcheck(i) = (0.01/4)*fastnorm(U, V1, Uq, Vq);
        objdiff(i)  = obj(i - 1) - objnew;
        
        if(objcheck(i) < objdiff(i)) 
            break;
        end
        R = Vq;
    end
    iterq(i) = q;
    obj(i) = objnew;
    rankOut(i) = size(Vq, 2);
    
    U = Uq;
    V0 = V1;
    V1 = Vq;
    
%     obj(i) = (1/2)*sum(spa.^2);
%     obj(i) = obj(i) + funRegC(diag(S), nnz(S), lambda, theta, regType);

    if(i > 1)
        delta = obj(i-1) - obj(i);
    else
        delta = inf;
    end

    fprintf('iter:%d; (obj:%.3d, tol:%.3d); rank %d; lambda: %.1f; power(iter %d, rank %d) \n', ...
        i, obj(i), delta, nnz(S), lambdai, pwIter, size(R, 2))
    runTime = toc(flagTime);
    if(abs(delta) < tol || runTime > maxTime)
        break;
    end
end

output.obj = obj(1:i-1);
[U, S, V] = svd(U, 'econ');
V = V1*V;
output.rank = diag(S);
output.rankin = rankIn(1:i);
output.rankout = rankOut(1:i);
output.objcheck = objcheck(1:i);
output.objdiff = objdiff(1:i);
output.iterq = iterq(1:i);

end

%% --------------------------------------------------------------
function temp = fastnorm(U1, V1, U2, V2)

[U, ~] = qr([U1, U2], 0);
[V, ~] = qr([V1, V2], 0);

temp = (U'*U1)*(V'*V1)' - (U'*U2)*(V'*V2)' ;
temp = temp(:);
temp = temp.^2;
temp = sum(temp);

end
