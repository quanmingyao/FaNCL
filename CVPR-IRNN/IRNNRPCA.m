function [ X, Y, output ] = IRNNRPCA( O, lambda, mu, theta1, theta2, para)

if(isfield(para, 'decay'))
    decay = para.decay;
else
    decay = 0.2;
end

objstep = para.objstep;

tol = para.tol*objstep;
tau = para.tau;
regType = para.regType;
maxIter = para.maxIter;

Y = zeros(size(O));
X = O;
S = diag(svd(X, 'econ'));
lambdaMax = max(S(:));

objcnt = 1;
obj = zeros(maxIter, 1);
for i = 1:maxIter
    % setup loop parameter
    switch(regType)
        case 1 % CAP
            % lambdai = abs(lambdaMax - lambda)*(decay^i) + lambda;
            % theta1i = theta1 + (decay^i)*lambdai;
            lambdai = lambda;
            theta1i = theta1;
        case 2 % Logrithm
            lambdai = lambda;
            theta1i = theta1;
        case 3 % TNN
            lambdai = abs(lambdaMax - lambda)*(decay^i) + lambda;
            theta1i = theta1;
        otherwise
            assert(false);
    end
    
    % get weight
    [ w ] = getWeight( diag(S), theta1i, lambdai, regType );
    
    % compute SVD of proximal map
    Z = X - (1/tau)*(X + Y - O);
    [U, S, V] = svd(Z, 'econ');
    
    % make up X
    S = max(diag(S) - w/tau, 0);
    nnzS = sum(S > 1e-6);
    S = diag(S);
    U = U(:,1:nnzS);
    V = V(:,1:nnzS);
    X = U*S(1:nnzS,1:nnzS)*V';
    
    % proximal on Y
    Y = Y - (1/tau)*(X + Y - O);
    Y = reshape(Y, numel(Y), 1);
    Y = proximalRegC(Y, length(Y), mu/tau, theta2, 1);
    Y = reshape(Y, size(X,1), size(X,2));

    if(mod(i, objstep) == 0)
        obj(objcnt) = getObjRPCA(X, S, Y, O, lambda, theta1, regType, mu, theta2);    
        fprintf('iter %d, obj %d, rank %d \n', i, obj(objcnt), nnzS);
        
        if(objcnt > 1 && abs(obj(objcnt) - obj(objcnt - 1)) < tol)
            break;
        end
        
        objcnt = objcnt +1;
    end
end

output.S = diag(S);
output.obj = obj(1:objcnt - 1);

end




