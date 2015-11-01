function [objVal] = getObjRPCA(X, S, Y, O, lambda, theta1, ...
    regType, mu, theta2)

objVal = (1/2)*sum(sum((X + Y - O).^2));
objVal = objVal +  funRegC(diag(S), nnz(S), lambda, theta1, regType);
objVal = objVal +  funRegC(Y(:), numel(Y), mu, theta2, 1);

% objVal = objVal + regValue( diag(S), lambda, theta1, regType );
% objVal = objVal + regValue(Y, mu, theta2, 1 );

end