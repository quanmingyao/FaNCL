clear;

Z = 6;

[U, S, V] = svd(Z, 'econ');

lambda = 10;
theta = 1;

% WSVT
s2 = diag(S);
X = Z;
rho = 1.02;
for i = 1:100
    w = logarithm_sg(s2, theta, lambda); 
    
    Pi = X - (1/rho)*(X - Z);
    [U, Pi, V] = svd(Pi, 'econ');
    Pi = diag(Pi);
    
    s2 = max(Pi - w/rho, 0);   
    
    X = U*diag(s2)*V';
end

for i = 1:length(s2)
    di = s2(i);
    zi = S(i, i);
    
    h0 = (1/2)*zi^2;
    hx = (1/2)*(di - zi)^2 + lambda*log(1 + di/theta);
    
    if(h0 < hx)
        s2(i) = 0;
    end
end

% proximal
s1 = diag(S);
s1 = proximalRegC(s1, length(s1), lambda, theta, 2);