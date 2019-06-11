function [Q, maxIter] = powerMethodAccMatComp( U1, V1, U0, V0, spa, bi, ...
    R, maxIter, tol)

Y = U1*((V1'*R)*(1 + bi));
Y = Y - U0*((V0'*R)*bi);
Y = Y + spa*R;

[Q, ~] = qr(Y, 0);
err = zeros(maxIter, 1);
for i = 1:maxIter
    % Y = A*(A'*Q);
    AtQ = ((1 + bi)*(Q'*U1))*V1';
    AtQ = AtQ - (bi*(Q'*U0))*V0';
    AtQ = AtQ';
    
    Y = U1*((V1'*AtQ)*(1 + bi));
    Y = Y - U0*((V0'*AtQ)*bi);
    Y = Y + spa*AtQ;
    
    [iQ, ~] = qr(Y, 0);
    
    err(i) = norm(iQ(:,1) - Q(:,1), 2);
    Q = iQ;
    
    if(err(i) < tol)
        break;
    end
end

maxIter = i;

end