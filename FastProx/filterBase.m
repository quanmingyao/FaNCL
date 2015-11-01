function [ R ] = filterBase( V1, V0, tol )

R = V0 - V1*(V1'*V0);
R = sum(R.^2, 1);
R = (R > tol);
R = [V1, V0(:, R)];

end

