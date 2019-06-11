function [u, s, v, i] = topsvd(A, u, round, stopeps)
% calculate the top svd of matrix A using round iterations
if(~exist('stopeps', 'var'))
    stopeps = 1e-4;
end

vo      = 0;
for i=1:round
    v = u'*A/(norm(u))^2;
    u = A*v'/(norm(v))^2;
    if norm(v-vo) < stopeps
        break
    end
    vo = v;
end
nu = norm(u);
nv = norm(v);
u  = u/nu;
v  = v'/nv;
s  = nu*nv;

end
