function [U, S, V, out] = AltMinLs (D, k, bias, para)

[row, col, data] = find(D);
[m, n] = size(D);

U = randn(m, k)*0.001;
V = randn(n, k)*0.001;

tol = para.tol;
maxIter = para.maxIter;

spa = sparse(row, col, data, m, n);

obj = zeros(maxIter, 1);
RMSE = zeros(1, maxIter);
Time = zeros(maxIter, 1);

tt = tic;

lsZ = partXY(U', V', row, col, length(col))';
lsZ = lsZ + bias - data;

obj(1)  = (1/2)*sum(lsZ.^2);
Time(1) = toc(tt);

stepsize = norm(U, 'fro') + norm(V, 'fro');
for i = 1:maxIter
    tt = tic;
    
    setSval(spa, lsZ, length(col));

    for j = 1:20
        lsU = U - (1/stepsize)*(spa *V);
        lsV = V - (1/stepsize)*(spa'*U);
        
        lsZ = partXY(lsU', lsV', row, col, length(col))';
        lsZ = lsZ + bias - data;
        lsobj = (1/2)*sum(lsZ.^2);
        
        if(lsobj < obj(i))
            U = lsU;
            V = lsV;
            obj(i + 1) = lsobj;
            
            stepsize = stepsize*0.99;
            break;
        else
            stepsize = stepsize*2;
        end
    end

    Time(i + 1) = Time(i) + toc(tt);
    delta = (obj(i) - obj(i + 1))/obj(i);
    
    fprintf('iter:%d, obj:%.3d(%.2d), ls:(%.2d, %d) \n', ...
        i, obj(i), delta, stepsize, j);
    % testing performance
    if(isfield(para, 'test'))        
        tempS = eye(size(U, 2), size(V, 2));
        RMSE(i) = MatCompRMSE(U, V, tempS, ...
            para.test.row, para.test.col, para.test.data);
        fprintf('RMSE %.2d \n', RMSE(i));
    end
    
    if(j == 20)
        break;
    end
    
    if(i > 5*k && delta < tol )
        break;
    end
    
    if(obj(i) < tol )
        break;
    end
end

[U, S, V] = filteroutBoost(U, V, size(U, 2));

out.obj = obj(1:i);
out.RMSE = RMSE(1:i);
out.rank = nnz(S);
out.Time = Time(1:i);

end

