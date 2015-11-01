function [U, S, V, out]=mc_alt(GXobs,lambda, para)
% [U, V, obj_vals, test_rmses, timelist] = mc_activealt(GXobs, lambda, GXtest, maxk, maxiter, mode)
%
% The active_alt algorithm for matrix completion problem; 
% solve the following optimization problem: 
%     
%     min_{X} 1/2 \sum_{i,j\in Omega} (A_{i,j} - X_{i,j})^2 + lambda * \|X\|_*
%
% Note: we recommend to shift the matrix to have zero mean before doing matrix completion 
% (since we use zero as initial solution). 
%
% Input Arguments: 
%   GXobs: observed rating matrix (A) -- a sparse matrix. 
%   lambda: regularization parameter -- a positive number. 
%   GXtest: testing ratings -- a sparse matrix. 
%   maxiter: maximum number of iterations -- a positive number. 
%   maxk: maximum number of rank -- a real number. 
%
% Output ArgumentS: 
%   U, V: The optimal X = UV'  
%   obj_vals: the history of objective function value for each iteration. 
%   test_rmses: the list of testing rmse for each iteration. 
%   timelist: the list of acummulated run time for each iteration. 
%

if(isfield('para', 'maxR') == 0)
    maxk = min(size(GXobs));
else
    maxk = para.maxR;
end

maxiter = para.maxIter;
tol = para.tol;

[m, n] = size(GXobs);
%obj_vals=zeros(maxiter,1);
GXobs=sparse(GXobs);

%% initial from zero
U=zeros(m,1); V=zeros(n,1);
R = GXobs;
initial_k = 10;

oldV = [];
nextk = initial_k;
flagTime = tic;
obj_vals = zeros(maxiter, 1);
RMSE = zeros(maxiter, 1);
Time = zeros(maxiter, 1);
for i = 1:maxiter 	
	kk = min(maxk, nextk);
	[u,s,v] = randomsvd(R, U, V, m, n, kk, oldV, 3);
	oldV = v;
	sing_vals=diag(s); 
	clear s;

	tmp=max(sing_vals-lambda,0);
	soft_singvals=tmp(tmp>0);
	no_singvals=length(soft_singvals);

	if (no_singvals == kk)
		nextk = ceil(kk*1.4);
	else
		nextk = no_singvals;
	end

	S = diag(soft_singvals);
	U = u(:,tmp>0); clear u;
	V = v(:,tmp>0); clear v;

	kk =  size(U,2);
	Z = S;
	for inner = 1:5
	   %% update S
		R = CompResidual(GXobs, (U*S)', V');
		A = inv(Z);
		AA = A+A';
		grad =  U'*R*V - 0.5*lambda*(AA)*S;
		xx = zeros(kk,kk);
		r = grad;
		p = r;
		rnorm = norm(r,'fro')^2;
		init_rnorm = rnorm;
		for cgiter = 1:10
			DUT = (V*p')';
			VDUT_Omega = CompProduct(GXobs, U', DUT);
			Ap = U'*VDUT_Omega*V+0.5*lambda*(p*AA);
			alpha = norm(r,'fro')^2/(sum(sum(p.*Ap)));
			xx = xx + alpha*p;
			r = r-alpha*Ap;
			rnorm_new = norm(r,'fro')^2;
%			fprintf('Inner iter %g, residual norm: %g\n', cgiter, rnorm_new);
			if ( rnorm < init_rnorm*1e-3)
				break;
			end
			beta = rnorm_new/rnorm;
			rnorm = rnorm_new;
			p = r+beta*p;
		end
		S = S + xx;

       %% Update Z
		Zi = (S*S')^(0.5);
        if(norm(Z - Zi, 'fro') < 1e-1/i)
            break;
        end
        Z = Zi;
	end

	[u, s, v] = svd(S);
	U=U*u(:,1:kk);
	V=V*v(:,1:kk);
	S = s(1:kk,1:kk);

	U = U*S;

	R = CompResidual(GXobs, U', V');
	train_err = norm(R,'fro')^2;

	objval_new = train_err/2 + lambda*sum(diag(S));
	obj_vals(i) = objval_new;
    
    if(i == 1)
        delta = inf;
    else
        delta = abs(obj_vals(i) - obj_vals(i - 1))/obj_vals(i);
    end
    
    Time(i) = toc(flagTime);
    if(isfield(para, 'test'))
        tempS = eye(size(U, 2), size(V, 2));
        if(para.test.m ~= m)
            RMSE(i) = MatCompRMSE(V, U, tempS, para.test.row, para.test.col, para.test.data);
        else
            RMSE(i) = MatCompRMSE(U, V, tempS, para.test.row, para.test.col, para.test.data);
        end
        fprintf('testing RMSE %.3d \n', RMSE(i));
    end
    
    if( delta < tol)
        break;
    end
    
	fprintf('Iter %.2d obj %.2d (%.3d)\n', i, objval_new, delta);
end

S = eye(size(U, 2), size(V, 2));

out.obj = obj_vals(1:i);
out.RMSE = RMSE(1:i);
out.Time = Time(1:i);
out.Rank = nnz(S);

