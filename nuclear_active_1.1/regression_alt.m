function [U, V, obj_vals, test_rmses, timelist]=regression_alt(A, B, At, Bt, lambda, maxk, maxiter)
% [U, V, obj_vals, test_rmses, timelist] = regression_alt(A, B, At, Bt, lambda, maxk, maxiter)
%
% The active_alt algorithm for matrix completion problem; 
% solve the following optimization problem: 
%     
%     min_X ||AX-B||_F^2/2 + lambda||X||_*
%
% Input Arguments: 
%   A, B: The training data. 
%   At, Bt: The testing data. 
%   lambda: regularization parameter -- a positive number.
%   maxk: maximum number of rank -- a real number. 
%   maxiter: maximum number of iterations -- a positive number. 
%
% Output ArgumentS: 
%   U, V: The optimal X = U*V'
%   obj_vals: the history of objective function value for each iteration. 
%   test_rmses: the list of testing rmse for each iteration. 
%   timelist: the list of acummulated run time for each iteration. 
%

maxNumCompThreads(1);
m = size(A,2);
r = size(A,1);
n = size(B,2);

U=zeros(m,1); V=zeros(n,1);
U = 0.001*randn(m,1); V = 0.001*randn(n,1);
S=zeros(1,1);

i=0; 
timebegin = cputime;
timesvd = 0;
totaltime = 0;
timelist= [];
oldV = [];
nextk = 10;
ssss = [0];
for iter = 1:maxiter 
	timebegin = cputime;

	kk = max(nextk, size(U,2));
	kk
	[u,s,v] = randomsvd_regression(U, V, -A'*(A*U), V, A', B', m, n, kk, oldV, 5);
	oldV = v;
	sing_vals=diag(s); clear s;

	tmp=max(sing_vals-lambda,0);
	soft_singvals=tmp(tmp>0);
	no_singvals=length(soft_singvals);
	if (no_singvals < kk )
		nextk = no_singvals;
	else
		nextk = ceil(kk*1.4);
	end
	kkk = size(U,2);
%	[qu ru] = comp_qr([U u(:,1:no_singvals)]);
%	[qv rv] = comp_qr([V v(:,1:no_singvals)]);
	[qu ru] = qr([ U u(:,1:no_singvals)],0);
	[qv rv] = qr([ V v(:,1:no_singvals)],0);
	for i=1:kkk
		if ( ru(i,i) <0)
			ru(i,i) = abs(ru(i,i));
			qu(:,i) = qu(:,i)*(-1);
		end
		if ( rv(i,i) < 0 )
			rv(i,i) = abs(rv(i,i));
			qv(:,i) = qv(:,i)*(-1);
		end
	end
	S = [ru(1:kkk,1:kkk)*rv(1:kkk,1:kkk)' zeros(size(U,2),no_singvals); zeros( no_singvals, size(V,2)), zeros(no_singvals) ];
	U = qu;
	V = qv;
	
	%% now: solve 0.5*||AU*S*V'-B||_F^2 + lambda*||S||_*
	%% sVD for AU
	AU = A*U;
	[vAU, sAU, ~] = svd(AU'*AU);
	sssAU = sqrt(diag(sAU));
	sAU = diag(sssAU);
	uAU = AU*vAU*diag(1./sssAU);

	%% transform problem to 0.5*||sAU*vAU'*S - uAU'*B*V||_F^2 + lambda*||S||_*
	Bbar = uAU'*B*V;
	Abar = sAU*vAU';
	AbarAbar = Abar'*Abar;
	AbarTB = Abar'*Bbar;
	h = max(svd(AbarAbar));
	%% solve the problem by proximal gradient

	for inner=1:4
		grad = AbarAbar*S - AbarTB;
		newS = S - grad/h;
		[u s v] = svd(newS);
		sss = diag(s);
		S = u*diag(max(sss-lambda/h,0))*v';
	end
	totaltime = totaltime + cputime - timebegin;
	timelist(i) = totaltime;

	[uuuu ssss vvvv] = svd(S);
	ddssss = diag(ssss);
	U = U*uuuu(:,ddssss>1e-10)*ssss(ddssss>1e-10, ddssss>1e-10);
	V = V*vvvv(:,ddssss>1e-10);

	obj = sum(diag(ssss))*lambda + norm((A*U)*V' -B, 'fro')^2/2;

	obj_vals(i)=obj;

	test_err = sqrt(norm(At*U*V'-Bt,'fro')^2/numel(Bt));
	test_rmses(i) = test_err;
	fprintf('Iter %g time %g obj %g testrmse %g\n', iter, totaltime, obj, test_err);
end

