function [U S V] = randomsvd(A,uu,vv, m,n,k, INIT, maxit)

istrans = 0;
if numel(INIT) == 0
	Omega = randn(n, k);
else
	tt = min(k,size(INIT,2));
	Omega = INIT(:,1:tt);
	if tt < k
		Omega = [Omega randn(n,k-tt)];
	end
end
Y = A*Omega+uu*(vv'*Omega);
Atrans = A';
[Q,~] = comp_qr(Y);
%[Q, ~] = qr(Y,0);
%for i=1:40
for i=1:maxit
	BB = Atrans*Q + vv*(uu'*Q);
	Y = A*BB + uu*(vv'*BB);
	[Q, ~] = comp_qr(Y);
%	[Q,~] = qr(Y,0);
end
[Q r] = qr(Y,0);
B = Q'*A+(Q'*uu)*vv';
[u S V] = svd(B,'econ');
U = Q*u;
