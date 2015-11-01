function [U S V] = randomsvd_regression(uu,vv, uu1, vv1, uu2, vv2, m,n,k, INIT, maxit)

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
Y = uu*(vv'*Omega) + uu1*(vv1'*Omega)+uu2*(vv2'*Omega);
[Q R] = qr(Y,0);
for i=1:40
%for i=1:5
	if i>= maxit
		break;
	end
	BB = vv*(uu'*Q) + vv1*(uu1'*Q) + vv2*(uu2'*Q);
	Y = uu*(vv'*BB) + uu1*(vv1'*BB)+ uu2*(vv2'*BB);
	Qold = Q;
	[Q,R] = qr(Y,0);
	angle = min(svd(Q'*Qold));
	if angle > 1-5e-2
		break
	end
end
[Q r] = qr(Y,0);
B = (Q'*uu)*vv' + (Q'*uu1)*vv1' + (Q'*uu2)*vv2';
[u S V] = svd(B,'econ');
U = Q*u;
