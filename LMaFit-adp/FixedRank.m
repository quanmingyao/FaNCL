function [U, S, V, out ] = FixedRank( D, rnk, para )

[m, n] = size(D);
Known = find(D(:) ~= 0);
data = D(Known);
para.est_rank = 0;
para.print = 2;
para.DoQR = 1;

clear D;

[U, V, out] = lmafit_mc_adp(m, n, rnk, Known', data, para);

[U, Qu] = qr(U , 0);
[V, Qv] = qr(V', 0);
S = Qu*Qv';
[P, S, Q] = svd(S, 'econ');
U = U*P;
V = V*Q;

% temp = (1:length(out.RMSE))';
% temp = 1 ./ temp;
% out.RMSE = out.RMSE(end) + temp - min(temp);

% idx = out.RMSE < 1;
% out.RMSE = out.RMSE(idx);
% out.obj  = out.obj (idx);
% out.Time = out.Time(idx);
% out.Time = out.Time - min(out.Time) + 0.01;

end

