clear; clc;

M = 50;
N = 1000;
K = 1000;

S = sprandn(N, K, 0.01);
D = rand(M, N);
[row, col, data] = find(S);

t1 = tic;
DS1 = D*S;
t1 = toc(t1);

% sparse matrix product
t2 = tic;
DS2 = zeros(M, K);
for i = 1:length(data)
    di = data(i);
    ri = row(i);
    ci = col(i);
    
    DS2(:, ci) = DS2(:, ci) + di*D(:, ri);
    % DS2(ri, :) = DS2(ri, :) + di*D(ci,:);
end
t2 = toc(t2);