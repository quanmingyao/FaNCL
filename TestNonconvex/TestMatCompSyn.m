clear; clc;

M = 2000;
N = M;
K = 5;

U = randn(M, K);
V = randn(N, K);
O = U*V';
G = randn(M, N);
D = O + 0.1*G;
ratio = 2*M*K*log(M)/(M*N);
traD = D.*(rand(M, N) < ratio);
traD = sparse(traD);
valD = (rand(M, N) < ratio);
tstD = (rand(M, N) < ratio);
tstD = O.*tstD;
tstD = sparse(tstD);

para.maxIter = 5000;
para.tol = 1e-3;
para.tau = 1.01;
para.maxTime = 10000;
para.speedup = 1;
para.decay = 0.9;

[trow, tcol, tval] = find(tstD);
para.test.row = trow;
para.test.col = tcol;
para.test.data = tval;
para.test.m = size(tstD, 1);
para.test.n = size(tstD, 2);

clear trow tcol tval tstD;

lambdaMax = topksvd(traD, 1, 5);
% gridLambda = lambdaMax*(0.6).^(5:12);
% 
% gridRMSE = zeros(size(gridLambda));
% for g = 1:length(gridLambda)
%     lambda = gridLambda(g);
%     
%     % [U, S, V, out{1}] = APGMatComp(full(traD), lambda, para );
%     [U, S, V] = AccSoftImpute(traD, lambda, para );
%     X = U*S*V';
%     
%     gridRMSE(g) = sqrt(norm((X - O).*valD, 'fro')^2/nnz(valD));
% 
%     clear X U S V;
% end
% 
% [~, lambda] = min(gridRMSE);
% lambda = gridLambda(lambda);
% 
% method = 1;
% t = tic;
% [U, S, V, out{1,method}] = APGMatComp(full(traD), lambda, para );
% Time(1,method) = toc(t);
% X = U*S*V';
% 
% NMSE(1,method) = norm((X - O), 'fro')/norm(X, 'fro');
% 
% t = tic;
% para.speedup = 1;
% para.exact = 1;
% [ U, S, V, out{2,method} ] = SoftImpute( traD, lambda, para);
% Time(2,method) = toc(t);
% X = U*S*V';
% 
% NMSE(2,method) = norm((X - O), 'fro')/norm(O, 'fro');
% 
% t = tic;
% [ U, S, V, out{3,method} ] = mc_alt( traD, lambda, para );
% Time(3,method) = toc(t);
% X = U*S*V';
% 
% NMSE(3,method) = norm((X - O), 'fro')/norm(X, 'fro');
% 
% t = tic;
% para.speedup = 1;
% para.regType = 1;
% [U, S, V, out{4,method}] = FastMatComp( traD, lambda, 1e+8, para );
% Time(4,method) = toc(t);
% X = U*S*V';
% 
% NMSE(4,method) = norm((X - O), 'fro')/norm(O, 'fro');
% 
% clear X U S V;

%% Fixed Rank
% method = 2;
% 
% estRank = 6;
% t = tic;
% [U, S, V, out{1, method} ] = SoftImputeALS( traD, 0, estRank, para );
% Time(1, method) = toc(t);
% X = U*S*V';
% 
% NMSE(1, method) = norm((X - O), 'fro')/norm(O, 'fro');
% 
% estRank = 100;
% t = tic;
% [U, S, V, out{2, method} ] = EOR1MP( traD, estRank, para );
% Time(2, method) = toc(t);
% X = U*S*V';
% 
% NMSE(2, method) = norm((X - O), 'fro')/norm(O, 'fro');

%%
for i = 3:5
    para.regType = i - 2;
    % lambda
    gridLambda = lambdaMax*(0.6).^(1:10);
    gridRMSE = zeros(size(gridLambda));
    for g = 1:length(gridLambda)
        lambda = gridLambda(g);
        para.speedup = 1;
        para.maxR = 2*K;

        switch(para.regType)
            case 1
                theta = lambda*2;
            case 2
                gridLambda(g) = gridLambda(g)*10;
                lambda = gridLambda(g);
                theta = sqrt(lambda);
            case 3
                lambda = 5.519574624223880;
                theta = 5;
        end

        [U, S, V] = FastMatComp( traD, lambda, theta, para );
        X = U*S*V';

        gridRMSE(g) = sqrt(norm((X - O).*valD, 'fro')^2/nnz(valD));

        clear X U S V;
        
        if(g > 3 && gridRMSE(g) >= gridRMSE(g-1))
            gridRMSE = gridRMSE(1:g);
            break;
        end
    end

    [~, lambda] = min(gridRMSE);
    lambda = gridLambda(lambda);
    
    % theta
    gridTheta = 2.^(-1:1);
    gridRMSE = zeros(size(gridTheta));
    for g = 1:length(gridTheta)
        para.speedup = 1;
        theta = gridTheta(g);

        switch(para.regType)
            case 1
                theta = lambda*theta;
            case 2
                theta = theta*sqrt(lambda);
            case 3
                theta = 5;
        end

        [U, S, V] = FastMatComp( traD, lambda, theta, para );
        X = U*S*V';

        gridRMSE(g) = sqrt(norm((X - O).*valD, 'fro')^2/nnz(valD));

        clear X U S V;
        
%          if(g > 1 && gridRMSE(g) >= gridRMSE(g-1))
%             gridRMSE = gridRMSE(1:g);
%             break;
%         end
    end

    [~, theta] = min(gridRMSE);
    theta = gridTheta(theta);
    
    clear gridTheta gridRMSE g gridLambda;
    
    switch(para.regType)
        case 1
            theta = lambda*theta;
        case 2
            lambda = lambda;
            theta = theta*sqrt(lambda);
        case 3
            theta = 5;
    end
    
%     t = tic;
%     para.speedup = 0;
%     [U, S, V, out{4,i}] = FastMatComp( traD, lambda, theta, para );
%     Time(4,i) = toc(t);
%     X = U*S*V';
%     NMSE(4,i) = norm((X - O), 'fro')/norm(O, 'fro');
%     clear X U S V;
    
    t = tic;
    para.speedup = 1;
    [U, S, V, out{5,i} ] = FastMatComp( traD, lambda, theta, para );
    Time(5,i) = toc(t);
    X = U*S*V';
    NMSE(5,i) = norm((X - O), 'fro')/norm(O, 'fro');
    clear X U S V;

%     t = tic;
%     [U, S, V, out{6,i}] = IRNNMatComp(full(traD), lambda, theta, para);
%     Time(6,i) = toc(t);
%     X = U*S*V';
%     NMSE(6,i) = norm((X - O), 'fro')/norm(O, 'fro');
%     clear X U S V;
% 
%     t = tic;
%     [U, S, V, out{7,i}] = GISTMatComp(full(traD), lambda, theta, para);
%     Time(7,i) = toc(t);
%     X = U*S*V';
%     NMSE(7,i) = norm((X - O), 'fro')/norm(O, 'fro');
%     clear X U S V;
end

clear D O U V t G traD tstD valD;
 
save(strcat(num2str(M), '-matcomp-1'));
