function [ w ] = getWeight(s, theta, lambda, regType )

switch(regType)
    case 1 % CAP
        w = cappedl1_sg(s, theta, lambda);
    case 2 % Logrithm
        w = logarithm_sg(s, theta, lambda);
    case 3 % TNN
        w = lambda*ones(size(s));
        w(1:theta) = 0;
    otherwise
        disp('not support!\n');
end

end