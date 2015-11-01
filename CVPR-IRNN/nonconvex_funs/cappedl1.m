function x = cappedl1(x,gamma,lambda)
x(x > gamma*lambda) = gamma*lambda;
x(x < -gamma*lambda) = -gamma*lambda;

