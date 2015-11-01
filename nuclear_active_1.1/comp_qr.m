function [Q, R] = comp_qr(Y)

YY = Y'*Y;
R = chol(YY);
Q = Y*inv(R);
