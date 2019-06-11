function [rst] = myPSNR(in, gnd, max)

rmse = in(:) - gnd(:);
rmse = mean(rmse.^2);
rmse = sqrt(rmse);

rst = 20*log10(max/rmse);

end