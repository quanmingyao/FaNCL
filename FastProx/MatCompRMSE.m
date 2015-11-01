function [ RMSE ] = MatCompRMSE( U, V, S, row, col, gndtruth )

U = U*S;
predict = partXY(U', V', row, col, length(gndtruth))';

RMSE = sqrt(sumsqr(predict - gndtruth)/length(gndtruth));

end

