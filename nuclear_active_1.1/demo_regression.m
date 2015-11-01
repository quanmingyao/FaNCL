load stock_subset.mat;
lambda = 10;

[U1,  V1, objlist_alt, testlist_alt, timelist_alt] = regression_alt(A, B, At, Bt, lambda, 100 , 10);
