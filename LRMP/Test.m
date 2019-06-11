clear;clc;

img = imread('Lena512.bmp');
img = double(img);
img = img/255;
img = img - mean(img(:));

Known = (rand(size(img)) > 0.5);
data = Known.*img;
[~, ~, data] = find(sparse(data));

opts.verbosity = 1;

[U, Theta, V, numiter ] = OR1MP(512, 512, 200, find(Known ~= 0), data );

rImg = U*diag(Theta)*V';

sqrt(sum((img(:) - rImg(:)).^2)/numel(img))