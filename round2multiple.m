function ys = round2multiple(xs, n)
%ROUND2MULTIPLE rounds to the closest multiple of a unit
%
% INPUT:
% xs : number(s) to round (scalar of multidimensional array)
% n : unit (scaler)
%
% OUTPUT:
% ys : the rounded number(s) (same size as x)
%
% EXAMPLE:
% round2multiple(15.3, 2) % round to the closest even number
%   (ans = 16)
% y = round2multiple([11, 45, 78], 10) % round to the closest multiple of 10
%   (ans = [10, 50, 80])
% 2017, Alberto Comin

ys = bsxfun(@(x, n) round(x/n)*n, xs, n);
end
