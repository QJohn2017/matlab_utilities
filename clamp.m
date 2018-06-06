function y = clamp(x, minVal, maxVal)
%CLAMP replaces elements outside boundaries with minVal or maxVal
%
% syntax:
%   y = clamp(x, minVal, maxVal)
% where:
%   x is any array
%   minVal and maxVal are either scalar or arrays compatible with x
%   according to the brooadcasting rules of bsxfun
%
% example:
% >> clamp([-2,-1,1,2,3],1,2)
% ans =
%      1     1     1     2     2
%
% >> clamp([[1, 2, 3; ...
%            4, 5, 6; ...
%            7, 8, 9]], [1,1,1], [1,5,9])
% ans =
%      1     2     3
%      1     5     6
%      1     5     9

% 2018 Alberto Comin

y = bsxfun(@min, bsxfun(@max, x, minVal), maxVal);

end

