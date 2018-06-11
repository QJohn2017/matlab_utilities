function [xsk, ysk] = skewness2d(...
  img, ... % image to be fitted
  x, ...   % array of x coordinates
  y, ...   % array of y coordinates
  mask ... % limit the fit to a mask or a circular region
  )
%SKEWNESS2d calculates skewness of an image
% 
% USAGE:
%   g  = skewness2d( img )
%   [gx, gy] = skewness2d( img)
%   g = skewness2d(img, x, y)
%   g = skewness2d(img, x, y, mask)
%   g = skewness2d(img, x, y, rho)
%
% INPUT:     
%   img: image to be fitted
%   x: array or matrix of x coordinates (optional)
%   y: array or matrix of y coordinates (optional)
%   mask: boolean mask or a scalar 'rho'. In the latter case the algorithm
%         creates a circular mask of radius 'rho' centered at the peak.
%
% OUTPUT
%   [xcm, ycm]: skewness2d along the x and y direction
%
% NOTES:
% - x and y can be specified either as 1d arrays or as 2d arrays obtained
%   using ndgrid or meshgrid
% - the function can be called with either one or two outputs: 
%   in the second case, it returns an array of two elements
 
% 2017 Alberto Comin, LMU Muenchen

%% initialize variables

if ~exist('x', 'var') || isempty(x), x = 1:size(img,2); end
if ~exist('y', 'var') || isempty(y), y = 1:size(img,1); end

if isvector(x) && isvector(y)
  assert(length(x)==size(img,2) & length(y)==size(img,1), ...
    'skewness2d:argChk', ...
    'x and y have not compatible size with img');
  [Y,X] = ndgrid(y, x);
else
  assert(all(size(x)==size(img)) & all(size(y)==size(img)), ...
    'skewness2d:argChk', ...
    'x and y have not compatible size with img');
  X = x;
  Y = y;
end


if ~exist('mask', 'var')
   mask = ones(size(img));
end

%% background subtraction
bg = min(img(:)); % quick solution, can be improved
img = img - bg;

%% calculating skewness

% convert the mask to a boolean matrix, if it was specified as a radius
if isscalar(mask)
  % find the peak position
  [~, peakind] = max(img(:));
  [iym, ixm] = ind2sub(size(img), peakind);
  xm = X(1, ixm);
  ym = Y(iym, 1);
  % consider the central part of the image
  mask = hypot(X-xm, Y-ym) < mask;
end

% apply mask
img = img .* mask;

% calculating the middle of the image
sumOfWeigths = sum(sum(img));
xcm = sum(sum(X .* img)) ./ sumOfWeigths;
ycm = sum(sum(Y .* img)) ./ sumOfWeigths;

% calcuating the standard deviation
xstd = sqrt(sum(sum((X-xcm).^2 .* img)) ./ sumOfWeigths);
ystd = sqrt(sum(sum((Y-ycm).^2 .* img)) ./ sumOfWeigths);

% calculating skewness
xsk = sum(sum(((X-xcm)/xstd).^3 .* img)) ./ sumOfWeigths;
ysk = sum(sum(((Y-ycm)/ystd).^3 .* img)) ./ sumOfWeigths;

% if called with only one output argument return array
if (nargout < 2)
   xsk(2) = ysk;
end
end

