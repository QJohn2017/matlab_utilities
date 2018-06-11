function [xcm, ycm] = centOfMass2d(...
  img, ... % image to be fitted
  x, ...  % array of x coordinates
  y, ...  % array of y coordinates
  mask ... % limit the fit to a mask or a circular region
  )
%centOfMass2d calculates center of mass of an image
% 
% USAGE:
%   g  = centOfMass2d( img )
%   [gx, gy] = centOfMass2d( img)
%   g = centOfMass2d(img, x, y)
%   g = centOfMass2d(img, x, y, mask)
%   g = centOfMass2d(img, x, y, rho)
%
% INPUT:     
%   img: image to be fitted
%   x: array or matrix of x coordinates (optional)
%   y: array or matrix of y coordinates (optional)
%   mask: boolean mask or a scalar 'rho'. In the latter case the algorithm
%         creates a circular mask of radius 'rho' centered at the peak.
%
% OUTPUT
%   [xcm, ycm]: coordinates of center of mass
%
% NOTES:
% - x and y can be specified either as 1d arrays or as 2d arrays obtained
%   using ndgrid or meshgrid
% - the function can be called with either one or two outputs: 
%   in the second case, it returns an array of two elements
 
% 2015 Alberto Comin, LMU Muenchen

%% initialize variables

if ~exist('x', 'var') || isempty(x), x = 1:size(img,2); end
if ~exist('y', 'var') || isempty(y), y = 1:size(img,1); end

if isvector(x) && isvector(y)
  assert(length(x)==size(img,2) & length(y)==size(img,1), ...
    'centOfMass2d:argChk', ...
    'x and y have not compatible size with img');
  [Y,X] = ndgrid(y, x);
else
  assert(all(size(x)==size(img)) & all(size(y)==size(img)), ...
    'centOfMass2d:argChk', ...
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

%% calculating center of mass

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

sumOfWeigths = sum(sum(img .* mask));
xcm = sum(sum(X .* img .* mask)) ./ sumOfWeigths;
ycm = sum(sum(Y .* img .* mask)) ./ sumOfWeigths;

% if called with only one output argument return array
if (nargout < 2)
   xcm(2) = ycm;
end
end

