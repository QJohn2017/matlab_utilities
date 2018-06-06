function [xi, yi] = sampleCurve(domain, func, noPoints, varargin)
%SAMPLECURVE samples a function at equidistant points
% Useful for representing a mathematical function using a reduced number of
% points, the idea is to have more points in the regions where the function
% varies more steeply.
%
% [xi, yi] = sampleCurve(func, noPoints, varargout)
%
% Inputs:
% func: function handle or array of function values
% domain: array [x1...xn] of domain values or pair [xMin, xMax], in the latter
%         case it is possible to specify the number of domainPoints
% noPoints: number of sampling points
% 
% Optional inputs:
% noDomainPoints: max no. of points used for initial function evaluation
%
% Outputs:
% xi: sampling points
% yi: function values at sampling points

% Alberto Comin, 2018

% default values
noDomainPoints = 1000;

% preprocess key value arguments
for i = 1:2:numel(varargin)
  switch varargin{i}
    case 'noDomainPoints'
      noDomainPoints = varargin{i+1};
    otherwise
      error('SampleCurve:argChk', 'unsupported argument ''%s''', varargin{i});
  end
end

if numel(domain)==2
  x = linspace(min(domain), max(domain), noDomainPoints)';
else
  x = sort(domain);
end

if isa(func,'function_handle')
  y = func(x);
else
  y = func;
end

% for the case func transposes its input, ensure shapes are the same
y = reshape(y, size(x));

% rescale function, otherwise one dimension risks to dominate
scaling = (max(x) - min(x)) / (max(y) - min(y));

% calculate cumulative distance along the curve
dl = cumsum(abs(hypot(gradient(x), scaling * gradient(y))));

li = linspace(dl(1), dl(end), noPoints)';
xi = interp1(dl, x, li);

if isa(func,'function_handle')
  yi = func(xi);
else
  yi = interp1(x, y, xi);
end

end
