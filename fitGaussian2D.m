function [fitRes, fitFunc, fitOutput] = ...
  fitGaussian2D(img, x, y, options)
% FITGAUSSIAN2D fits a bidimensional Gaussian surface
%
% It automatically determines the starting fit parameters and uses the
% analytical gradient for more accuracy.
%
% Examples:
% fitRes = fitGaussian2D(img)
% fitRes = fitGaussian2D(img, x, y)
%
% Input:
% img: 2D image
%
% Optional Input:
% x: array with x values
% y: array with y values
% options: cell array with options for lsqcurvefit
%
% Output:
% fitRes: structure with best fit paramters
% fitFunc: handle of the fit functions
% fitOutput: structure with results of fitting routine
%
% Notes:
%   x and y can be either vectors or matrixes of the same size as img (e.g.
%   obtained using meshgrid or ndgrid)
%
% Requires:
% Optimization Toolbox (for lsqcurvefit)

% 2017-2018 Alberto Comin

if ~exist('x', 'var') || isempty(x), x = 1:size(img,2); end
if ~exist('y', 'var') || isempty(y), y = 1:size(img,1); end

if ~exist('options', 'var') || isempty(options)
  fitOptions = optimoptions('lsqcurvefit', 'Jacobian', 'off');
else
  fitOptions = optimoptions('lsqcurvefit', options{:});
end

% transform data to be compatible with lsqcurvefit
if isvector(x) && isvector(y)
  x = reshape(x, 1, []);
  y = reshape(y, [], 1);
  [Y,X] = ndgrid(y, x);
else
  X = x;
  Y = y;
  x = X(1, :);
  y = Y(:, 1);
end

% convert img to single column as required by lsqcurvefit
img1d  = img(:);

% estimate offset and amplitude
offset = min(img1d);
amplitude = max(img1d)-min(img1d);

% normalize the data
img1d = (img1d - offset) / amplitude;

% estimate center of gaussian                               
[~,imax] = max(img(:));
[iy0, ix0] = ind2sub(size(img), imax);

% estimate standard deviation
sx = std(X(:), max(0,img1d));
sy = std(Y(:), max(0,img1d));

% initial parameters
p0 = [offset, ...           % offset
      amplitude, ...        % amplitude
      x(ix0), ...               % x0
      y(iy0), ...               % y0
      sx, ...               % std_x
      sy];                  % std_y
% lower bounds
pmin = [ -0.1, ... % offset
         0.5, ... % amplitude
         x(ix0) - sx, ...         % x0
         y(iy0) - sy, ...         % y0
         0.5 * sx, ...        % std_x
         0.5 * sy];           % std_y
% upper bounds
pmax = [ 0.1 ...  % offset
         1.5, ...  % amplitude
         x(ix0) + sx, ...          % x0
         y(iy0) + sy, ...          % y0
         1.5 * sx, ...         % std_x
         1.5 * sy];            % std_y   

[p, resnorm, residual, exitflag, output, lambda, jacobian] = ...
  lsqcurvefit( ...
  @(p, xy) gaussian2d(p, xy(:,1), xy(:,2)), ...
  p0, [X(:), Y(:)], img1d, pmin, pmax, fitOptions);

% rescale fit results
p(1) = p(1) * amplitude + offset;
p(2) = p(2) * amplitude;

% export fitting function as output
fitFunc = @(x, y) gaussian2d(p, x, y);

std2FWHM = 2*sqrt(2*log(2)); % standard deviation to FWHM

fitRes = struct( ...
  'offset', p(1),            'amplitude', p(2), ...
  'x0',     p(3),            'y0',        p(4), ...
  'sx',     p(5),            'sy',        p(6), ...
  'FWHM_x', p(5)*std2FWHM,   'FWHM_y',    p(6)*std2FWHM);

fitOutput = struct( ...
  'resnorm',  resnorm,       'residual', residual, ...
  'exitflag', exitflag,      'output',   output, ...
  'lambda',   lambda,        'jacobian', jacobian);

if nargout>1
  fitOutput.residual = reshape(residual, size(img));
  fitOutput.jacobian = reshape(full(jacobian), size(img,1), size(img,2), length(p));
end
end

function [z, J] = gaussian2d(p, x, y)
% gaussian2d calculates a bidimensional gaussian and its Jacobian
%
% Input:
% p: an array containing the fit parameters
% xy: a Nx2 matrix with columns of x and y data
%
% Output:
% z: the calculated guassian surface
% J: the calculated Jacobian
%
% Notes:
% the width parameter is the standard deviation

z = p(1) + p(2) * ...
    exp(-(x - p(3)).^2 / 2 / p(5).^2 ...
        -(y - p(4)).^2 / 2 / p(6).^2);
          
if nargout > 1
  J = zeros(length(z), length(p));
  J(:,1) = ones(size(z));
  J(:,2) = (z - p(1)) ./ p(2);
  J(:,3) = (x - p(3)) / p(5)^2 .* (z - p(1));
  J(:,4) = (y - p(4)) / p(6)^2 .* (z - p(1));
  J(:,5) = (x - p(3)).^2 ./ p(5).^3 .* (z - p(1));
  J(:,6) = (y - p(4)).^2 ./ p(6).^3 .* (z - p(1));
end             
end

