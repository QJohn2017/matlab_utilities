function h = pcol(varargin)
%PCOL is an improved pcolor which works with tensors
%
% It automatically removes the grid using "shading flat",
% squeezes the inputs to remove eventual singleton dimensions
% and add axes labels:
%
% Syntax:
%   pcol(C)
%   pcol(X,Y,C)
%   pcol(axes_handles,...)
%   h = pcolor(...)
%
% Example:
%   >> x = -10 : 0.1 : 9.9; % horizontal array 
%   >> y = shiftdim(x, 1); % vertical array
%   >> z = shiftdim(x, -1); % array along 3rd direction
%   >> c = exp(- (x.^2 + y.^2/8 + z/4));
% 
%   >> figure(2)
%   >> subplot(3,1,1)
%   >> pcol(x,y,c(:,:,100))
%   >> subplot(3,1,2)
%   >> pcol(z,x,c(:,100,:))
%   >> subplot(3,1,3)
%   >> pcol(z,y,c(100,:,:))
%
%   >> whos x y z c
%     Name        Size                    Bytes  Class     Attributes
% 
%     c         200x200x200            64000000  double              
%     x           1x200                    1600  double              
%     y         200x1                      1600  double              
%     z           1x1x200                  1600  double              

% 2018 Alberto Comin

if nargin == 1
  h = gca;
  c = squeeze(varargin{1});
  x = 1 : size(c, 2);
  y = 1 : size(c, 1);
  xlab = 'x';
  ylab = 'y';
  clab = inputname(1);
elseif nargin == 3
  h = gca;
  x = squeeze(varargin{1});
  y = squeeze(varargin{2});
  c = squeeze(varargin{3});
  xlab = inputname(1);
  ylab = inputname(2);
  clab = inputname(3);
elseif nargin == 4
  h = varargin{1};
  x = squeeze(varargin{2});
  y = squeeze(varargin{3});
  c = squeeze(varargin{4});
  xlab = inputname(2);
  ylab = inputname(3);
  clab = inputname(4);
else
  error('pcol:argChk', 'wrong number of arguments');
end

pcolor(h, x, y, c);
shading(h, 'flat');

xlabel(xlab);
ylabel(ylab);

hbar = colorbar;
hbar.Label.String =  clab;

if nargout == 0, clear h; end
end

