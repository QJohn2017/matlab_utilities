function m=nmin(varargin)
% nmin gives the minumum of sevaral variables

% 2017, Alberto Comin

m=varargin{1};
for n=2:length(varargin)
  m = min(m, varargin{n});
end

end
