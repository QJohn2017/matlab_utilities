function m=nmax(varargin)
% nmax gives the maximum of sevaral variables

% 2017, Alberto Comin

m=varargin{1};
for n=2:length(varargin)
  m = max(m, varargin{n});
end

end
