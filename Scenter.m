function [Sc cf] = Scenter(S)
[nx ny] = size(S);
R = repmat((1:nx)',[1 ny]);
c = vecWMfinal(R,S);
np = vecWMfinal((1:nx)',ones(nx,1));
dc = c-np;
loc = dc>=0;
ndown = max(dc(loc));
nup = min(dc(~loc));
if isempty(ndown)
ndown = 0;
end
if isempty(nup)
nup = 0;
end
nt = nx + abs(nup) + abs(ndown);
Sc = zeros(nt,ny);
ni = 1 + abs(ndown) - dc;
cf = abs(ndown) + np;
for i = 1:ny
    Sc(ni(i):ni(i)+nx-1,i) = S(:,i);
end
end
