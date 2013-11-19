function M = rotflip(I)
M = I(:,end:-1:1)';
end