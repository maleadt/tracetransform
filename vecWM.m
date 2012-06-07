function c = vecWM(I)
%% Finding the weighted median of a set of column vectors
[N M] = size(I);
D = repmat((0:N-1)',[1 M]);
c = vecWMed(D,I);


