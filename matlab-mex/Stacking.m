% Rvector: two vectors of orthogonal angles
% idx: indices of k, n, and c to store the the vectors idx = [k n c]
function Sinogram = Stacking(Rvector, idx, SinoIn)
    SinoIn(:,idx(1)+1+idx(4)*idx(2),idx(3)+1)=Rvector(:,1);
    SinoIn(:,idx(1)+(2*idx(4)+1)+idx(4)*idx(2),idx(3)+1)=flipud(Rvector(:,2));
    Sinogram = SinoIn;
end