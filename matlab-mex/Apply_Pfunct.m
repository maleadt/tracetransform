%% Applying the P-functionals to the sinogram
% If the P-functionals are not based on orthonormal polynomials the 
% P-functionals are immediately applied to the sinogram. If we use
% the Hermite functionals (orthonormal) we have to at first align 
% the columns of the sinogram (Sinogram_center), getting the Nearest
% Orthonormal Sinogram via SVD, and then apply the orthonormal functionals.
function CircusF = Apply_Pfunct(Stack_sinogram, Code_Pfunct, flag)
    [~, N_angle Numb_T]=size(Stack_sinogram);
    Numb_P = length(Code_Pfunct);
    CircusF=zeros(N_angle,Numb_P*Numb_T);

    for i=1:Numb_T
            S = Stack_sinogram(:,:,i);
            [Sc cf] = Sinogram_center(S);   %Alignment of the sinograms, that is changing from x domain to r
        if flag == 1
            [U,Sk,V]=svd(Sc);               %Applying the SVD 
            sz_Sk = size(Sk);
            Sk = eye(sz_Sk);
            Sc = U*Sk*V';                  %Nearest orthonormal sinogram
        end
        
        for j = 1:Numb_P
            if Code_Pfunct(j) > 3           % Othornormal functionals
                NOS = Sc;%Sc(cf:end,:);    %Obtaining the positive domain
            else
                NOS = S;
            end
            CircusF(:,Numb_P*(i-1)+j)=functP(NOS,Code_Pfunct(j),cf); %Applying the P functionals
        end
    end
end

function [Sc cf] = Sinogram_center(S)
    [nx ny] = size(S);
    R = repmat((1:nx)',[1 ny]);
    c = vecWMfinal(R,S);
    np = vecWMfinal((1:nx)', ones(nx,1));
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

function CircusF = functP(NOS,code,cf)
    rng= 10;
    if code == 1
        CircusF = mean(abs(diff(NOS))); 
    elseif code == 2
        CircusF = vecWMfinal(NOS,abs(NOS));
    elseif code == 3
        L = size(NOS,1);
        CircusF = trapz(linspace(-1,1,L),abs(fft(NOS)/L).^4);
    else
        [m n]=size(NOS);
        [P ~] = hermite_wpol(code-3,cf,m,rng); % Hermite orthonormal polynomials
        Pmat=repmat(P',1,n);
        CircusF = trapz(Pmat.*NOS);
    end
end
