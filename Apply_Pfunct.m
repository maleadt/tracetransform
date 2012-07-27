function CircusF = Apply_Pfunct(Stack_sinogram,Code_Pfunct,flag)
%% Applying the P-functionals to the sinogram
% If the P-functionals are not based on orthonormal polynomials the 
% P-functionals are immediately applied to the sinogram. If we use
% the Hermite functionals (orthonormal) we have to at first align 
% the columns of the sinogram (Sinogram_center), getting the Nearest
% Orthonormal Sinogram via SVD, and then apply the orthonormal functionals.

[~, N_angle Numb_T]=size(Stack_sinogram);
Numb_P = length(Code_Pfunct);
CircusF=zeros(N_angle,Numb_P*Numb_T);

for i=1:Numb_T
    %Alignment of the sinograms, that is changing from x domain to r
    % domain transform: pixels reordered (somewhat similar)
    S = Stack_sinogram(:,:,i);  % S = sinogram matrix
    [Sc cf] = Sinogram_center(S);   % Sc = aligned sinogram, cf = scalar???

    if flag == 1
        %% Apply SVD to find NOS
        [U,Sk,V]=svd(Sc);   % Sc= 140x120 -> U = 140x140, Sk=140x120, V=120x120
        sz_Sk = size(Sk);   % == size(Sc) == 140x120
        Sk = eye(sz_Sk);    %% replacing with identidy matrix?
        Sc = U*Sk*V';
        % Sc is NOS
    end
    
    for j = 1:Numb_P
        if Code_Pfunct(j) > 3           % Othornormal functionals
            NOS = Sc;%Sc(cf:end,:);    %Obtaining the positive domain
        else
            NOS = S;                    % use aligned (which contains the same, just with some additional zeros)
        end
        CircusF(:,Numb_P*(i-1)+j)=functP(NOS,Code_Pfunct(j),cf); %Applying the P functionals
    end
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