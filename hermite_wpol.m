%Obtains the Hermite weighted functions
%   n = Order of the polynomial
%   cf = Center of symmetry
%   N = Number of samples
%   range = Range in the continuous domain
function [psi xc] = hermite_wpol(n, cf, N, range)

    Nneg = cf ;
    Npos = N - cf + 1;
    xneg = linspace(-range, 0,Nneg);
    xpos = linspace(0, range,Npos);
    xc = union(xneg,xpos);

    psi0 = (1/(pi)^.25)*exp(-xc.^2/2);
    psi1 = 2*((2*sqrt(pi))^-.5)*xc.*exp(-xc.^2/2);
    H0 = 1;
    H1 = 2*xc;

    if n == 0
        psi = psi0;
    elseif n == 1
        psi = psi1;
    else
        for i = 1:n-1
            H2 = 2*xc.*H1 - 2*i*H0;
            H0 = H1;
            H1 = H2;
        end
        psi = (2^(i+1)*factorial(i+1)*sqrt(pi))^(-0.5)*exp(-xc.^2/2).*H2;
    end
end