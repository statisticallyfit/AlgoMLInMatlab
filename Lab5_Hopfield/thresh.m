% ARUGMENT: vector (a) of activations: a_i = sum(w_ij * x_j)
% OUTPUT: Return vector of inputs (xs) that must be 1 or -1 based on sign
% of a (if a_i >= 0 then x_i = 1 else if a_i < 0 then x_i = -1)
function f = thresh(a)

    I = find(a >= 0);
    f = -1*ones(length(a), 1);
    f(I) = 1;
end