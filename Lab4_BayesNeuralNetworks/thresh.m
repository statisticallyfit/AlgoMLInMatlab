% ARUGMENT: vector (a) of activations: a_i = sum(w_ij * x_j)
% OUTPUT: Return vector of inputs (xs) that must be 1 or -1 based on sign
% of a (if a_i >= 0 then x_i = 1 else if a_i < 0 then x_i = -1)
function threshed = thresh(a)

    %I = find(a >= 0);
%    f = -1*ones(length(a), 1);
%    f(I) = 1;

    % Doing thresh while preserving matrix shape, where a is a matrix
    [N, I] = size(a);
    iPos = find(a >= 0);
    negOnes = -1 * ones(N, I);
    negOnes(iPos) = 1;
    
    threshed = negOnes;
end