%sigmoid = @(a) 1 ./ (1 + exp(-a));
function s = sigmoid(a)
    s = 1 ./ (1 + exp(-a));
end