function [W1, W2] = errorbackpropschool(X, t)
    
    % Initialize weights and biases
    hidden = 3;
    W1 = rand(hidden, size(X,2));
    W2 = rand(1, hidden + 1);
    % Loop T times
    T = 50000;
    eta = 0.01;
    alpha = 0.012;
    h = @(W) [ones(1,10); tanh(W*X')];
    y = @(W1, W2) sigmf(W2*h(W1), [1 0]);
    for k = 1:T
        hval = h(W1);
        yval = y(W1, W2);
        for i = 1:hidden
            % grad1 has no i = 0 term, it starts at i = 1, j = 0.
            % This means we start at second term of W2, as it starts at i = 0.
            grad1(i,:) = W2(i+1)*((yval - t').*(1-hval(i+1,:).^2))*X + alpha*W1(i,:);
        end
        grad2 = (yval - t')*hval' + alpha*W2;
        W1 = W1 - eta*grad1;
        W2 = W2 - eta*grad2;
    end
end