% ERROR BACKPROP for 2-layer network with 3 hidden units

% H = num hidden units
% X = N x I data matrix
% t = N x 1 target vector
function [Whidden, Wout] = errorBackpropTwoLayer(X, H, t)
    
    % (1) INITIALIZE weights and biases
    %rng('default') % keeping the same seed makes decision boundary LINEAR - bad
    
    [N, I] = size(X);
    
    %H = 3; % num hidden units
    Whidden = rand(H, I);
    Wout = rand(1, H + 1);
    
    T = 50000;
    eta = 0.01;
    alpha = 0.012;
    
    % Activity rule
    aHidden = @(wH, X) wH * X'; % hidden layer activation
    hFunc = @(wH, X, N) [ones(1,N); tanh( aHidden(wH, X) )]; % adds a row of ones for the W0 biases (I think)
    
    aOutput = @(wH, wO, X, N) wO * hFunc(wH, X, N); % output layer activation
    yFunc = @(wH, wO, X, N) sigmoid( aOutput(wH, wO, X, N) );
    
    for count = 1:T
        % Activity rule (doing)
        h = hFunc(Whidden, X, N);
        y = yFunc(Whidden, Wout, X, N);
        
        for i = 1:H
            % grad1 has no i = 0 term, it starts at i = 1, j = 0.
            % This means we start at second term of W2, as it starts at i = 0.
            gradHidden(i,:) = Wout(i+1) * ((y - t') .* (1 - h(i+1, :) .^2 )) * X;
            mHidden(i, :) = gradHidden(i, :)  + alpha * Whidden(i,:); 
        end
        gradOut = (y - t') * h';
        mOut = gradOut + alpha * Wout; 
        
        % Learning rule
        Whidden = Whidden - eta * mHidden;
        Wout = Wout - eta * mOut;
    end
end