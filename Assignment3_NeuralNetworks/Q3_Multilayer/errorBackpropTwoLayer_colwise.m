% ERROR BACKPROP for 2-layer network with 3 hidden units

% H = num hidden units
% X = N x I data matrix
% t = N x 1 target vector
function [Whidden, Wout] = errorBackpropTwoLayer_colwise(X, H, targets)
    
    % (1) INITIALIZE weights and biases
    %rng('default') % keeping the same seed makes decision boundary LINEAR - bad
    
    [N, I] = size(X);
    
    %H = 3; % num hidden units
    Whidden = rand(I , H);
    Wout = rand(H + 1, 1);
    
    T = 50000;
    eta = 0.01;
    alpha = 0.012;
    
    % Activity rule
    aHidden = @(wH, X) X * wH; % hidden layer activation (N x H matrix)
    hFunc = @(wH, X, N) [ones(N,1) , tanh( aHidden(wH, X) )]; % adds a row of ones for the W0 biases (N x (H+1)) matrix
    
    aOutput = @(wH, wO, X, N) hFunc(wH, X, N) * wO; % output layer activation  (N x 1 matrix)
    yFunc = @(wH, wO, X, N) sigmoid( aOutput(wH, wO, X, N) );  % (N x 1) matrix
    
    for count = 1:T
        % Activity rule (doing)
        h = hFunc(Whidden, X, N);       % N x (H+1)
        y = yFunc(Whidden, Wout, X, N); % N x 1
        
        for i = 1:H
            % grad1 has no i = 0 term, it starts at i = 1, j = 0.
            % This means we start at second term of W2, as it starts at i = 0.
            gradGHidden(i,:) = Wout(i+1) * ( ((y - targets) .* (1 - h(:, i+1) .^2 ) ) )' * X;
            
            %% updated colwise until above line, rest is not colwise. 
            gradMHidden(i, :) = gradGHidden(i, :)  + alpha * Whidden(i,:); 
        end
        
        gradGOut = (y - targets') * h';
        gradMOut = gradGOut + alpha * Wout; 
        
        % Learning rule
        Whidden = Whidden - eta * gradMHidden;
        Wout = Wout - eta * gradMOut;
    end
end