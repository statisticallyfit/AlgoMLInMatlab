% findM = calculates the objective function with weight-decay regularization
% findGradM = gradient of objective function
% findK = kinetic energy function, argument = p (momentum)
% H(X, P) = E(X) + K(P) = hamiltonian function
% X = position
% P = momentum
% alpha = learning rate
function [weights, weightsIndep] = hamiltonianMCBayesSingleNeuron(X, targets, alpha)
    

    %% (1) Initialize values
    
    lag = 100; %200;      %2000;
    burnin = 1000 ; %10000;
    %T = 3; % num times through entire algorithm
    Tau = 50; %20; % num times to leapfrog
    
    numIndepSamples = 100;
    
    T = burnin + numIndepSamples * lag; % TODO is this correct?
    epsilon = 0.05 ; %smaller epsilon allows larger acceptance rate
    
    
    [N, I] = size(X); 
    
    weights = zeros(T+1, I);
    W = rand(I, 1); 
    
    weights(1, :) = W' ; 
    
    numAccepted = 0; 
    
    
    M = calcM(X, W, targets, alpha);         % initial value of objective function
    gradM = calcGradM(X, W, targets, alpha); % initial value of grad of objective function
    
    
    %% (2) Update for t = 1:T
    for t = 1:T
        
        % momentum is sampled from the standard normal: 
        % P ~ Normal(0, 1), with correct vector sizes
        P = randn(size(W));  
        
        % Evaluate Hamiltonian: H(X, P) = E(X) + K(P)
        H = M + calcK(P);
        
        newW = W; % I x 1
        newGradM = gradM;  % Update the gradient of modified objective function
        
        %% (2) Update position and momentum equations using the 
        % leapfrogging algorithm, over Tau steps
        
        for tau = 1:Tau
            
            % (L1) make half a step in momentum P
            P = P - (epsilon / 2) * newGradM; 
            
            % (L2) make a full step in position X
            newW = newW + epsilon * P; 
            
            % (L3) make the last half a step in momentum P
            newGradM = calcGradM(X, newW, targets, alpha);
            P = P - (epsilon / 2) * newGradM; 
            
            % Taking care of state in this algorithm: storing previous 
            % X new values: 
            %W_stored_sub(tau, :) = newW'; 
        end
        
        % Update the Hamiltonian H
        newM = calcM(X, newW, targets, alpha);
        newK = calcK(P);
        newH = newM + newK;
        
        deltaH = newH - H;  % compute delta hamiltonian
        
        
        %% (3) Accept / reject: 
        
        if deltaH < 0 % first proposal
            accept = 1;
        elseif rand() < exp( - deltaH) % second proposal
            accept = 1; 
        else 
            accept = 0;
        end
        
        %% Update the variables for another run
        
        if accept % if proposal was accepted, update W, energy (M), gradEnergy (gradM)
            W = newW; 
            M = newM; 
            gradM = newGradM; 
        end
        
        numAccepted = numAccepted + accept; 
        %W_stored = [W_stored; W_stored_sub];
        
        weights(t + 1, :) = W';
        
    end % end t = 1:T loop
    
    fprintf('Acceptance rate: %f\n', (numAccepted / T));
    
    
    
    % Selecting independent sample: starting from burning+lag, keep every lag(th)
    % sample until we hit T
    weightsIndep = weights(burnin + lag : lag  : T, :);
    
end





%% Calculates the gradient of M, or modified objective function. 

% X is N x I matrix containing the input vectors, row-wise. 
% W is I x 1 vector, where I = dimension of the input data. 
% targets = N x 1 vector containing all targets. 
% alpha = learning rate
function gradM = calcGradM(X, w, targets, alpha)
    % compute activations: 
    a = X * w; % a is N x 1 vector
    
    % calculate outputs using sigmoid
    y = sigmoid(a); % is N x 1 vector
    
    % compute errors
    e = targets - y;
    
    % compute the gradient of G(w), the objective function
    gradG = - X' * e; 
    
    % gradM is gradient of M(w) = G(w) + alpha * E(w)
    gradM = gradG + alpha * w;
end



%% Calculates the value of the objective function with weight-decay
% regularization
% X = N x I data matrix
% w = I x 1 weights vector
% targets = N x 1 targets vector
% alpha = learning rate
function M = calcM(X, w, targets, alpha)
    
    % Objective function is G(w)
    y = @(W) sigmoid(X * W);  %N x 1
    G = @(W) -(targets' * log(y(W) )  + (1-targets)' * log(1 - y(W)) );  % 1x1
    E = @(W) W'*W / 2;
    
    % calculate the value of M
    M = G(w) + alpha * E(w);
end


%% Calculate the potential
function k = calcK(p) % p is a vector
    k = p' * p / 2;
end
