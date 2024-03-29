% findE = energy function, argument = x (position)
% findGradE = gradient of energy function
% findK = kinetic energy function, argument = p (momentum)
% H(X, P) = E(X) + K(P) = hamiltonian function
% X = position
% P = momentum
function [X_stored, X_term, numAccepted, T] = hamiltonianMonteCarlo(findE, findGradE, findK)
    
    % (1) Initialize
    T = 3; % num times through entire algorithm
    Tau = 20; % num times to leapfrog
    epsilon = 0.05; % stepsize in leapfrog
    
    X_stored_sub = zeros(Tau, 2); 
    X_stored = zeros(1, 2); 
    X_term = zeros(T+1, 2);
    X = [-0.9, -0.7]; 
    X_stored(1, :) = X; 
    X_term = X; 
    
    numAccepted = 0; 
    
    E = findE(X); % initial value of energy
    gradE = findGradE(X);
    
    
    
    % (2) Update for t = 1:T
    for i = 1:T
        
        % momentum is sampled from the standard normal: 
        % P ~ Normal(0, 1), with correct vector sizes
        P = randn(size(X));  % sample size(X) random normal numbers
        
        % Evaluate Hamiltonian: H(X, P) = E(X) + K(P)
        H = E + findK(P);
        
        % Initialize X_new to the first X
        newX = X;
        
        % Update the gradient of E
        newGradE = gradE; 
        
        % Update position and momentum equations using the 
        % leapfrogging algorithm, over Tau steps
        for tau = 1:Tau
            % (L1) make half a step in momentum P
            P = P - (epsilon / 2) * transpose(newGradE); 
            % (L2) make a full step in position X
            newX = newX + epsilon * P; 
            % (L3) make the last half a step in momentum P
            newGradE = findGradE(newX);
            P = P - (epsilon / 2) * transpose(newGradE); 
            
            % Taking care of state in this algorithm: storing previous 
            % X new values: 
            X_stored_sub(tau, :) = newX; 
        end
        
        % Update the Hamiltonian H
        newE = findE(newX);
        newK = findK(P);
        newH = newE + newK;
        
        deltaHamiltonian = newH - H; 
        
        
        % (3) Decide whether to accept / reject: 
        
        if deltaHamiltonian < 0 % first proposal
            accept = 1;
            % if the random uniform Unif(0,1) sampled number is less than exp ...
        elseif rand() < exp( - deltaHamiltonian) % second proposal: always accepted (??)
            accept = 1; 
        else 
            accept = 0;
        end
        
        % (4) Update the variables for another run
        
        if accept % if proposal was accepted, update the variables
            X = newX; 
            E = newE; 
            gradE = newGradE; 
        end
        
        numAccepted = numAccepted + accept; 
        X_stored = [X_stored; X_stored_sub];
        X_term(i + 1, :) = X;
        
    end % end i = 1:T loop
end