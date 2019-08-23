% Pstar = function handle from which to simulate from, used to compute
% acceptance probability
% Target distribution = P = Pstar / Z, where Z = normalizing constant so
% that Pstar becomes a probability distribution. 
% L = lengthscale (range) of the Pstar function. 
% epsilon = standard deviation (lengthscale) of the proposal distribution. 
% Q = proposal distribution = Normal(mu_t, epsilon). 

% N = number of samples to draw, after burn in
function [X,acc] = MetropolisSampling(N, L, epsilon, Pstar)
    % parameters
    %burnin = 0; % number of burn-in iterations
    burnin = (L / epsilon)^2;
    %T = N + burnin; 
    
    lag = 1; % iterations between successive samples
    
    %epsilon = 1; % standard deviation of Gaussian proposal
    x0 = -1; % start point
    
    
    % storage
    X = zeros(N,1); % samples drawn from the Markov chain
    acc = [0 0]; % vector to track the acceptance rate
    
    % MH routine
    for n = 1:burnin
        [x,a] = MHstep(x0,epsilon); % iterate chain one time step
        acc = acc + [a 1]; % track accept-reject status
    end
    
    
    for n = 1:N
        %for j = 1:lag
        
        [x,a] = MHstep(x0, epsilon); % iterate chain one time step
        acc = acc + [a 1]; % track accept-reject status
        
        %end
        X(n) = x; % store the n-th sample
    end
    
end

function [xNew,a] = MHstep(xCurrent,sig, Pstar)

    xProposal = normrnd(xCurrent,sig); % generate candidate from Gaussian
    acceptProb = min(1, Pstar(xProposal) / Pstar(xCurrent)); % acceptance probability
    u = rand; % uniform random number
    
    %if u <= acceptProb % if accepted
    %    xNew = xProposal; % new point is the candidate
    %    a = 1; % note the acceptance
    %else % if rejected
    %    xNew = xCurrent; % new point is the same as the old one
    %    a = 0; % note the rejection
    %end
    
    if u < acceptProb
        xNew = xProposal; % accepting the new state
        a = 1; % mark as accepted
    else % if rejected
        xNew = xCurrent; % new point is same as the old one
        a = 0; % mark as rejected
    end
end

function probX = targetdist(x)
    probX = exp(-x.^2) .* (2 + sin(x*5) + sin(x*2));
end 