%% INPUT: 
% Pstar = function handle from which to simulate from, used to compute
% acceptance probability
% Target distribution = P = Pstar / Z, where Z = normalizing constant so
% that Pstar becomes a probability distribution. 
% L = lengthscale (range) of the Pstar function. 
% proposalWidth = standard deviation (lengthscale) of the proposal distribution. 
% Q = proposal distribution = Normal(mu_t, epsilon). 

% N = number of samples to draw, after burn in

%% OUTPUT: 
% X = the markov chain sequence (samples) from the target distribution. 
% X represents the samples from the posterior, P = P* / Z

function [X] = my_metropolis(N, L, proposalWidth, xInit, Pstar)

    % The burn in is the lower bound of number of iterations needed to
    % generated independent samples, since markov chains generate
    % sequentially dependent samples. 
    burnin = (L / proposalWidth)^2;

    % After burnin, generate N samples. 
    T = N + burnin; 
    
    % Proposal distribution with current mean and proposal width (sigma)
    sampleProposalQ = @(mu, sigma) normrnd(mu, sigma);
    
    % Preparing the posterior samples using the Markov chain
    X = zeros(N,1); 
    
    % Let first current mean be 0.5 since any beta distribution, and hence
    % the posterior, have x-values between 0 and 1. Let the first xInit
    % be midway between 0 and 1. 
    %xInit = 0.5; 
    
    % Sample the proposal distribution
    X(1) = sampleProposalQ(xInit, proposalWidth); 
    
    % Generate N uniform random numbers. 
    u = rand(1, T); 
    
    % Metropolis Routine
    for t = 1:T-1
        % Generate the next proposal value from the symmetric proposal
        % distribution Q
        muProposal = sampleProposalQ(X(t), proposalWidth); 
        
        acceptProb = min(1, Pstar(muProposal) / Pstar(X(t))); % acceptance probability
        
        % Check if the proposal for the coin bias f is within range: 0 <= f <= 1
        proposalInRange = muProposal >= 0 && muProposal <= 1;
        
        if (u(t) < acceptProb) && (proposalInRange)
            X(t + 1) = muProposal; % accepting the new state
            
        else  % if rejected or proposal is not in range, reject the proposed value.
            X(t + 1) = X(t); % new point is same as the old one
        end
    end
    
    
    % TODO must pick out every Mth sample to get every independent samples.
    % Step 1: keep the samples after the burnin
    X = X(burnin + 1:end);
    
end

