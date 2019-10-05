

%% PARAMETERS of algorithm: 
% X = data matrix (N x I)
% t = target vector (N x 1)
% N = number of samples
% I = number of neuron inputs
% alpha = weight decay learning rate (hyperparameter)

%% ARGUMENTS to function: 
% X
% t
% alpha, eta

%% OUTPUT of function: 
% weightIndepSample = the T x 1 cell array of I x 1 independent weight vectors sampled 
% from the posterior numerator (Pstar) of weights. 

%% GOAL: sample the weights from the posterior of the weights using
% Metropolis, then average results to get the probability of target vector being = to one. 
% This is the average of the  neuron output under the posterior. 
% This method of estimation takes the WHOLE posterior distribution into account, rather than a single optimized
% weight value w_MAP under posterior (like in the other single neuron bayes
% file)

function probTargetIsOne = bayesTargetProbabilityForSingleNeuron(X, t, alpha)

    % Define posterior distribution for W
    
    y = @(w) sigmoid(X * w);  %N x 1
    G = @(w) -(t' * log(y(w) )  + (1-t)' * log(1 - y(w)) );  % 1x1
    E = @(w) w' * w / 2;  %sum(W.^2, 2)' / 2  % 1x1
    M = @(w) G(w) + alpha * E(w) ;
    Pstar = @(w) exp(-M(w)); % 1x1

    % Metropolis algorithm 
    proposalWidth = 0.1; 
    weightsIndepSample = MetropolisMultivariateSampling(X, proposalWidth, Pstar);
    
    % learned y part ... TODO

end