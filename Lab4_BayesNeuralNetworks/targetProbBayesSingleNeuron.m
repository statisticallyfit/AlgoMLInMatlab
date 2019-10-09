

%% PARAMETERS of algorithm: 
% X = data matrix (N x I)
% t = target vector (N x 1)
% N = number of samples
% I = number of neuron inputs
% alpha = weight decay learning rate (hyperparameter)
% proposalWidth = needed for proposal sampling in metropolis montecarlo.

%% ARGUMENTS to function: 
% X = size N x I matrix with training data. 
% testX: a matrix much longer than X, and is usually M x I, where M is
% large number, and I = dimension of X
% targets = targets vector is N x 1


%% OUTPUT of function: 
% weightIndepSample = the T x 1 cell array of I x 1 independent weight vectors sampled 
% from the posterior numerator (Pstar) of weights. 

%% GOAL: sample the weights from the posterior of the weights using
% Metropolis, then average results to get the probability of target vector being = to one. 
% This is the average of the  neuron output under the posterior. 
% This method of estimation takes the WHOLE posterior distribution into account, rather than a single optimized
% weight value w_MAP under posterior (like in the other single neuron bayes
% file)

function probTargetIsOne = targetProbBayesSingleNeuron(X, targets, testX)

    % Define posterior distribution for W
    alpha = 0.01; 
    y = @(W) sigmoid(X * W);  %N x 1
    G = @(W) -(targets' * log(y(W) )  + (1-targets)' * log(1 - y(W)) );  % 1x1
    E = @(W) W'*W / 2;  %sum(W.^2, 2)' / 2  % 1x1
    M = @(W) G(W) + alpha * E(W) ;
    Pstar = @(W) exp(-M(W)); % 1x1

    % Metropolis algorithm 
    proposalWidth = 0.1; 
    [~, weightsIndep] = MetropolisMultivariateSampling(X, proposalWidth, Pstar);
    
    % learned y part : calculate predicted probabilities of targets = 1
    % testX = M x I
    % weightsIndep = R x I, where R = number of independent samples. 
    
    probTargetIsOne = mean( sigmoid(weightsIndep * testX') );

end