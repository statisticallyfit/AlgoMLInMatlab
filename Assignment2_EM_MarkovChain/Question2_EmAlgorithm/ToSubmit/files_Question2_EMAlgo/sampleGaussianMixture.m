%%% INPUT: 

%%% D = dimension of each Gaussian, the number of variables per
% multivariate Gaussian distribution Xd, in the vector X = (X1, ... XD)
%%% K = number of clusters, or number of Guassians in the mixture. 
%%% PI = mixture probabilities in the gaussian mixture pdf, length(pi) = k
%%% p = the gaussian mixture pdf formula
%%% SIGMAs = cell array containing the D x D covariance matrices of each gaussian.
% So the kth SIGMA matrix corresponds to the SIGMA matrix of the kth
% Gaussian. 
%%% MUs = cell array containg the 1 x D MU  vectors of each gaussian. 

%%% OUTPUT: 
% X = resulting N x D matrix of samples from the gaussian mixture. 
% ks = the cluster label corresponding to which Gaussian we drew from. 
function [X] = sampleGaussianMixture(N, MUs, SIGMAs, PI)
    
    D = length(MUs{1});
    K = length(MUs);
    
    % Generate N samples
    X = zeros(N, D); % N x D matrix
    
    for n = 1:N
        
        % Step 1: sample k from mixtures probs (pi)
        % sample from Unif(0,1)
        unifNum = rand; 
        k = min(find(unifNum < cumsum(PI)));

        % Step 2: sample x-value from multivariate gaussian
        % that has the corresponding parameters, muk, sigmak
        X(n, :) = mvnrnd(MUs{k}, SIGMAs{k}); % placing into row n
    end
end