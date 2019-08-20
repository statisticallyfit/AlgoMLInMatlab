%%% D = dimension of each Gaussian, the number of variables per
% multivariate Gaussian distribution Xd, in the vector X = (X1, ... XD)
%%% K = number of clusters, or number of Guassians in the mixture. 
%%% pi = mixture probabilities in the gaussian mixture pdf, length(pi) = k
%%% p = the gaussian mixture pdf formula
%%% SIGMAs = cell array containing the D x D covariance matrices of each gaussian.
% So the kth SIGMA matrix corresponds to the SIGMA matrix of the kth
% Gaussian. 
%%% MUs = cell array containg the 1 xD MU  vectors of each gaussian. 

% X = resulting N x D matrix of samples from the gaussian mixture. 
function [X] = sampleGaussianMixture(N, MUs, SIGMAs)
    
    D = length(MUs{1});
    K = length(MUs);
    
    % Generate N samples
    %N = 1000;
    X = zeros(N, D); % N x D matrix, one col per dimension
    % so rowwise ,it is one sample of dimension D 
    
    for n = 1:N
        
        % Step 1: sample k from mixtures probs (pi)
        % sample from Unif(0,1)
        unifNum = rand; 
        % find the k
        k = min(find(unifNum < cumsum(pi)));

        % Step 2: sample x-value from multivariate gaussian
        % that has the corresponding parameters, muk, sigmak
        X(n, :) = mvnrnd(MUs{k}, SIGMAs{k}); % placing into row
    end
end