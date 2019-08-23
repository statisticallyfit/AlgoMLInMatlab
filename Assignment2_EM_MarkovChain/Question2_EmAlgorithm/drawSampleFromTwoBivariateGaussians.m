

%% GOAL: draw a sample from a mixture of two bivariate Gaussian densities. 

%%%%% INPUT: 
% phi = the given mixing coefficients ( 1 x K )
% mu1, sigma1 = parameters of the first bivariate gaussian 
% mu2, sigma2 = parameters of the second bivariate gaussian
% where mu1, mu2 are D x 1 mean vectors, and sigma1, sigma2 are D x D
% covariance matrices, where D = 2 (since the guassians are bivariate)

% D = 2 = num dimensions of the Gaussians
% K = 2 = number of Gaussians (corresponding to each cluster)

% N = sample size to draw from the mixture of gaussians. 

%%%%% OUTPUT: 
%%% X = N x D matrix of data values


function [X, ks] = drawSampleFromTwoBivariateGaussians(mu1, mu2, sigma1, sigma2, N, phi)

    %rng('default') % reproducible
    
    givenMUs = {mu1, mu2};
    givenSIGMAs = {sigma1, sigma2};
    
    [X, ks] = sampleGaussianMixture(N, givenMUs, givenSIGMAs, phi); 
end
