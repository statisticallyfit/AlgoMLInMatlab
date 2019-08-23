%% GOAL:  Sample the data, perform EM algorithm, and average all the results over the B iterations. 

%%%%% INPUT: 
%% phi = the given mixing coefficients ( 1 x K )
% mu1, sigma1 = parameters of the first bivariate gaussian 
%% mu2, sigma2 = parameters of the second bivariate gaussian
% where mu1, mu2 are D x 1 mean vectors, and sigma1, sigma2 are D x D
% covariance matrices, where D = 2 (since the guassians are bivariate)
%% D = 2 = num dimensions of the Gaussians
% K = 2 = number of Gaussians (corresponding to each cluster)
% N = sample size to draw from the mixture of gaussians. 
% B = number of iterations to do the EM algorithm.

%%%%% OUTPUT:
%% avgMUs = the average matrix (K x D) of all MU (K x D) structures created during each
% iteration of the EM algorithm. 
%% avgSIGMAs = the average cell array (K x 1) containing all the D x D SIGMA
% matrices created during each iteration of the EM algorithm. 
%% avgPIs = the average vector (1 x K) of all the mixture coefficients, PI, created 
% during each iteration of the EM algorithm. 

function [avgMU, avgSIGMA, avgPI] = averageEMResults(B, mu1, mu2, sigma1, sigma2, N, phi)

    rng('default') % reproducible
    
    
    % D = dimension of each Gaussian, K = number of Gaussians (cluster)
    D = 2; 
    K = 2; 
    
    % Storing all the sigmas, mus, and pis over the B iterations
    allSIGMAs = cell(B, 1);
    allMUs = cell(B, 1);
    allPIs = cell(B, 1);
    
    allsigmas = cell(B, 1);
    allmus = cell(B, 1);
    allpis = cell(B, 1);
    
    
    %% Step 3: repeat steps 1 and 2 for B times
    for iter = 1:B
        
        %% Step 1: generate sample of size N from mixture of two bivariate Gaussians.
        [X, ks] = drawSampleFromTwoBivariateGaussians(mu1, mu2, sigma1, sigma2, N, phi);

        %% Step 2: do the EM algorithm
        %[MU, SIGMA, PI] = EMAlgo(X, phi);
        [MU, SIGMA, PI] = EMAlgoGaussianMixture_1(X, phi);
        %[Mu, Sigma, Pi] = EMAlgoGaussianMixture_1(X, phi);
        
        allMUs{iter} = MU;
        %allmus{iter} = Mu;
        
        allSIGMAs{iter} = SIGMA;
        %allsigmas{iter} = Sigma;
        
        allPIs{iter} = PI;
        %allpis{iter} = Pi;
        
    end
    
    %% Step 3: Average the results

    % average all the mu's
    m = cat(3, allMUs{:});
    avgMU = mean(m, 3);
    %m2 = cat(3, allmus{:});
    %avgmu = mean(m2, 3);
    
    % average all the pi's
    p = cat(3, allPIs{:});
    avgPI = mean(p, 3);
    %p2 = cat(3, allpis{:});
    %avgpi = mean(p2, 3);
    
    % average all the sigmas
    s = cat(3, allSIGMAs{:});
    avgSIGMA = mean(s, 3);
    
    %s2 = cat(3, allsigmas{:});
    %avgsigma = cat(s2, 3);
    
end
