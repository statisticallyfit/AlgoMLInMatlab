%% GOAL:  Sample the data, perform EM algorithm, and average all the results over the B iterations. 

%%%%% INPUT: 
%% phi = the given mixing coefficients ( 1 x K )
%% givenMUs = K x 1 cell array holding means of the k  D-dimensional gaussians
%% givenSIGMAs = K x 1 cell array holding covariance matrices of the k D-dimensional gaussians
% where mu_k is a 1 x D mean vectors, and sigma_k is D x D  covariance
% matrix
%% D = num dimensions of the Gaussians
% K = number of Gaussians (corresponding to each cluster)
% N = sample size to draw from the mixture of gaussians. 
% B = number of iterations to do the EM algorithm.

%%%%% OUTPUT:
%% avgMUs = the average matrix (K x D) of all MU (K x D) structures created during each
% iteration of the EM algorithm. 
%% avgSIGMAs = the average cell array (K x 1) containing all the D x D SIGMA
% matrices created during each iteration of the EM algorithm. 
%% avgPIs = the average vector (1 x K) of all the mixture coefficients, PI, created 
% during each iteration of the EM algorithm. 

function [avgMU, avgSIGMA, avgPI, allMUs, allSIGMAs, allPIs] = averageEMResults(B, givenMUs, givenSIGMAs, N, phi)

    rng('default') % Setting the seed once before EM runs B times. 
    
    % D = dimension of each Gaussian, K = number of Gaussians (cluster)
    D = length(givenMUs{1}); 
    K = length(givenMUs); 
    
    % Storing all the sigmas, mus, and pis over the B iterations
    allSIGMAs = cell(B, 1);
    allMUs = cell(B, 1);
    allPIs = cell(B, 1);
    
    
    %% Step 3: repeat steps 1 and 2 for B times
    for iter = 1:B
        
        %% Step 1: generate sample of size N from mixture of two bivariate Gaussians.
        
         X = sampleGaussianMixture(N, givenMUs, givenSIGMAs, phi); 

        %% Step 2: do the EM algorithm
        [MU, SIGMA, PI] = my_em(X, phi, K);
        
        allMUs{iter} = MU;
        allSIGMAs{iter} = SIGMA';
        allPIs{iter} = PI;
    end
    
    %% Step 3: Average the results

    % average all the mu's
    m = cat(3, allMUs{:});
    avgMU = mean(m, 3);
    
    
    % average all the pi's
    p = cat(3, allPIs{:});
    avgPI = mean(p, 3);
    
    % average all the sigmas
    sk = cell(B, 1); 
    
    for k = 1:K
        for b = 1:B
            sk{b} = allSIGMAs{b}{k};
        end
        
        c = cat(3, sk{:});
        avgSIGMA{k} = mean(c, 3);
        
    end
    
end
