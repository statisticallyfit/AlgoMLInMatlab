% pi = the given mixing coefficients ( K x 1 )
% mu1, sigma1 = parameters of the first bivariate gaussian 
% mu2, sigma2 = parameters of the second bivariate gaussian
% where mu1, mu2 are D x 1 mean vectors, and sigma1, sigma2 are D x D
% covariance matrices, where D = 2 (since the guassians are bivariate)

% D = 2 = num dimensions of the Gaussians
% K = 2 = number of Gaussians (corresponding to each cluster)

function EMAlgo(mu1, mu2, sigma1, sigma2, pi)

    %% Step 1: generate sample of size N from mixture of two bivariate
    % Gaussians. 
    D = 2; 
    K = 2; 
    N = 100; 
    
    % Probability density of mixture: 
    p = @(x) pi(1) * mvnpdf(x, mu1, sigma1) + pi(2) * mvnpdf(x, mu2, sigma2);

    MUs = {mu1, mu2};
    SIGMAs = {sigma1, sigma2};
    
    X = sampleGaussianMixture(N, MUs, SIGMAs); 
    
    % Initialize the starting values for means and covariances for each k
    % NOTE: these are of course different from the given values since we
    % must work our way up to getting close to the original estimates.
    centroids = kmeans(K, X);
    m = cell(D, 1);
    s = cell(D, D);
    
    for k = 1:K
        m{k} = centroids(k, :)'; %% TODO: check since dim of kmeans output 
        %is not matching the centroids of m expected dimension...
        
    
    
    
    %% Step 2: do the EM algorithm
    
end