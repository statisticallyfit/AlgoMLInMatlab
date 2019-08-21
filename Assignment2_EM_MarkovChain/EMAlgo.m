% phi = the given mixing coefficients ( 1 x K )
% mu1, sigma1 = parameters of the first bivariate gaussian 
% mu2, sigma2 = parameters of the second bivariate gaussian
% where mu1, mu2 are D x 1 mean vectors, and sigma1, sigma2 are D x D
% covariance matrices, where D = 2 (since the guassians are bivariate)

% D = 2 = num dimensions of the Gaussians
% K = 2 = number of Gaussians (corresponding to each cluster)

%%% X = N x D matrix of data values

%%% MU = K  x D matrix of D-dimensional mean vectors per row for each of the
% k clusters. 

%%% SIGMA = K x 1 cell array of the D x D covariance matrices per cluster. 

function EMAlgo(mu1, mu2, sigma1, sigma2, phi)

    %% Step 1: generate sample of size N from mixture of two bivariate
    % Gaussians. 
    D = 2; 
    K = 2; 
    N = 100; 
    
    % Probability density of mixture: 
    p = @(x) phi(1) * mvnpdf(x, mu1, sigma1) + phi(2) * mvnpdf(x, mu2, sigma2);

    givenMUs = {mu1, mu2};
    givenSIGMAs = {sigma1, sigma2};
    
    X = sampleGaussianMixture(N, givenMUs, givenSIGMAs); 
    
    
    %% Step 2: do the EM algorithm
    
    % (1) INITIALIZATION: 
    
    % Initialize the starting values for means and covariances for each
    % cluster k.
    % NOTE: these are of course different from the given values since we
    % must work our way up to getting close to the original estimates.
    
    % using the matlab kmeans function to get the labels for each data
    % point. (Each label says to which cluster the data point belongs)
    [clusterLabels, centroids] = kmeans(X, K); % K x D matrix of mean vectors per cluser. 
    MU = cell(K, 1);
    SIGMA = cell(K, 1);
    
    for k = 1:K
        % Get the mean for this current cluster k. 
        MU{k} = centroids(k, :); % store as row vector form. 
        
        % Get the row indices for which the cluster label matches current cluster k
        clusterRows = find(clusterLabels == k);
        % Get the data for the above row indices
        clusterData = X(clusterRows, :);
        % Getting covariance of this cluster of the data. 
        SIGMA{k} = cov(clusterData); 
        
    end
    
    
    % (2) EXPECTATION STEP: 
    
    % evaluate the responsibilities using the current parameter values.
    % Means to calculate the probability that each data point belongs to
    % each cluster. 
    
    % Matrix to hold the multivariate pdf value for each every data point for every cluster.
    % One row per data point, one column per cluster.
    pdf = zeros(N, K);

    % For each cluster...
    for k = 1 : K

        % Evaluate the Gaussian for all data points for cluster 'k'.
        %pdf(:, k) = mvnpdf(X, MU(k, :), SIGMA{k});
        % note: sum(..., 2) means sum along the rows, so result of sum is
        % N x 1 matrix
        pdf(:, k) = exp( (-1/2)* sum( (X - MU{k}) * inv(SIGMA{k}) .* (X - MU{k}), 2) / ...
            sqrt( (2 * pi)^D * det(SIGMA{k})); 
    end

    % Complete the calculation of the mixture probability. 
    % Multiply each pdf value by the prior probability for cluster.
    
    % row-wise multiply, same as bsxfun(@times, p, phi^T)
    priorPdfValue = pdf .* transpose(phi);
    totalPriorPdfValues = sum(priorPdfValue, 2); 
    % row-wise divide, same as bsxfun(@rdivide, p, t)
    resp = priorPdfValue ./ totalPriorPdfValues;  
    
    
    
    % (3) MAXIMIZATION STEP: 
    % Re-estimate the parameters using the current responsibilities. 
    
    % store the current means
    oldMU = MU;  
    
    % Calculate the new prior probabilities 
    Nk = sum(resp, 1); % sum along the columns (vertically) => (1 x K)
    phi = Nk / N; % 1 x K
    
    % Calculate new means for each cluster
    %newMeans = (transpose(resp) * X)
    
    for k = 1 : K

        % Calculate the prior probability for cluster 'j'.
        %pi(k) = mean(resp(:, k), 1);

        % Calculate the new mean for cluster 'j' by taking the weighted
        % average of all data points.
        % MU(k, :) = ...
        MU{k} = (transpose(resp(:, k)) * X) ./ sum(resp(:, k), 1);  %weightedAverage(resp(:, k), X);

        % Calculate the covariance matrix for cluster 'k' by taking the 
        % weighted average of the covariance for each training example. 

        sigma_k = zeros(D, D);

        % Subtract the cluster row mean vector from all data points.
        Xm = X - MU{k}; % bsxfun(@minus, X, MU(k, :)); % applies minus elementwise  to X. 

        % Calculate the contribution of each training example to the covariance matrix.
        %for (n = 1 : N)
        %    sigma_k = sigma_k + (resp(n, k) .* (Xm(n, :)' * Xm(n, :)));
        %end
        % Divide by the sum of weights.
        %sigma{k} = sigma_k ./ sum(resp(:, k));
        
        SIGMA{k} = (  (resp(:, k) .* Xm)' * Xm) / sum(resp, 1); 
    end

    % Check for convergence.
    if (mu == prevMu)
        break
    end
        
    
end