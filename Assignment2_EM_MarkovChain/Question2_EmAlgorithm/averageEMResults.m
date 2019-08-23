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

function [avgMUs, avgSIGMAs, avgPIs] = averageEMResults(B, mu1, mu2, sigma1, sigma2, N, phi)

    % D = dimension of each Gaussian, K = number of Gaussians (cluster)
    D = 2; 
    K = 2; 
    
    % Storing all the sigmas, mus, and pis over the B iterations
    allSIGMAs = cell(B, 1);
    allMUs = cell(B, 1);
    allPIs = cell(B, 1);
    
    %% Step 3: repeat steps 1 and 2 for B times
    for iter = 1:B
        
        %% Step 1: generate sample of size N from mixture of two bivariate Gaussians.
        X = drawSampleFromTwoBivariateGaussians(mu1, mu2, sigma1, sigma2, N, phi);

        %% Step 2: do the EM algorithm
        [MU, SIGMA, PI] = EMAlgo(X);
        
        allMUs{iter} = MU;
        allSIGMAs{iter} = SIGMA;
        allPIs{iter} = PI;
        
    end
    
    %% Step 3: Average the results

    % average all the mu's
    m = cat(3, allMUs{:});
    avgMUs = mean(m, 3);
    
    % average all the pi's
    p = cat(3, allPIs{:});
    avgPIs = mean(p, 3);
    
    % average all the sigmas
    s = cat(3, SIGMA{:});
    avgSIGMA = mean(s, 3);
    
end
 


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


function [X] = drawSampleFromTwoBivariateGaussians(mu1, mu2, sigma1, sigma2, N, phi)

    rng('default') % reproducible
    
    D = 2; 
    K = 2; 
    
    PI = phi; 
    givenMUs = {mu1, mu2};
    givenSIGMAs = {sigma1, sigma2};
    
    X = sampleGaussianMixture(N, givenMUs, givenSIGMAs, PI); 
end





%% GOAL: do the EM algorithm, given a data matrix X (N x D). 

%%%%%% INPUT: 
%%% X = N x D matrix of data values

%%%%%% OUTPUT: 
%%% MU = K  x D matrix of D-dimensional mean vectors per row for each of the
% k clusters. 

%%% SIGMA = K x 1 cell array of the D x D covariance matrices per cluster. 

%%% PI = the 1 x K matrix of mixing coefficients, optimized by the
%%% algorithm. 

function [MU, SIGMA, PI] = EMAlgo(X)

    % (1) INITIALIZATION: 
    
    % Initialize the starting values for means and covariances for each
    % cluster k.
    
    % using the matlab kmeans function to get the labels for each data
    % point. (Each label says to which cluster the data point belongs)
    [clusterLabels, MU] = kmeans(X, K); % K x D matrix of mean vectors per cluser. 
    %MU = cell(K, 1);
    SIGMA = cell(K, 1);
    
    
    oldMU = zeros(K, D);
    
    for k = 1:K
        % Get the row indices for which the cluster label matches current cluster k
        clusterRows = find(clusterLabels == k);
        % Get the data for the above row indices
        clusterData = X(clusterRows, :);
        % Getting covariance of this cluster of the data. 
        SIGMA{k} = cov(clusterData); 
        
    end
    
    
    % Repeat until no change in means.
    
    while (sum(sum(abs(MU - oldMU))) > 1e-2)
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
            pdf(:, k) = mvnpdf(X, MU(k, :), SIGMA{k});
            % note: sum(..., 2) means sum along the rows, so result of sum is
            % N x 1 matrix
            %pdf(:, k) = exp( (-1/2)* sum( (X - MU(k, :)) * inv(SIGMA{k}) .* (X - MU(k,:)), 2) ) / ...
            %    sqrt( (2 * pi)^D * det(SIGMA{k}) ) ; 
        end

        % Complete the calculation of the mixture probability. 
        % Multiply each pdf value by the prior probability for cluster.

        % row-wise multiply, same as bsxfun(@times, p, phi^T)
        priorPdfValue = bsxfun(@times, pdf, PI); % pdf .* PI
        totalPriorPdfValues = sum(priorPdfValue, 2); 
        % row-wise divide, same as bsxfun(@rdivide, p, t)
        resp = priorPdfValue ./ totalPriorPdfValues;  


        % (3) MAXIMIZATION STEP: 
        % Re-estimate the parameters using the current responsibilities. 

        % store the current means
        oldMU = MU;  

        % Calculate the new prior probabilities 
        Nk = sum(resp, 1); % sum along the columns (vertically) => (1 x K)
        PI = Nk / N; % 1 x K

        for k = 1 : K
            % Calculate the new mean for cluster 'j' by taking the weighted
            % average of all data points.
            MU(k, :) = (transpose(resp(:, k)) * X) ./ sum(resp(:, k), 1);  %weightedAverage(resp(:, k), X);

            % Calculate the covariance matrix for cluster 'k' by taking the 
            % weighted average of the covariance for each training example. 

            

            % Subtract the cluster row mean vector from all data points.
            Xm = X - MU(k,:); % bsxfun(@minus, X, MU(k, :)); % applies minus elementwise  to X. 

            sigma_k = zeros(D, D);
            % Calculate the contribution of each training example to the covariance matrix.
            for (n = 1 : N)
                sigma_k = sigma_k + (resp(n, k) .* (Xm(n, :)' * Xm(n, :)));
                %sigma_k = (sigma_k +  transpose(sigma_k)) / 2; % to fix asymmetry and errors
            end
            % Divide by the sum of weights.
            SIGMA{k} = sigma_k ./ sum(resp(:, k));

            %SIGMA{k} = (  (resp(:, k) .* Xm)' * Xm) / sum(resp, 1); 
        end
    end
    
    
end

