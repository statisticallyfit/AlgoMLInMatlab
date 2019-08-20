% D = 2 = number of dimensions per Guassian pdf  (since each Gaussian is
% bivariate)
% N = number of data points
% K = 2 = number of clusters = number of bivariate Guassians
% mu_k = the mean vector for the kth Gaussian 
% sigma_k = the covariance matrix for the kth Gaussian
% PI_k = the mixing probability for the kth gaussian
% X = data matrix
function [mu, sigma] = EMAlgoGaussianMixture_1(X)

    % (1) Initialize the vector of means, covariance matrix, and prior
    % probabilities for all K=2 components (all K=2 gaussian mixtures)
    
    N = size(X, 1);

    K = 2;  % The number of clusters.
    D = 2;  % The vector lengths.

    % Randomly select k data points to serve as the initial means.
    % TODO: update this to use kmeans?
    indeces = randperm(N);
    mu = X(indeces(1:K), :);

    sigma = [];

    % Use the overall covariance of the dataset as the initial variance for each cluster.
    for (k = 1 : K)
        sigma{k} = cov(X);
    end

    % Assign equal prior probabilities to each cluster.
    pi = ones(1, K) * (1 / K);
    
    

    % Run Expectation Maximization

    % Responsibilities matrix to hold the probability that each data point 
    % belongs to each cluster.
    % One row per data point, one column per cluster.
    resp = zeros(N, K);

    % Loop until convergence.
    for (iter = 1:1000)

        fprintf('  EM Iteration %d\n', iter);

        %%% E step: Expectation: 
        % Calculate the probability for each data point for each distribution.
        % This is the responsibility. 

        % Matrix to hold the pdf value for each every data point for every cluster.
        % One row per data point, one column per cluster.
        pdf = zeros(N, K);

        % For each cluster...
        for (k = 1 : K)

            % Evaluate the Gaussian for all data points for cluster 'j'.
            pdf(:, k) = gaussianND(X, mu(k, :), sigma{k});
        end

        % Multiply each pdf value by the prior probability for cluster.
        %    pdf  [m  x  k]
        %    phi  [1  x  k]   
        %  pdf_w  [m  x  k]
        numerator = bsxfun(@times, pdf, pi); % this is the numerator of the responsibilities expression.

        % Divide the weighted probabilities by the sum 
        % of weighted probabilities for each cluster.
        denominator = sum(numerator, 2); 
        resp = bsxfun(@rdivide, numerator, denominator);

        % (3) M step: Maximization step: 
        % Re-estimate the parameters using the current responsibilities. 
        %% Calculate the probability for each data point for each distribution.

        % Store the previous means.
        prevMu = mu;    

        % For each of the clusters...
       % Nk = sum(resp);
        %PI = Nk / N; 
       % MU = (transpose(resp) * X) ./ transpose(Nk);
       % SIGMA = resp' * 
        for (k = 1 : K)

            % Calculate the prior probability for cluster 'j'.
            pi(k) = mean(resp(:, k), 1);

            % Calculate the new mean for cluster 'j' by taking the weighted
            % average of all data points.
            mu(k, :) = weightedAverage(resp(:, k), X);

            % Calculate the covariance matrix for cluster 'j' by taking the 
            % weighted average of the covariance for each training example. 

            sigma_k = zeros(D, D);

            % Subtract the cluster row mean vector from all data points.
            Xm = bsxfun(@minus, X, mu(k, :)); % applies minus elementwise  to X. 

            % Calculate the contribution of each training example to the covariance matrix.
            for (n = 1 : N)
                sigma_k = sigma_k + (resp(n, k) .* (Xm(n, :)' * Xm(n, :)));
            end

            % Divide by the sum of weights.
            sigma{k} = sigma_k ./ sum(resp(:, k));
        end

        % Check for convergence.
        if (mu == prevMu)
            break
        end

    % End of Expectation Maximization    
    end
end

