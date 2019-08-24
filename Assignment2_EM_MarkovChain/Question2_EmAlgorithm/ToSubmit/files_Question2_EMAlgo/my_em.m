%% GOAL: do the EM algorithm, given a data matrix X. 

%%%%%% INPUT: 
% X = N x D matrix of data values
% PI = the starting 1 x K matrix of mixing coefficients


%%%%%% OUTPUT: 
% MU = K  x D matrix of D-dimensional mean vectors per row for each of the
% k clusters. 
% SIGMA = K x 1 cell array of the D x D covariance matrices per cluster. 
% PI = the 1 x K matrix of mixing coefficients, optimized by the
% algorithm. 

function [MU, SIGMA, PI] = my_em(X, PI, K)
    D = size(X, 2);
    
    [MU, SIGMA] = initStep(X, K);
    
    
    % Repeat until no change in means.
    
    numIters = 0; % keep track of how many iterations it takes to achieve tolerance.
    oldMU = zeros(K, D);
    
    while (sum(sum(abs(MU - oldMU))) > 1e-2)
      
        % store the current means
        oldMU = MU; 
        
        resp = expectationStep(X, MU, SIGMA, PI);
        
        [MU, SIGMA, PI] = maximizationStep(resp, X);
  
        numIters = numIters + 1;
    end
    
    
    fprintf('Num iterations in EM algo: %d \n', numIters);
end


%% (1) INITIALIZATION:---------------------------------------------------------------

% Initialize the starting values for means and covariances for each
% cluster k.
function [MU, SIGMA] = initStep(X, K)
    
    % using the matlab kmeans function to get the labels for each data
    % point. (Each label says to which cluster the data point belongs)
    [clusterLabels, MU] = kmeans(X, K); % K x D matrix of mean vectors per cluser. 
    
    
    % Initialize the covaraince matrices of each cluster. 
    SIGMA = cell(K, 1);
    
    for k = 1:K
        % Get the row indices for which the cluster label matches current cluster k
        clusterRows = find(clusterLabels == k);
        % Get the data for the above row indices
        clusterData = X(clusterRows, :);
        % Getting covariance of this cluster of the data. 
        SIGMA{k} = cov(clusterData); 
    end
end


%% (2) EXPECTATION STEP: ---------------------------------------------------------------
    
% evaluate the responsibilities using the current parameter values.
% Means to calculate the probability that each data point belongs to
% each cluster. 
function resp = expectationStep(X, MU, SIGMA, PI)
    N = size(X, 1);
    K = size(MU, 1);
    
    pdf = zeros(N, K);

        % For each cluster...
        for k = 1 : K
            % Evaluate the Gaussian for all data points for cluster 'k'.
            pdf(:, k) = mvnpdf(X, MU(k, :), SIGMA{k});
        end

        % Complete calculation of the mixture probability: Multiply each pdf value by the prior probability for cluster.

        priorPdfValue = bsxfun(@times, pdf, PI); % (row-wise divide) pdf .* PI
        totalPriorPdfValues = sum(priorPdfValue, 2); 
        resp = priorPdfValue ./ totalPriorPdfValues;  % (row-wise divide) bsxfun(@rdivide, priorPdfValue, totalPriorPdfValues)
end



%% (3) MAXIMIZATION STEP:  ---------------------------------------------------------------
% Re-estimate the parameters using the current responsibilities. 

function [MU, SIGMA, PI] = maximizationStep(resp, X)

    D = size(X, 2);
    K = size(resp, 2);
    N = size(X, 1);

    % Calculate the new prior probabilities 
    Nk = sum(resp, 1); % sum along the columns (vertically) => (1 x K)
    PI = Nk / N; % 1 x K
    MU = bsxfun(@rdivide, resp' * X, Nk');
    SIGMA = cell(K, 1);
    
    % Update the covariance matrices
    for k = 1 : K
        
        % Subtract the cluster row mean vector from all data points.
        Xm = X - MU(k,:); % bsxfun(@minus, X, MU(k, :)); % applies minus elementwise  to X. 
        
        sigma_k = zeros(D, D);
        % Calculate the contribution of each training example to the covariance matrix.
        for (n = 1 : N)
            sigma_k = sigma_k + (resp(n, k) .* (Xm(n, :)' * Xm(n, :)));
        end
        % Divide by the sum of weights.
        SIGMA{k} = sigma_k ./ sum(resp(:, k));
    end
end