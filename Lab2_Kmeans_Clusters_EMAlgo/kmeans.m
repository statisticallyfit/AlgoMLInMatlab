
%%% K  = number of data clusters (given by user)
%%% D = number of dimensions in the data (num predictors?)
%%% N = number of data points
%%% means_k = vector of means for the kth cluster
%%% data = x_n = vector containing D components where the ith component is x_i.
%%% This is one data point (a D-length vector)
%%% NOTE: the entire set of data points is {x_n} = N x D matrix
%%% predictorIndices = the indices of the predictors in the data matrix
%%% (which cols are the predictors in the data)

%%% X = data matrix, is N x D matrix, where each predictor x1, x2, .. xD is
%%% along the column and each predictor has N observations (along the rows)

function [means, resp, distance] = kmeans(K, X)

    %%%% KMEANS Algorithm %%%%
    % The Simple perceptron version


    % define the distance function
    dist = @(x,y) (1/2) * sum(( x - y) .^2);

    % (1) Initialization step: set K means to random values

    [N, D] = size(X); 
    
    means = zeros(K,D); % means is a K x D matrix

    %rng(117375); % set the seed for random reproducibility. 
    for d = 1:D
        xd = X(:,d);
        means(:, d) = (max(xd) - min(xd)) * rand(K, 1) + min(xd);
    end
    % have initialized the means with random values. 
    oldMeans = zeros(K, D); % keeping the old means
    %stepmeans = means;


    % Repeat until no change in means.
    while (sum(sum(abs(means - oldMeans))) > 1e-2)

        % (2) Assignment step: each data point is assigned to the  nearest
        % mean. Create responsibilities = indicator variables to 
        % represent assignment of data points to clusters. 
        resp = zeros(N, K); % N x K matrix 
        oldMeans = means; 

        for n = 1:N % for the nth row ...
            % create distance vector, length K, num clusters
            distance = zeros(1, K); % K-length array

            % MEANING: for a fixed row n in the data, get distance between all K means
            for j = 1:K % for the jth cluster...
                % get the jth mean vector of length D , (row)
                % get nth row of all predictors in predictor matrix X
                % matrix, 
                % ... and get their distance. (means: KxD, data: NxK)
                distance(j) = dist(means(j, :), X(n, 1:D));
            end 

            [val, k] = min(distance);  % et k = index of min element in distance vector
            
            % Assign one since k is assigned to be the same as k-hat = argmin
            % The rest of the locations shall be 0, since r_kn = 1 if k
            % =k-hat and 0 if k != k-hat
            resp(n, k) = 1;  
            % VIEW: resp is NxK matrix with one 1 per row, rest is zeroes. 
        end

        % (3) Update step: model parameters are adjusted to match
        % the sample means of the data points that they are
        % responsible for. 
        
        %stepmeans = means;
        for k = 1:K % for cluster k ...
            for d = 1:D % for dimension d... % Sum the rows on col k of responsibilities * predictor d + 1       
                 % keeping only X row elments with nonzero
                 % responsibilities. 
                means(k, d) = sum(  resp(:, k) .* X(:, d)  ); % data(:, d+1));
            end
            means(k, :) = means(k, :) / sum(resp(:, k)); % means on kth row divided by sum of responsibiliies on kh column
        end
        
        %% THE VECTORIZED VERSION OF ABOVE UPDATE STEP: 
        %stepmeans = (transpose(resp) * X) ./ transpose(sum(resp));
        
    end
    
end 
