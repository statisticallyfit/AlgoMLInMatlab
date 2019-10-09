% PARAMETERS of algorithm: 
% eta = learning rate (hyperparameter)
% alpha = weight decay rate
% I = number of weights
% N = number of data samples. 
% X = the N x I matrix containing N input vectors on the rows, each of length I
% t = target vector of length N, containing all the targets, corresponding
% to each sample (n). 
% w = weights vector of size I x 1 (so is column vector)


% ----------------------------------------------------------
% ARGUMENTS of function: 
% X = data set , N x I matrix
% initialWeights: I x 1 vector of weights
% targets = N x 1 vector of classes (0 or 1)
% eta = for calculating gradG
% alpha = for calculating gradM


% GOAL: using batch learning, BAYESIAN gradient descent to compute the
% optimal weights of the single neuron network. 
% The only update from single neuron descent algo is the weight
% regularization step in the delta W step. 

% The result is W_MAP (the weights maximized under the posterior)

function wMAP = gradDescentBayesSingleNeuron(X, initialWeights, targets, eta, alpha)

    %% STEP 1: Initialization
    [~, I] = size(X);
    
    w = initialWeights;  % setting initial weights manually
    %w = rand(I, 1); 
    
        
    NUM_ITER = 50000; 
    
    for iter = 1: NUM_ITER   
      
        %% STEP 2: Activity rule
        
        % Compute all the activations: a = sigma(i -> I) w_i * x)i (n)
        a = X * w; 
        
        % Find neuron's output: 
        y = sigmoid(a);
        
        %% STEP 3: Learning rule: for each target value t(n), find the error signal
        % and adjust the weights: 
        
        e = y - targets;                % t = N x 1, y is N x 1 vector
        gradG = X' * e;                 % compute gradient of object function G(w)
        gradM = gradG + alpha * w;      % Compute gradient of new objective function from weight-decay regularization: M(w) 
        deltaW = - eta * gradM;         % change in W
        
        % Updating the weights (and bias)
        w = w + deltaW;
    end 
    
    wMAP = w; 
    
end
