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
% eta
% X, t
% I

% GOAL: using batch learning, BAYESIAN gradient descent to compute the
% optimal weights of the single neuron network. 
% The only update from single neuron descent algo is the weight
% regularization step in the delta W step. 

% The result is W_MAP (the weights maximized under the posterior)

function wMAP = gradDescentBayesSingleNeuron(X, t, eta, alpha)

    %% STEP 1: Initialization
    [~, I] = size(X);
    
    w = [0, 1, 1]'; 
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
        e = y - t; % t = N x 1, y is N x 1 vector
        
        % compute gradient of object function G(w)
        gradG = transpose(X) * e;  
        
        % Compute gradient of new objective function from weight-decay regularization: M(w) 
        gradM = gradG + alpha * w;
        
        % Now the change in W is this: -eta*(gradG + alpha * w)
        deltaW = - eta * gradM;  
        
        % Updating the weights (and bias)
        w = w + deltaW;
    end 
    
    wMAP = w; 
    
end
