% GOAL: Weights optimizer using Gradient Descent for Hopfield Neural
% Network (better than Hebbian, by page 516 in Mackay)


% PARAMETERS of algorithm: 
% eta = learning rate (hyperparameter)
% alpha = weight decay ???????
% I = number of neurons
% N = number of data samples. 
% X = the N x I matrix containing N input vectors on the rows, each of length I
% t = target matrix size I x I, same as X but all (-1) are replaced by (0)
% w = weights matrix of size I x I


% ----------------------------------------------------------
% ARGUMENTS of function: 
% eta, alpha
% X
% T

% GOAL: using batch learning, simple gradient descent to compute the
% optimal weights of the Hopfield network. 
function W = hopfieldGradientDescent(X, T, eta, alpha)

    %% STEP 1: Initialization
    [N, I] = size(X);
    
    W = X' * X; % initialize the weights using Hebb rule (then W is symmetric)
       
        
    NUM_ITER = 10; 
    
    for iter = 1: NUM_ITER   
      
        % Ensuring the self-weights are zero: 
        for i = 1:I
            W(i, i) = 0; 
        end
        
        %% STEP 2: Activity rule
        
        % Compute all the activations: a_i = sigma(W_ij * x_j)
        a = X * W;   %X = NxI, W = IxI, so X*W = NxI
        
        % Find all neurons' output: 
        Y = sigmoid(a); % a is N x I
        
        %% STEP 3: Learning rule: 
        % and adjust the weights: 
        E = T - Y; % E = N x I
        
        % compute gradient of object function G(w)
        gradG = X' * E;  
        gradG = gradG + gradG'; % symmetrizing gradients for Hopfield network requirement
        
        % Compute gradient of new objective function from weight-decay regularization: M(w) 
        %gradM = gradG - alpha * W; % make step
        
        % Now the change in W is this: -eta*(gradG + alpha * w)
        %deltaW = eta * gradM;  
        
        % Updating the weights (and bias)
        %W = W + deltaW;
        W = W + eta * (gradG - alpha*W);
    end
    
end