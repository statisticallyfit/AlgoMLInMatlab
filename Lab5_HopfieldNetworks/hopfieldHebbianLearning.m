% GOAL: Weights optimizer using Hebbian Learning for Hopfield Neural
% Network 


% PARAMETERS of algorithm: 
% eta = learning rate (hyperparameter)
% I = number of neurons
% N = number of data samples. 
% X = the N x I matrix containing N input vectors on the rows, each of length I
% w = weights matrix of size I x I


% ----------------------------------------------------------
% ARGUMENTS of function: 
% eta
% X


function W = hopfieldHebbianLearning(X, eta)

    %% Initialization
    [N, I] = size(X);
    
    W = X' * X; % initialize the weights using Hebb rule (then W is symmetric)
       
    % Initialize x to a corrupted memory (the exercise)
    %n = 1; % memory number
%    x = X(n, :)';
    % Corrupting some random bits
    %x(16, 1) = -x(16, 1);
    %x(2,1)  = -x(2,1);
    %x(5,1)  = -x(5,1);
    %x(20,1)  = -x(20,1);
    %x(13,1)  = -x(13,1);
        
    NUM_ITER = 10; 
    
    for iter = 1: NUM_ITER   
      
        % Ensuring the self-weights are zero: (correct step?)
        for i = 1:I
            W(i, i) = 0; 
        end
        
        %% Activity rule
        
        % Compute all the activations: a_i = sigma(W_ij * x_j)
        A = X * W;   %X = NxI, W = IxI, so X*W = NxI
        %A = W * x;
        
        % Neurons update their states SYNCHRONOUSLY to: x_i = thresh(a_i)
        X = thresh(A);
        %x = thresh(A);
        
        
        %% Learning rule: 
        W = eta * (X' * X);  %paranthesizing x^T * x to ensure result is hermitian
        %W = eta * x * x';
    end
    
    %colormap gray; 
    %cmap = colormap;    
    %cmap = flipud(cmap); 
    %colormap(cmap);
    %figure(3)
    %imagesc(reshape(x, 5, 5)');
    %axis square
    
end