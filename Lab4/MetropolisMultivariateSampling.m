%% PARAMETERS of algorithm: 
% X = data matrix (N x I)
% targets = target vector (N x 1)
% N = number of samples
% I = number of neuron inputs
% alpha = weight decay learning rate (hyperparameter)

%% ARGUMENTS to function: 
% X
% t
% alpha, eta

%% OUTPUT of function: 
% weightIndepSample = the T x 1 cell array of I x 1 independent weight vectors sampled 
% from the posterior numerator (Pstar) of weights. 

%% GOAL: sample the weights from the posterior of the weights using
% Metropolis, then average results to get an estimate that takes the WHOLE
% posterior distribution into account, rather than a single optimized
% weight value w_MAP under posterior (like in the other single neuron bayes
% file)
% This is a Bayesian approach because we take the whole posterior
% distribution of the weights into account, rather than computing a single
% optimized value for weights, w_MAP. 

function [Wstored, indepSample] = MetropolisMultivariateSampling(X, targets)

    rng('default');
    
    
    % Define posterior distribution for W
    
    % Define posterior distribution for W
    alpha = 0.01;
    proposalWidth = 0.1; 
    y = @(W) sigmoid(X * W);       % N x 1
    G = @(W) -(targets' * log(y(W) )  + (1-targets)' * log(1 - y(W)) ) ; % 1x1
    E = @(W) W'*W / 2;             % 1x1         %sum(W.^2, 2)' / 2  
    M = @(W) G(W) + alpha * E(W) ; % 1x1
    Pstar = @(W) exp(-M(W));       % 1x1

    % Metropolis algorithm (lab solution)

    %% STEP 1: initialize values
    lag = 2000;
    R = 30 * lag; 
    burnin = 10000;
    T = burnin + R;
    [N, I] = size(X); 
    
    
    Wstored = zeros(T, I); 
    Wstored(1, :) = [0, 0, 0];
    W = repmat({ zeros(I, 1) }, T, 1); %cell(T, 1); %zeros(T, I);
    
    % Define proposal distribution and acceptance ratio
    muQ = @(w) w; 
    sigmaQ = diag(proposalWidth * ones(I, 1)); 
    Qsample = @(MU, SIGMA) mvnrnd(MU, SIGMA); % gaussian with mean at weights value
    
    A = @(wQ, w) Pstar(wQ) / Pstar(w); 

    
    
    numAccepted = 0; 
    accept = 0;
    
    % Loop T - 1 times
    for t = 1:T-1
        % must transpose qsample since it is rowwise 1 x I
        
        %wQ = transpose(   Qsample(  muQ(W{t}), sigmaQ)   ); 
        %acceptanceProb = A(wQ, W{t});
        wQ = transpose(   Qsample(  muQ(Wstored(t, :)' ), sigmaQ)   ); 
        acceptanceProb = A(wQ, Wstored(t, :)' );

        % Decide if to accept
        %unifNum = rand(); %generate uniform random number U between 0 <= U <= 1
        
        
        if acceptanceProb >= 1
            accept = 1;
        elseif accept > rand()
            accept = 1;
        else 
            accept = 0;
        end
        
        if accept
            W{t + 1} = wQ;
            Wstored(t + 1, :) = wQ'; 
            
            numAccepted = numAccepted + accept;
        else 
            W{t + 1} = W{t}; % old state
            Wstored(t+1, :) = Wstored(t, :);
        end
        % if accept, then the new state is the proposal value
        %if acceptanceProb >= 1 || acceptanceProb > unifNum
        %    Wstored{t + 1} = wQ;  
        %    numAccepted = numAccepted + 1; 
        %else 
        %    % else the new weights matrix is same as the old one
        %    Wstored{t + 1} = Wstored{t}; 
        %end


    end
    
    % Independent sample is gained by keeping every (lag)th sample, after
    % burnin. 
    
    indepSample = Wstored(burnin + lag : lag : T, :);
    %indepSample = { W{burnin + lag: lag : T}  }; 

    acceptanceRate = numAccepted / (T - 1); 
    fprintf('Acceptance rate = %f \n', acceptanceRate);

end