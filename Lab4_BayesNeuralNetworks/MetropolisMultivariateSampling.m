%% PARAMETERS of algorithm: 
% X = data matrix (N x I)
% t = target vector (N x 1)
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

function indepSample = MetropolisMultivariateSampling(X, proposalWidth, Pstar)

    % Define posterior distribution for W
    
    %y = @(w) sigmoid(X * w);  %N x 1
    %G = @(w) -(t' * log(y(w) )  + (1-t)' * log(1 - y(w)) );  % 1x1
    %E = @(w) w' * w / 2;  %sum(W.^2, 2)' / 2  % 1x1
    %M = @(w) G(w) + alpha * E(w) ;
    %Pstar = @(w) exp(-M(w)); % 1x1

    % Metropolis algorithm (lab solution)

    %% STEP 1: initialize values
    lag = 2000;
    R = 30 * lag; 
    burnin = 10000;
    T = burnin + R;
    
    %proposalWidth = 0.1; 
    
    ws = cell(T, 1); %zeros(T, I); 
    numAccepted = 0; 
    ws{1} = [0; 0; 0];   % The xInit (initial value of weights)
    
    [N, I] = size(X); 

    
    % Define proposal distribution and acceptance ratio
    muQ = @(w) w; 
    sigmaQ = diag(proposalWidth * ones(I, 1)); 
    Qsample = @(MU, SIGMA, w) mvnrnd(MU(w), SIGMA); % gaussian with mean at weights value
    
    A = @(wQ, w) Pstar(wQ) / Pstar(w); 

    % Loop T - 1 times
    for t = 1:T-1
        % must transpose qsample since it is rowwise 1 x I
        wQ = transpose(Qsample(muQ, sigmaQ, ws{t})); 
        acceptanceProb = A(wQ, ws{t});

        % Decide if to accept
        unifNum = rand(); %generate uniform random number U between 0 <= U <= 1
        
        % if accept, then the new state is the proposal value
        if acceptanceProb >= 1 || acceptanceProb > unifNum
            ws{t + 1} = wQ;  
            numAccepted = numAccepted + 1; 
        else 
            % else the new weights matrix is same as the old one
            ws{t + 1} = ws{t}; 
        end


    end
    
    % Independent sample is gained by keeping every (lag)th sample, after
    % burnin. 
    indepSample = { ws{burnin + lag: lag : T}  }; 

    acceptanceRate = numAccepted / (T - 1); 
    fprintf('Acceptance rate = %f \n', acceptanceRate);

end