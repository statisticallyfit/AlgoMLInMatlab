% X = data matrix (N x I)
% t = target vector (N x 1)
% N = number of samples
% I = number of neuron inputs
% alpha = weight decay learning rate (hyperparameter)

function weightSample = singleNeuronBayesMetropolis(X, t, alpha)

    % Define posterior distribution for W
    
    y = @(w) sigmoid(X * w);  %N x 1
    G = @(w) -(t' * log(y(w) )  + (1-t)' * log(1 - y(w)) );  % 1x1
    E = @(w) w' * w / 2;  %sum(W.^2, 2)' / 2  % 1x1
    M = @(w) G(w) + alpha * E(w) ;
    Pstar = @(w) exp(-M(w)); % 1x1

    % Metropolis algorithm (lab solution)

    %% STEP 1: initialize values
    lag = 2000;
    burnin = 10000;
    T = burnin + 30*lag;
    proposalWidth = 0.1; 
    
    weightSample = zeros(T, I); 
    numAccepted = 0; 
    w = [0; 0; 0];  % columnwise I x 1 vector of weights
    weightSample(1, :) = w' ; 

    % Define proposal distribution and acceptance ratio

    % mean = W, sigma = diag of proposal size for gaussian
    mu = w; 
    sigma = diag(proposalWidth * ones(I, 1)); 
    Qsample = @(w) mvnrnd(mu, sigma);
    
    A = @(wProposal, w) Pstar(wProposal) / Pstar(w); 

    % Loop T - 1 times
    for i = 1:T-1
        % must transpose qsample since it is rowwise 1 x I
        Wprime = Qsample(w)'; 
        Avalue = A(Wprime, w);

        % Decide if to accept
        if Avalue >= 1
            accept = 1; 
        elseif Avalue > rand()
            accept = 1; 
        else 
            accept = 0; 
        end

        if accept
            w = Wprime;
        end

        numAccepted = numAccepted + accept; 

        weightSample(i+1, :) = w' ; 

    end

acceptanceRate = numAccepted / (T - 1); 