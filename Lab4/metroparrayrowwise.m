function [Wstored, Windep] = metroparrayrowwise(X, t, proposalWidth)
    % Metropolis algorithm (lab solution)
    % note: I adapted for W to be I x 1 vector not 1 x I as in lab solution

    rng('default')

    % Define posterior distribution for W
    %alpha = 0.01;
    y = @(W) sigmoid(W * X');  %N x 1
    G = @(W) -(t' * log(y(W)' )  + (1-t') * log(1 - y(W))' );  % 1x1
    E = @(W) sum(W .^ 2) / 2;  %sum(W.^2, 2)' / 2  % 1x1
    M = @(W) G(W) + alpha * E(W) ;
    Pstar = @(W) exp(-M(W)); % 1x1

    
    [N, I] = size(X); 

    %% STEP 1: initialize values
    lag = 2000;
    burnin = 10000;
    T = burnin + 30*lag;
    %proposalWidth = 0.1; 
    Wstored = zeros(T, I); 
    numAccepted = 0; 
    W = [0 0 0];  % row-wise 1 x I vector of weights
    Wstored(1, :) = W; 


    % Define proposal distribution and acceptance ratio

    % mean = W, sigma = diag of proposal size for gaussian
    Qsample = @(W) mvnrnd(W, diag(proposalWidth * ones(I, 1)));
    A = @(Wprime, W) Pstar(Wprime) / Pstar(W); 

    % Loop T - 1 times
    for i = 1:T-1
        % must transpose qsample since it is rowwise 1 x I
        Wprime = Qsample(W);  
        Avalue = A(Wprime, W);

        % Decide if to accept
        if Avalue >= 1
            accept = 1; 
        elseif Avalue > rand()
            accept = 1; 
        else 
            accept = 0; 
        end

        if accept
            W = Wprime;
        end

        numAccepted = numAccepted + accept; 

        Wstored(i+1, :) = W ; 

    end

    acceptanceRate = numAccepted / (T - 1)


    % Bayesian part: sum the sampled output functions
    % to find average neuron output

    % note: starting from burning+lag, keep every lag(th)
    % sample until we hit T
    Windep = Wstored(burnin + lag : lag : T, :);
end