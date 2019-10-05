function [weights, weightsIndep] = MetropolisMultivariateSampling(X, Pstar)
    % Metropolis algorithm (lab solution)
    % note: I adapted for W to be I x 1 vector not 1 x I as in lab solution

    rng('default')

    %% STEP 1: initialize values
    lag = 2000;
    burnin = 10000;
    T = burnin + 30*lag;
    
    %alpha = 0.01; 
    proposalWidth = 0.1; 
    numAccepted = 0; 
    
    [N, I] = size(X); 
    
    weights = repmat({ zeros(I, 1) }, T, 1);
    W = [0; 0; 0];  % columnwise I x 1 vector of weights


    % Define proposal distribution and acceptance ratio
    muQ = @(w) w; 
    sigmaQ = diag(proposalWidth * ones(I, 1)); 
    
    % This is the proposal distribution
    Qsample = @(MU, SIGMA) mvnrnd(MU, SIGMA); % gaussian with mean at weights value
    % Method for calculating acceptance ratio. 
    A = @(wQ, w) Pstar(wQ) / Pstar(w); 

    
    for t = 1:T-1
        % must transpose qsample since mvnrnd() produces vector rowwise 1 x I
        wQ = transpose( Qsample(  muQ(W) , sigmaQ) );  
        acceptProb = A(wQ, W );

        % Decide if to accept
        if acceptProb >= 1
            accept = 1; 
        elseif acceptProb > rand()
            accept = 1; 
        else 
            accept = 0; 
        end

        if accept
            W = wQ;
        end

        numAccepted = numAccepted + accept; 

        weights{t + 1} = W; % even if W is not updated to wQ, still put the old value here for storage.

    end

    acceptanceRate = numAccepted / (T - 1); 
    fprintf('Acceptance rate = %f \n', acceptanceRate);
    
    
    % Selecting independent sample: starting from burning+lag, keep every lag(th)
    % sample until we hit T
    weightsIndep = { weights{burnin + lag: lag : T}  };
end