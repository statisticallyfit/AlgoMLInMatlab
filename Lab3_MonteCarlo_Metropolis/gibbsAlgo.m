% Gibbs Algoritm: 

function [mu, lambda] = gibbsAlgo(N, X)

    % Prior parameters --- can be left to experimentation. 
    mu0 = 0; 
    sigma0 = sqrt(2);
    a = 1; 
    b = 1;

    % Posterior distributions of mu and lambda are: 
    % for mu: mu ~ Normal(muN, sigmaN^2)
    % for lambda: lambda ~ Gamma(aN, bN), 

    muN = @(lambda) (mu0 + N * lambda * sigma0^2 * mean(X))/(1 + N * lambda * sigma0^2);
    lambdaN = @(lambda) sigma0^2 / (1 + N*lambda * sigma0^2);

    % note: need mu arg since these are the parameters of 
    % the LAMBDA posterior.
    aN = a + N/2;
    bN = @(mu) 1/(b + (1/2) * sum((X - mu) .^ 2) );

    % Defining the random number generators of the conditional
    % posterior distributions of mu and lambda
    posteriorMu = @(lambda) normrnd(muN(lambda), lambdaN(lambda));
    posteriorLambda = @(mu) gamrnd(aN, bN(mu));

    % (1) Initialize: 
    T = 1000; 
    mu = zeros(1, T);
    lambda = zeros(1, T);
    mu(1) = 1.15; 
    lambda(1) = 0.7; % ????

    % (2) Generate new state: 
    % loop T times: 
    for t = 1:T-1
        mu(t+1) = posteriorMu(lambda(t));
        lambda(t+1) = posteriorLambda(mu(t+1));
    end
    
    
end