% Declare the parameter values
a = 12; % prior parameter alpha
b = 12; % prior parameter beta
nH = 12; % number of heads in N tosses
N = 50; % number of total tosses

% The posterior parameters: 
aNew = a + nH;
bNew = N - nH + b ;


% This is the numerator of the Beta(a + nH, N - nH + b) distribution, from
% which we sample, assuming the denominator is intractable. 
Pstar = @(f) f .^ (a + nH - 1) .* (1 - f).^(N - nH + b-1) ;

% Parameters to metropolis algorithm

% proposal width (standard deviation) of the proposal distribution
proposalWidth = 0.025; 
% L = lengthscale of Pstar distribution 
L = 1;
xInit = 0.5; % choose this so that we start in the middle of the
% posterior for f, since f is between 0 and 1. 


rng('default'); % for reproducibility

% Sample to get the markov chain from the posterior.
X = my_metropolis(10000, L, proposalWidth, xInit, Pstar);


% part (b) PLotting the prior with the posterior

figure(1); clf;

xValues = 0:0.01:1;

% Posterior: Beta(aNew, bNew) = Beta(24, 50)
plot(xValues, betapdf(xValues, aNew, bNew), 'Color', 'b', 'LineWidth', 3)
hold on; 
% Prior : Beta(a = 12, b = 12)
plot(xValues, betapdf(xValues, a, b), 'Color', 'g', 'LineWidth', 2, 'LineStyle', '--')


xlabel('Coin Bias (f)');
ylabel('Probability Density');
title('Posterior, and Prior');

legend({'Posterior', 'Prior'});



% part (c) Plotting all three densities together. 

% Normalizing the sampled MCMC values so that the distributions match up on
% the same y-scale: 

figure(2); clf; 

plot(xValues, betapdf(xValues, aNew, bNew), 'Color', 'b', 'LineWidth', 3)
hold on; 
% Prior : Beta(a = 12, b = 12)
plot(xValues, betapdf(xValues, a, b), 'Color', 'g', 'LineWidth', 2, 'LineStyle', '--')

normalizedPosterior = Pstar(xValues) / beta(a + nH, N - nH + b);
plot(xValues, normalizedPosterior, 'Color', 'r', 'LineWidth', 5, 'LineStyle', '--')

xlabel('Coin Bias (f)');
ylabel('Probability Density');
title('Posterior, Prior, and MCMC-Sample');

legend({'Posterior', 'Prior', 'MCMC-Sampled Posterior'})