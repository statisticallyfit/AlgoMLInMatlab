% Declaring the values of x1, x2, x3
x1 = 120;
x2 = 130 ;
x3 = 128;

% part (a)
% Likelihood plot for x1
plotLifetimeLikelihood(x1, 0, 700, 'r');


% part (b) All likelihood plots together
plotAllLifetimeLikelihoods(x1, x2, x3, 0, 700)
