%addpath(genpath('/datascience/projects/statisticallyfit/github/learningmathstat/AlgoMLInMatlab/Assignment2_EM_MarkovChain/Question3_MetropolisBayes/' )); %MetropolisSampling.m'))
% my metropolis algo is is not multivariate; need m ultivariate here below.



% load dataset
load toy.dat
[N, I] = size(toy)
X = [ones(N, 1), toy(:, 1:2)]
targets = toy(:, 3) % targets is third column

proposalWidth = 0.1


% Define posterior distribution for W
y = @(W) sigmoid(X * W);  %N x 1
G = @(W) -(targets' * log(y(W) )  + (1-targets)' * log(1 - y(W)) );  % 1x1
E = @(W) W'*W / 2;  %sum(W.^2, 2)' / 2  % 1x1
M = @(W) G(W) + alpha * E(W) ;
Pstar = @(W) exp(-M(W)); % 1x1


[wstored1, windep1] = metroparraycolwise(X, proposalWidth, Pstar);
windep1(1:5, );

[wstored2, windep2] = metroparrayrowwise(X, targets, proposalWidth);
windep2(1:5, );

[wstored3, windep3] = MetropolisMultivariateSampling(X, proposalWidth, Pstar);
windep3{1:5}




% Sum the sampled output functions to find average neuron output. 
% This is the expected value of posterior of weights. 
% This is equal to probability that targets = 1. 

learnedY = @(x) zeros(1, length(x))

for i = 1:length(windep2) % num rows
    W = windep2(i, :); % ith row
    %lY = sigmf(W * x', [1, 0])
    learnedY = @(x) [learnedY(x); sigmf(W * x', [1,0])];
end

learnedY = @(x) sum(learnedY(x)) / length(windep2)


% Plotting
figure(1); clf

% Sample autocorrelation plot
subplot(1, 2, 1)
lag = 2000;
acf(wstored2(:, 2), lag);

% Predictive distribution plot
%subplot(1,2,2)
figure(2); clf
plot(X(1:5, 2), X(1:5, 3), 'r.', 'MarkerSize', 30); % rows 1:5 in data have points belonging to left group
hold on
plot(X(6:10, 2), X(6:10, 3), 'b.', 'MarkerSize', 30) % the rows 6-10 have points belonging to different group
xlim([0 10]);
ylim([0 10]); 
axis square
title('Predictive Distribution'); xlabel('x1'); ylabel('x2')
hold on

x1 = linspace(0, 10);
x2 = x1; 
[x1 x2] = meshgrid(x1, x2);

xs = [ones(10000, 1), x1(:), x2(:)];
learnedY_values = reshape(learnedY(xs), 100, 100);
%contour(x1, x2, learnedY_values, [0.12, 0.27, 0.73, 0.88], '--k'); 
contour(x1, x2, learnedY_values, 5, '--k', 'LineWidth', 2); 
hold on
contour(x1, x2, learnedY_values, [0.5 0.5], 'k', 'LineWidth', 3);