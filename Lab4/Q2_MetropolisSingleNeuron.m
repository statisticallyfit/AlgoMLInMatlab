%addpath(genpath('/datascience/projects/statisticallyfit/github/learningmathstat/AlgoMLInMatlab/Assignment2_EM_MarkovChain/Question3_MetropolisBayes/' )); %MetropolisSampling.m'))
% my metropolis algo is is not multivariate; need m ultivariate here below.



% load dataset
load toy.dat
[N, I] = size(toy)
X = [ones(N, 1), toy(:, 1:2)]
t = toy(:, 3) % targets is third column



% Define posterior distribution for W
alpha = 0.01
y = @(W) sigmoid(X * W)  %N x 1
G = @(W) -(t' * log(y(W) )  + (1-t)' * log(1 - y(W)) )  % 1x1
E = @(W) W'*W / 2  %sum(W.^2, 2)' / 2  % 1x1
M = @(W) G(W) + alpha * E(W) 
Pstar = @(W) exp(-M(W)) % 1x1
proposalWidth = 0.1

[ws1, windep1] = tempmetropneuron(X, t, proposalWidth);

ws1

windep1

windep2 = tempschool(X, t, proposalWidth);

windep2 

[ws3, windep3] = MetropolisMultivariateSampling(X, proposalWidth, Pstar);

ws3{:}








learnedY = @(x) zeros(1, length(x))

for i = 1:length(Windep)
    W = Windep(i, :) % ith row
    %lY = sigmf(W * x', [1, 0])
    learnedY = @(x) [learnedY(x); sigmf(W * x', [1,0])]
end

learnedY = @(x) sum(learnedY(x)) / length(Windep)
