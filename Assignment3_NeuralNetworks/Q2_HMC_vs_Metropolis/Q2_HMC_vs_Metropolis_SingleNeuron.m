% load dataset
load toy.dat

[N, I] = size(toy)
X = [ones(N, 1), toy(:, 1:2)] % apply ones on the column to account for biases in weights vector
targets = toy(:, 3) % targets is third column

alpha = 0.01
proposalWidth = 0.1

%% Running the Metropolis and Hamiltonian algorithms
[weightsStored_HMC, weightsIndep_HMC] = hamiltonianMCBayesSingleNeuron(X, targets, alpha);

weightsIndep_HMC

[weightsStored_Met, weightsIndep_Met] = metropolisMCBayesSingleNeuron(X, targets, alpha, proposalWidth);

weightsIndep_Met


%% Plotting Learned Y mean (predicted probabilities) from Metropolis

% Getting the test data
x1 = linspace(0, 10);
x2 = x1; 
[x1 x2] = meshgrid(x1, x2);

testX = [ones(10000, 1), x1(:), x2(:)];

% learned y part : calculate predicted probabilities of targets = 1
% testX = M x I
% weightsIndep = R x I, where R = number of independent samples. 

probTargetIsOne_Met = mean( sigmoid(weightsIndep_Met * testX') );

probTargetIsOne_Met

probTargetIsOne_HMC = mean( sigmoid(weightsIndep_HMC * testX') );

probTargetIsOne_HMC


% Autocorrelations for Metropolis
figure(1); clf
lag = 2000;
acf(weightsStored_Met(:, 2), lag);


% Predictive distribution for Hamiltonian
figure(2); clf
plot(X(1:5, 2), X(1:5, 3), 'r.', 'MarkerSize', 30); % rows 1:5 in data have points belonging to left group
hold on
plot(X(6:10, 2), X(6:10, 3), 'b.', 'MarkerSize', 30) % the rows 6-10 have points belonging to different group
xlim([0 10]);
ylim([0 10]); 
axis square
title('Predictive Distribution - Metropolis'); xlabel('x1'); ylabel('x2')
hold on

% NOTE: x1, x2, probs/learnedYs are all 100 x 100 matrices
% need to reshape so probs are also 100 x 100
probs = reshape(probTargetIsOne_Met, 100, 100); 
contour(x1, x2, probs, 5, '--k', 'LineWidth', 2); 
hold on
contour(x1, x2, probs, 1,'k', 'LineWidth', 3); % [0.5 0.5] in place of 1




% Autocorrelations for Hamiltonian
figure(3); clf
lag = 60;
acf(weightsStored_HMC(:, 2), lag);


% Predictive distribution for Hamiltonian
figure(4); clf
plot(X(1:5, 2), X(1:5, 3), 'r.', 'MarkerSize', 30); % rows 1:5 in data have points belonging to left group
hold on
plot(X(6:10, 2), X(6:10, 3), 'b.', 'MarkerSize', 30) % the rows 6-10 have points belonging to different group
xlim([0 10]);
ylim([0 10]); 
axis square
title('Predictive Distribution - Hamiltonian'); xlabel('x1'); ylabel('x2')
hold on

% NOTE: x1, x2, probs/learnedYs are all 100 x 100 matrices
% need to reshape so probs are also 100 x 100
probs = reshape(probTargetIsOne_HMC, 100, 100); 
contour(x1, x2, probs, 5, '--k', 'LineWidth', 2); 
hold on
contour(x1, x2, probs, 1,'k', 'LineWidth', 3); % [0.5 0.5] in place of 1


