N = 3 % number of data points
I = 3 % dimension of data (dimension of each data point) (it is 2 but 
% here setting it to 3 to account for weights biases)

% initial weights
initialWeights = [0 1 1]'; 

% Data points
xp1 = [1 2]; % red (class 0)
xp2 = [2 3]; % red (class 0)
xp3 = [5 4]; % blue (class 1)

% the N x I data matrix
X = [xp1; xp2; xp3];
X = [ones(1, N); X']' % adding ones to account for weight biases

% Set class 1 = blue, class 0 = red
targets = [0 0 1]' % the N x 1 vector

proposalWidth = 0.1
alpha = 0.01
eta = 0.01

% Plot the data
figure(1); clf
plot(X(1:2, 2), X(1:2, 3), 'b.', 'MarkerSize', 30); % rows 1:5 in data have points belonging to left group
hold on
plot(X(3, 2), X(3, 3), 'r.', 'MarkerSize', 30) % the rows 6-10 have points belonging to different group
xlim([0 6]); ylim([0 7]); 
xlabel('x1'); ylabel('x2')
axis square

%% Calculates the optimal weights value. 
wMAP = gradDescentBayesSingleNeuron(X, initialWeights, targets, eta, alpha)

%%Initial probabilities of being in class 1
y = @(W) sigmoid(X * W)  %N x 1

y(initialWeights) % this is the initial probabilities of inputs belonging to class 1

%% Probabilities that inputs belong to class 1

y(wMAP)

