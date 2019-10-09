%load toy2.dat
N = 4 % number of data points

I = 3 % dimension of data (dimension of each data point) (it is 2 but 

% here setting it to 3 to account for weights biases)
% Data points
xp1 = [1 4]; % red (class 0)
xp2 = [3 2]; % blue (class 1)
xp3 = [3 5]; % blue (class 1)
xp4 = [7 3] ;% red (class 0)
% the N x I data matrix
X = [xp2; xp3; xp1; xp4];
X = [ones(1, N); X']' % adding ones to account for weight biases

% Set class 1 = blue, class 0 = red
targets = [0 1 1 0]'; % the N x 1 vector
% Plot the data
figure(1); clf
plot(X(1:2, 2), X(1:2, 3), 'b.', 'MarkerSize', 30); % rows 1:5 in data have points belonging to left group
hold on
plot(X(3:4, 2), X(3:4, 3), 'r.', 'MarkerSize', 30) % the rows 6-10 have points belonging to different group
xlim([0 8]); ylim([0 8]); 
xlabel('x1'); ylabel('x2')
axis square
hold on


%% Doing error  backpropagation
numHiddenUnits = 30

[Whidden, Wout] = errorBackpropTwoLayer(X, numHiddenUnits, targets);


%% Calculate learned values
learnedY = @(X) sigmoid(Wout * [ones(1, 10000); tanh(Whidden * X')]) ; %yFunc(Whidden, Wout, 10000);
x1 = linspace(0, N);
x2 = x1; 
[x1 x2] = meshgrid(x1, x2);
xs = [ones(10000, 1), x1(:), x2(:)];
learnedY_values = learnedY(xs);

%% Plot predictions
figure(1); clf
plot(X(1:2, 2), X(1:2, 3), 'b.', 'MarkerSize', 30); % rows 1:5 in data have points belonging to left group
hold on
plot(X(3:4, 2), X(3:4, 3), 'r.', 'MarkerSize', 30) % the rows 6-10 have points belonging to different group
xlim([0 8]); ylim([0 8]); 
xlabel('x1'); ylabel('x2')
axis square
hold on
contour(x1 ,x2, reshape(learnedY_values, 100, 100), 5, '--k', 'LineWidth', 2); hold on
contour(x1 ,x2, reshape(learnedY_values, 100, 100), 1, 'k', 'LineWidth', 2); 