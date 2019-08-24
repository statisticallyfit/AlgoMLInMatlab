K = 2; % number of clusters (number of Gaussians)
D = 2; % number of dimensions in each Gaussian

mu1 = [3, 4];
mu2 = [9, 8];

sigma1 = [1, 0.5; 0.5, 1];
sigma2 = [1, -0.2; -0.2, 1];

phi = [0.3, 0.7];
N = 1000;

%%Step 1: generate sample from the mixture of two bivariate Gaussians
rng('default') % reproducible

X = sampleGaussianMixture(N, {mu1, mu2}, {sigma1, sigma2}, phi);

%%Step 2: implement the EM algorithm and get some results
[MU, SIGMA, PI] = my_em(X, phi, K);

% Plotting the resulting mixture of gaussians: 

% Probability density of mixture using the given parameters: 
%p_initial = @(x) phi(1) * mvnpdf(x, mu1, sigma1) + phi(2) * mvnpdf(x, mu2, sigma2) ;

% PDF of mixture using the parameters estimated by the EM algorithm.  
p_new = @(x) PI(1) * mvnpdf(x, MU(1,:), SIGMA{1}) + PI(2) * mvnpdf(x, MU(2,:), SIGMA{2}) ;


% Plotting the sampled data
figure(1); clf; 
X1 = X(:, 1);
X2 = X(:, 2);
scatter(X1, X2) % the two dimensions sampled from data

hold on; 

% Deterministic part: using contour to plot the mixture of gaussians
% making the grid of values
x1 = linspace(min(X1), max(X1));     % 1 x 100
x2 = linspace(min(X2), max(X2));     % 1 x 100

% joining xx2 on the right of xx1
[xx1 xx2] = meshgrid(x1, x2);        % xx1 is 100 x 100

% get the values from the multivariate pdf using the meshgrid
f = p_new([xx1(:), xx2(:)]);

% contour plot 
% note: must reshape f to be 100x100 since both the  xx1 and xx2 are 100 x 100. 
contour(xx1, xx2, reshape(f, 100, 100))
xlabel('x1') % dimension 1
ylabel('x2') % dimension 2


% Showing the results in a table
% Means comparison
meanTable = table(mu1', MU(1,:)', mu2', MU(2,:)', 'VariableNames', {'GivenMu1', 'EstMu1', 'GivenMu2', 'EstMu2'})

% Sigma comparison
sigma1Table = table(sigma1, SIGMA{1}, 'VariableNames', {'GivenSigma1', 'EstSigma1'})

sigma2Table = table(sigma2, SIGMA{2}, 'VariableNames', {'GivenSigma2', 'EstSigma2'})

 % Prior mixture probabilities comparison
phiTable= table(phi', PI', 'VariableNames', {'GivenPrior', 'EstPrior'})



%%Doing the EM algorithm steps from the start for B = 100 iterations
B = 100;
[avgMU, avgSIGMA, avgPI, allMUs, allSIGMAs, allPIs] = averageEMResults(B, {mu1, mu2}, {sigma1, sigma2}, N, phi);

% Showing that results are consistent (but some outpus are switched around)
allMUs{1:5}

disp('First Cluster Covariance Matrices')
for i = 1:5
    fprintf('i = %d\n', i)
    allSIGMAs{i}{1}
end

disp('Second Cluster Covariance Matrices')
for i = 1:5
    fprintf('i = %d\n', i)
    allSIGMAs{i}{2}
end

allPIs{1:5}


% Compare original vs estimated parameters in a table

% Means comparison
avgMeansTable = table(mu1', avgMU(1,:)', mu2', avgMU(2,:)', 'VariableNames', {'GivenMu1', 'EstMu1', 'GivenMu2', 'EstMu2'})


% Sigma comparison
avgSigma1Table = table(sigma1, avgSIGMA{1}, 'VariableNames', {'GivenSigma1', 'EstSigma1'})

avgSigma12able = table(sigma2, avgSIGMA{2}, 'VariableNames', {'GivenSigma2', 'EstSigma2'})

 % Prior mixture probabilities comparison
 avgPiTable = table(phi', avgPI', 'VariableNames', {'GivenPrior', 'EstPrior'})
