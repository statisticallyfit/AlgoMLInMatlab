%% x = data point array
%% theta = the parameter of the likelihood
%%% X = the lifetime random variable of the electronic component

function plotLifetimeLikelihood(x, thetaMin, thetaMax, color)

    % Generate theta values
    thetas = thetaMin : 0.01 : thetaMax; 
    
    % likelihood function as a function of theta (t)
    % so x = fixed
    lik = @(theta) (1 ./ theta .^2) .* x .* exp(-x ./ theta);
    likValues = lik(thetas);
    
    if x <= 0
        likValues = zeros(1, length(thetas));
    end
    
    % Plotting the likelihood for a single given 'x'
    figure(1); clf;
    plot(thetas, lik(thetas), 'b', 'LineWidth', 2, 'Color', color)
    
    xlabel('\theta');
    ylabel('Likelihood')
    
    legend(['P(x = ', num2str(x), ' | theta)'])
    
    
end