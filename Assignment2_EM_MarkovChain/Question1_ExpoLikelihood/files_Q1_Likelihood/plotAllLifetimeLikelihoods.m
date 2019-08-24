

function plotAllLifetimeLikelihoods(x1, x2, x3, thetaMin, thetaMax)

    % Generate theta values
    thetas = thetaMin : 0.01 : thetaMax; 
    
    % likelihood function as a function of theta (t)
    % so x = fixed
    lik = @(theta, x) (1 ./ theta .^2) .* x .* exp(-x ./ theta);
    
    % Plotting the likelihood for a single given 'x'
    figure(2); clf;
    plot(thetas, lik(thetas, x1), 'r', 'LineWidth', 2)
    hold on;
    plot(thetas, lik(thetas, x2), 'b', 'LineWidth', 2)
    plot(thetas, lik(thetas, x3), 'g', 'LineWidth', 2)
        
    
    % vertical dotted lines at theta = 60, 65, 64
    thetaCritical = [x1 x2 x3]/2;

    ys = 0 : 0.00001 : 0.005;
    for i = 1:length(thetaCritical)
        hold on;
        xs = thetaCritical(i) * ones(1, length(ys));
        plot(xs, ys, 'LineStyle', '--');
    end
    
    hold on; 
    xlabel('\theta');
    ylabel('Likelihood')
    
    legend({['P(x1 = ', num2str(x1), ' | theta)'], ...
        ['P(x2 = ', num2str(x2), ' | theta)'], ...
        ['P(x3 = ', num2str(x3), ' | theta)'], ...
        ['theta = ', num2str(x1/2)], ['theta = ', num2str(x2/2)], ['theta = ', num2str(x3/2)]})
end