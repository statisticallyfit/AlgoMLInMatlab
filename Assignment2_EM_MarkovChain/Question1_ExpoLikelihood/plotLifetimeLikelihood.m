%% x = data point array
%% theta = the parameter of the likelihood
%%% X = the lifetime random variable of the electronic component

function plotLifetimeLikelihood(x, thetaMin, thetaMax, color)

    % likelihood function as a function of theta (t)
    % so x = fixed
    lik = @(theta) (1 ./ theta .^2) .* x .* exp(-x ./ theta);
    %p = @(lambda) (1 ./ lambda) .* exp(-x ./ lambda) ./ (exp(-1 ./ lambda) - exp(-20 ./ lambda) );
    if x <= 0
        lik = 0;
        %p = 0;
    end
    
    figure(1); 
    % Generate theta values
    thetas = thetaMin : 0.01 : thetaMax; 
    %lambdas = 0 : 0.01 : 100; 
    plot(thetas, lik(thetas), 'b', 'LineWidth', 2, 'Color', color)
    %plot(lambdas, lik(lambdas), 'b', 'LineWidth', 2)
    
    
    
end