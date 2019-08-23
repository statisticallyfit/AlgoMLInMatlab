function [means, energy] = analyzeKmeans(K, X)
    [means, resp, distance] = kmeans(K, X);
    
    % calculate energy
    % note: resp * dist^T (is same as) dist * resp^T
    energy = sum(resp * transpose(distance)) ;
    % resp = N x K, distance = 1 x K
    
    
    % Plot data with kmeans
    figure(1); clf
    scatter(X(:,1), X(:,2)); %%% note: here assuming only two dimensions in X
    axis square; 
    box on
    hold on
    plot(means(:, 1), means(:, 2), 'ksq', 'markersize', 15, 'Color', 'r')

end