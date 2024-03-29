function [ pdf ] = gaussianND(X, mu, Sigma)
    %GAUSSIANND 
    %      X - Matrix of data points, one per row.
    %     mu - Row vector for the mean.
    %  Sigma - Covariance matrix.

    % Get the vector length.
    D = size(X, 2);

    % Subtract the mean vector from every data point.
    meanDiff = bsxfun(@minus, X, mu);

    % Calculate the multivariate gaussian.
    pdf = 1 / sqrt((2*pi)^D * det(Sigma)) * exp(-1/2 * sum((meanDiff * inv(Sigma) .* meanDiff), 2));
    %pdf2 = 1 / sqrt((2*pi)^n * det(Sigma)) * exp(-1/2 * sum((meanDiff * inv(Sigma) * meanDiff'), 2));
end

