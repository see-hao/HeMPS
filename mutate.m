function y = mutate(x, mu, sigma, lb, ub)

    nVar = size(x, 2);
    nMu = ceil(mu * nVar);
    j = randi(nVar, [size(x, 1), nMu]);
    if numel(sigma) > 1
        sigma = sigma(j);
    end
    y = x;
    if nMu > 1
        y(j) = x(j) + sigma .* randn(size(x, 1), nMu);
    else
        y(:, j) = x(:, j) + sigma .* randn(size(x, 1), nMu);
    end
    
    %repair the new value.
    y = max(y, lb);
    y = min(y, ub);

end