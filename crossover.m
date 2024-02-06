function [y1, y2] = crossover(x1, x2, lb, ub)

    alpha = rand(size(x1));
    
    y1 = alpha .* x1 + (1 - alpha) .* x2;
    y2 = alpha .* x2 + (1 - alpha) .* x1;

    %repair the new value.
    y1 = max(y1, lb);
    y1 = min(y1, ub);
    y2 = max(y2, lb);
    y2 = min(y2, ub);
    
end