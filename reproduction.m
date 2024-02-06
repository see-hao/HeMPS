function offspring = reproduction(parent, proC, disC, proM, disM, lb, ub)

    parent = parent([1:end,1:ceil(end/2)*2-end], :);
    [N, D] = size(parent);
    
    % SBX
    beta   = zeros(N/2, D);
    mu     = rand(N/2, D);
    
    parent1 = parent(1:N/2, :);
    parent2 = parent(N/2+1, :);
    beta(mu<=0.5) = (2 * mu(mu<=0.5)).^(1 / (disC+1));
    beta(mu>0.5)  = (2 - 2*mu(mu>0.5)).^(-1 / (disC+1));
    beta = beta .* (-1).^randi([0,1],N/2,D);
    beta(rand(N/2,D) < 0.5) = 1;
    beta(repmat(rand(N/2,1) > proC, 1, D)) = 1;
    offspring = [(parent1+parent2)/2 + beta.*(parent1-parent2)/2
                 (parent1+parent2)/2 - beta.*(parent1-parent2)/2];
    
    % Polynomial mutation
    LB   = repmat(lb, N, 1);
    UB   = repmat(ub, N, 1);
    Site = rand(N,D) < proM/D;
    mu   = rand(N,D);
    temp = Site & mu<=0.5;
    offspring(temp) = offspring(temp) + (UB(temp)-LB(temp)) .* ((2.*mu(temp)+(1-2.*mu(temp)).*(1-(offspring(temp)-LB(temp))./(UB(temp)-LB(temp))).^(disM+1)).^(1/(disM+1))-1);
    temp = Site & mu>0.5; 
    offspring(temp) = offspring(temp) + (UB(temp)-LB(temp)) .* (1-(2.*(1-mu(temp))+2.*(mu(temp)-0.5).*(1-(UB(temp)-offspring(temp))./(UB(temp)-LB(temp))).^(disM+1)).^(1/(disM+1)));
    
    % repair
    offspring = max(offspring, lb);
    offspring = min(offspring, ub);
    
end