function popb = blx_op(pop, problem)
    % BLX crossover
    nPop = size(pop, 1);
    lb = problem.domain(1,:);
    ub = problem.domain(2,:);
    % BLX & normally distributed mutation parameters
    pCrossover = 0.7;                         % Crossover Percentage
    nCrossover = 2*round(pCrossover*nPop/2);  % Number of Parnets (Offsprings)
    pMutation  = 0.4;                         % Mutation Percentage
    nMutation  = round(pMutation*nPop);       % Number of Mutants
    mu         = 0.02;                        % Mutation Rate
    sigma      = 0.1*(ub-lb);                 % Mutation Step Size

    i1 = randi(nPop, [nCrossover/2, 1]);
    i2 = randi(nPop, [nCrossover/2, 1]);
    p1 = pop(i1, :);
    p2 = pop(i2, :);
    [popc1, popc2] = crossover(p1, p2, lb, ub);
    popc = [popc1; popc2];

    % Mutation        
    i = randi(nPop, [nMutation, 1]);
    p = pop(i, :);
    popm = mutate(p, mu, sigma, lb, ub);
    
    popb = [popc; popm];
    % repair
    popb = max(popb, lb);
    popb = min(popb, ub);
    rng('shuffle');
end