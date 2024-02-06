function offspring = sbx_op(pop, problem)
    % SBX & polynomial mutation parameters
    proC = 1;    % crossover probability
    disC = 20;   % distribution index of sbx
    proM = 1;    % expectation of number of bits doing mutation
    disM = 20;   % distribution index of polynomial mutation
    lb         = problem.domain(1,:);
    ub         = problem.domain(2,:);

    % SBX & polynomial mutation
    offspring = reproduction(pop, proC, disC, proM, disM, lb, ub);

    % repair
    offspring = max(offspring, lb);
    offspring = min(offspring, ub);
    rng('shuffle');
end