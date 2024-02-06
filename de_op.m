function newpoint = de_op(pop, problem)
    % (DE/rand/1)
    global params;
    lb = problem.domain(1,:);
    ub = problem.domain(2,:);
    
    % DE operators parameters
    deF  = 0.5;
    deCR = 0.5;
    
    idx = randperm(params.popsize);
    idx(idx == 1) = [];
    for i = 2 : params.popsize
        idxtemp = randperm(params.popsize);
        idxtemp(idxtemp == i) = [];
        idx = [idx; idxtemp];
    end
    a = idx(:, 1);
    b = idx(:, 2);
    c = idx(:, 3);

    % Mutation
    newpoint = pop(a,:) + deF*(pop(b,:) - pop(c,:));
    % Crossover
    jrandom             = ceil(rand(params.popsize,1) * problem.pd);
    randomarray         = rand(params.popsize, problem.pd);
    deselect            = randomarray < deCR;
    linearInd           = sub2ind(size(deselect),1:params.popsize,jrandom');
    deselect(linearInd) = true;
    newpoint(~deselect) = pop(~deselect);

    % repair
    newpoint = max(newpoint, lb);
    newpoint = min(newpoint, ub);
    rng('shuffle');
end