function pop = getPop(popsize)

    empty_individual.Position         = [];
    empty_individual.Cost             = [];
    empty_individual.Rank             = [];
    empty_individual.DominationSet    = [];
    empty_individual.DominatedCount   = [];
    empty_individual.CrowdingDistance = [];

    pop  = repmat(empty_individual, popsize, 1);

end