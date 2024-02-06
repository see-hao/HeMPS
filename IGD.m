function score = IGD(Population,optimum)
% <min>
% Inverted generational distance
    
    PopObj = Population;
    if size(PopObj,2) ~= size(optimum,2)
        score = nan;
    else
        score = mean(min(pdist2(optimum,PopObj),[],2));
    end
end