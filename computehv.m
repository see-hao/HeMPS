% function hv = computehv(xtrain, ftrain)

F=ftrain;

r = max(F);

popStruct = getPop(size(xtrain, 1));
popStruct = assignV(xtrain, ftrain, popStruct);
[~, P] = nonDominatedSort(popStruct);


hv = hypervolume(F(P{1},:), r, 1000);

% end