close all
clear 
clc

format long;
format compact;

dimension      = [8, 10, 12, 14, 16, 18, 20];%维数
popsize        = 20;%种群大小
totalrun       = 3;%跑3次
currentFolder = pwd;
src_model = loadSourceData();
datadir = strcat(currentFolder, '\data.mat');
for i=4%依次检验问题
    problem = ['dtlz' num2str(i)];
    fprintf('Running on %s...\n', problem);
    xtrain = lhsdesign(popsize, 10);
        for j = 1:7%跑1次
            sop               = testmop(problem,10, 3);%sop为测试问题函数
            sop.popsize       = popsize;%问题规模
            ftrain = sop.func(xtrain);
            src_models = src_model(((i-1)*21+(j-1)*3+1) : ((i-1)*21+(j-1)*3+3));
%             src_models = src_model;
            [~, ~]       = temomps(sop, src_models, xtrain, ftrain);
%               [~, ~]       = temomps(sop, src_models, xtrain, ftrain, j);
%             [pop, objs, newpoints]       = temomps(sop);
%             popStruct = getPop(size(pop, 1));
%             popStruct = assignV(pop, objs, popStruct);
%             [~, F] = nonDominatedSort(popStruct);
%             x = pop(F{1},:);
%             y = objs(F{1},:);
        end
end
load chirp
sound(y,Fs)