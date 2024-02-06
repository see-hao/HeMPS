function [xtrain, ftrain] = temomps(problem, src_models, x, y)
    global nFEs params ;
    % set the algorithm parameters
%     currentFolder = pwd;
%     datadir = strcat(currentFolder, '\result\dtlz6_16.mat');
    pd = size(src_models{1,1}.X,2);
%     types = ['+', '-'];
    datadir = ['result\' problem.name '-' num2str(pd)  '.mat'];%文件名称
%     datadir = ['result\' problem.name types(type) 'IMS.mat'];%文件名称
    %%挑选模型
%     src_models{1, 1} = src_model{1, 1};
%     src_models{1, 2} = src_model{1, 2};
%     src_models{1, 3} = src_model{1, 3};
%     src_models{1, 4} = src_model{1, 7};

    [db_x, db_f] = init(problem, x, y);
    ftrain = db_f.ftrain;
    xtrain = db_x.xtrain;
%     pop = db_x.xtrain;
    alltime = 0;

%     hvs = [];
%     igds = [];
    times = [];
    while nFEs < params.maxFEs
        tic;
        weight = rand(1, problem.od);
        weight = weight / sum(weight);
        rng('shuffle');
        ytrain = aggregate(ftrain, weight);
%         ysrc1  = aggregate(db_f.fsrc1, weight);
%         ysrc2  = aggregate(db_f.fsrc2, weight);
%         ysrc3  = aggregate(db_f.fsrc3, weight);
%         y_min  = min([ytrain; ysrc1; ysrc2; ysrc3]);
        y_min  = min(ytrain);
        
        % src1 building
%         ysrc1 = normalize(ysrc1, []);
%         src_models{1} = fitrgp(db_x.xsrc1, ysrc1);
%         % src2 building
%         ysrc2 = normalize(ysrc2, []);
%         src_models{2} = fitrgp(db_x.xsrc2, ysrc2);
%         % src3 building
%         ysrc3 = normalize(ysrc3, []);
%         src_models{3} = fitrgp(db_x.xsrc3, ysrc3);
        %%%%
        [~, ytrain] = normalize([], ytrain);
        model = tsgp(xtrain, ytrain, src_models);
        
        % select pop from training set
        [~, idx] = sort(ytrain);
        idx = idx(1:params.popsize);
        pop = xtrain(idx, :);
        
        % reproduction, selection
        pop1 = de_op(pop, problem);
        pop2 = blx_op(pop, problem);
        pop3 = sbx_op(pop, problem);
        popall = [pop1; pop2; pop3];
        [m, s] = model.predict(popall);
        EI = ei(m, s, y_min);
        [~, idx] = max(EI);
        soi = popall(idx, :);
        
        f_soi = problem.func(soi);
        nFEs = nFEs + 1;
        xtrain = [xtrain; soi];
        ftrain = [ftrain; f_soi];
        
%         hv = computehv(xtrain, ftrain);

%         igd = IGD(ftrain, PF); 
%         hv = HV(ftrain, PF);
%         igd = IGD(f_soi, PF);
%         hv = HV(f_soi, PF);
%         igds = [igds; igd];
%         hvs = [hvs; hv];

%         if size(xtrain, 1) > 150
%             x = cell2mat(newpoint(:, 1));
%             y = cell2mat(newpoint(:, 2));
%             popStruct = getPop(size(x, 1));
%             popStruct = assignV(x, y, popStruct);
%             [~, F] = nonDominatedSort(popStruct);
%             sz = 0;
%             newx = soi;
%             newy = f_soi;
%             for i = 1 : 10
%                 Fi = F{1,i};
%                 Fi_sz = size(Fi, 2);
%                 sz = sz + Fi_sz;
%                 newx = [newx; x(F{i},:)];
%                 newy = [newy; y(F{i},:)];
%                 if sz > 100
%                     newx = newx(1:100, :);
%                     newy = newy(1:100, :);
%                     xtrain = newx;
%                     ftrain = newy;
%                     break;
%                 end
%             end
%         end
        toc;
        time = toc;
        alltime = alltime + time;
        times = [times; alltime];
        save(datadir,'xtrain', 'ftrain', 'times');
        disp(['FE = ' num2str(nFEs) ': time = ' num2str(toc) 's']);
%         if alltime > 1000
%             break;
%         end     
    end
%     sz1 = size(xtrain, 1);
%     if sz1 < 300
%         diff = 300 - sz1;
%         sz2 = size(newpoint, 1);
%         x = cell2mat(newpoint(:, 1));
%         y = cell2mat(newpoint(:, 2));
%         x = x(sz2 - diff : sz2, :);
%         y = y(sz2 - diff : sz2, :);
%         xtrain = [x; xtrain];
%         ftrain = [y; ftrain];
%     end
        disp(['Alltime = ' num2str(alltime) 's']);
end

%% initialisation process
function [db_x, db_f] = init(problem, x, y)
    global nFEs params;
    % parameter settings
    params.popsize  = problem.popsize;  % population size of DE
    params.maxFEs   = 400;
    
    ntrain = params.popsize;
        
    % training data
    % Target Task
%     xtrain = create_x(ntrain, problem.pd);
    xtrain = x;
    ftrain = y;
%     for i = 1:ntrain
%         ftrain(i , :) = problem.func(xtrain(i, :));
%     end
    nFEs = ntrain;
    
    db_x.xtrain = xtrain;
    db_f.ftrain = ftrain;
end

function x = create_x(n, od)
    x = zeros(n, od);
    for i = 1:n
       x(i,:) = randi([1, 5], 1, od);
    end
end

function y = aggregate(f, w)
    % Tchebycheff Aggregation
    m = size(f, 2);
    for i = 1 : m
        tmp(:, i) = f(:,i) * w(i);
    end
    y = max(tmp, [], 2);
end

function [ytest,ytrain] = normalize(ytest, ytrain)
    y=[ytest; ytrain];
    miny = min(y);
    maxy = max(y);
    y = (y-miny) / (maxy-miny);
    
    ntest = length(ytest);
    ytest = y(1 : ntest);
    ytrain = y(ntest+1 : end);
end

function EI = ei(mean, s, f_min)
    diff = f_min - mean;
    norm = diff ./ s;
    EI = diff .* normcdf(norm) + s .* normpdf(norm);
end
