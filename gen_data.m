function [ytest,ytrain,xtest,xtrain,src_models] = gen_data(n_train,n_test,d)
    rng('default')
    ld = [0.5 0.5]; % Weight vector for Tchebycheff Aggregation
    btar = [0 0];
    bsrc1 = [10 3];
    bsrc2 = [30 3];
    %% Training Data
    % Target Task
    xtrain = lhsdesign(n_train,d);
    ftrain = dtlz1b(xtrain,btar);
    ytrain = aggregate(ftrain,ld); % Tchebycheff Aggregation
    
    %% Test Data
    % Task 1
    xtest = lhsdesign(n_test,d);
    ftest = dtlz1b(xtest,btar);
    ytest = aggregate(ftest,ld); % Tchebycheff Aggregation
    
    %% Build Src_models
    % Src 1
    b = [0,0];
    n_srcdata = 250;
    x_src = lhsdesign(n_srcdata,d);
    ftrain = dtlz1b(x_src,bsrc1);
    y_src = aggregate(ftrain,ld); % Tchebycheff Aggregation
    [y_src,~] = normalize(y_src,[]);
    src_models{1} = fitrgp(x_src,y_src);
    
    % Src 2
    b = [10,3];
    n_srcdata = 250;
    x_src = lhsdesign(n_srcdata,d);
    ftrain = dtlz1b(x_src,bsrc2);
    y_src = aggregate(ftrain,ld); % Tchebycheff Aggregation
    [y_src,~] = normalize(y_src,[]);
    src_models{2} = fitrgp(x_src,y_src);
    
    %% Normalize
    [ytest,ytrain] = normalize(ytest,ytrain);
end

function [ytest,ytrain] = normalize(ytest,ytrain)
    y=[ytest;ytrain];
    miny = min(y);
    maxy = max(y);
    y = (y-miny)/(maxy-miny);
    
    ntest = length(ytest);
    ytest = y(1:ntest);
    ytrain = y(ntest+1:end);
    
end

function y=aggregate(f,ld)
    % Tchebycheff Aggregation
    m = size(f,2);
    for i = 1:m
        tmp(:,i) = f(:,i)*ld(i);
    end
    y = max(tmp,[],2);
end