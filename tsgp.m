classdef tsgp
% TSGP (Transfer Stacking Gaussian Process) is an algorithm that adaptively stacks
% pre-built gaussian process models from both the source and target domains
% in order to improve the predictive performance of the target regression
% problem.
% 
% TSGP takes as inputs (x_target,y_target,source_models)
% Accepted inputs are shaped (n,d); n = # of instances, d = dimensions
% Accepted outputs are shaped (n,1), source_models are cells of
% RegressionGP models.
% 
% Example Use:
% model = TSGP(x_target,y_target,source_models)
% yhat = model.predict(x_test)    
    properties
        tar_model
        src_models
        option
        kpca_model
        oos_y
        src_features
        a         
    end
    
    methods
        function o = tsgp(x,y,src_models)
            if nargin ~= 0
                o = tsgp();
                o.tar_model = fitrgp(x,y,'KernelFunction','ardsquaredexponential');%目标高斯过程模型
%                 o.src_models = src_models;%源模型
                o.option.sigma = 5;
                o.option.dims = size(src_models{1,1}.X,2);
%                 o.option.dims = 10;
                o.option.type = 0;
                o.option.beta = 0.9;
                o.option.cpc = 0.85;
                o.kpca_model = kpca_train(x, o.option);
                %不选
                o.src_models = src_models;  
                %选
%                 o.src_models = select_model(o, src_models, 3); 
                o.oos_y = get_oos_y(o,x,y);%去一法,十分耗时
                o.src_features = get_src_features(o,x);%x输入源模型得到的ys,b
                o.a = calc_a(o,y);
            end
        end
        
        function oos_y = get_oos_y(o,x,y)
            %% Generating (Out of Sample) Target Features
            n=length(y);
%             k=n;
            if n > 60
                k = 60 + sqrt(n - 60);
                k = ceil(k);
            else k = n;
            end
            cv=cvpartition(n,'kfold',k);
            % Generating Output at Training Input Locations. Using CV.
            oos_y = zeros(n,1);
            for i=1:k
                x_train = x(cv.training(i),:);
                y_train = y(cv.training(i));
                
                % Copying hyperparameters Beta, Sigma and Kernel Parameters from
                % target_model
                KP=o.tar_model.KernelInformation.KernelParameters;
                tmp_mdl = fitrgp(x_train,y_train,'KernelFunction','ardsquaredexponential','FitMethod','none','KernelParameters',KP,'Beta',o.tar_model.Beta,'Sigma',o.tar_model.Sigma);
                oos_y(cv.test(i),:) = predict(tmp_mdl,x(cv.test(i),:));
            end
        end
        
        function src_features = get_src_features(o,x)
            %% Generating Source Features
            nsrc = length(o.src_models);
            src_features = zeros(size(x,1),nsrc);
            for i = 1:nsrc
                src_features(:,i) = o.src_models{i}.predict(o.kpca_model.mappedX);
            end            
        end
        
        function a = calc_a(o,y)
            %% Calculating Coefficients Meta-Regression
            n = length(y);
            O=[o.src_features o.oos_y ones(n,1)];            
           
            d = size(O,2);
            lb = zeros(d,1);
            ub = ones(d,1);            
            options = optimoptions('lsqlin','Algorithm','interior-point');
            a = lsqlin(O,y,[],[],ones(1,d),1,lb,ub,[],options); 
        end
        
        function [yhat,std] = predict(o,x)
            %% Making Predictions
            nsrc = length(o.src_models); % number of src_models
            n = size(x,1);
            X = kpca_train(x, o.option);
            % Initializing matrices for speed
            A_y = ones(n,nsrc+2);
            A_std = ones(n,nsrc+1);
            
            % Populating A_y and A_var
            for i = 1:nsrc
                [A_y(:,i),A_std(:,i)] = predict(o.src_models{i},X.mappedX);%%%%%%%%%%
            end
            [A_y(:,end-1),A_std(:,end)] = predict(o.tar_model,x);
            
            A_var = A_std.^2;
            
            % Calculating yhat and std
            yhat = A_y*o.a;
            var = A_var*(o.a(1:end-1).^2);
            std = sqrt(var);
        end
    end
    
end