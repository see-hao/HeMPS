function model = kpca_train(X,options)
% DESCRIPTION
% Kernel principal component analysis (KPCA)
%
%       mappedX = kpca_train(X,options)
%
% INPUT
%   X            Training samples (N*d)
%                N: number of samples
%                d: number of features
%   options      Parameters setting
%
% OUTPUT
%   model        KPCA model
%
%
% Created on 9th November, 2018, by Kepeng Qiu.
 
 
 
% number of training samples
L = size(X,1);
 
% Compute the kernel matrix
n = size(X, 1);
norms = sum(X'.^2);
% K1 = -norms'*ones(1,n);
% K2 = - ones(n, 1)*norms;
% K3 = 2*X*X';
K = exp((-norms'*ones(1,n) - ones(n, 1)*norms + 2*X*X')/(2*options.sigma^2));
 
% Centralize the kernel matrix
unit = ones(L,L)/L;
K_c = K-unit*K-K*unit+unit*K*unit;
 
% Solve the eigenvalue problem
[V_s,D] = eig(K_c/L);
lambda = diag(D);
[~, order] = sort(diag(-D));
V_s = V_s(:, order);
 
% Normalize the eigenvalue
% V_s = V ./ sqrt(L*lambda)';
 
% Compute the numbers of principal component
 
 
% Extract the nonlinear component
if options.type == 1 % fault detection
    dims = find(cumsum(lambda/sum(lambda)) >= 0.85,1, 'first');
else
    dims = options.dims;
end
mappedX  = K_c* V_s(:,1:dims) ;
T = X' * mappedX; 
% Store the results
model.mappedX =  mappedX ;
model.V_s = V_s;
model.lambda = lambda;
model.K_c = K_c;
model.L = L;
model.dims = dims;
model.X = X;
model.K = K;
model.unit = unit;
model.T = T;
model.sigma = options.sigma;
 
% Compute the threshold
model.beta = options.beta;% corresponding probabilities
% [SPE_limit,T2_limit] = comtupeLimit(model);
% model.SPE_limit = SPE_limit;
% model.T2_limit = T2_limit;
 
end
