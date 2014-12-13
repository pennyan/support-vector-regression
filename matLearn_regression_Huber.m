function [ model ] = matLearn_regression_Huber(X,y,options)
% matLearn_regression_Huber(X,y,options)
%
% Description:
%   - Predicts parameters based on Huber loss function
%   - Huber loss combines both l1 and l2 losses
% 
% Loss functions:
%   - HuberLoss: minimizes Huber loss function
%   - HuberLossW: minimizes weighted Huber loss function
%   - HuberLossL2: minimizes Huber loss function with L2 regularization
%   - HuberLossWL2: minimizes weighted Huber loss function with L2 regularization
%  
%   -> These functions have variables 
%      r = normalized absolute value of the residuals
%      Id = vector of indicator function = 1 if r <= transition
%
% Options:
%   - addBias: adds a bias variable (default: 0)
%   - transition: boundary between minimizing either 
%                 Least Sqaures (LS) or Least Absolute Deviations (LAD) 
%                 residual values < transition uses LS (default: 0.9)
%   - weight: Weight on each data point(default: 0)
%   - lambdaL2: strength of L2-regularization parameter (default:0)
% 
% Author:
% 	- Adrian Wong (2014)


[nTrain,nFeatures] = size(X);

[addBias,transition,z,lambdaL2] = myProcessOptions(options,...
    'addBias'    ,0,...
    'transition' ,0.9,...
    'weight'     ,0,...  
    'lambdaL2'   ,0);

% Add to X if triggered by user
if addBias
   X = [ones(nTrain,1) X];
   nFeatures = nFeatures + 1;
end

wLS = (X'*X)^-1*(X'*y); %initial esitmate of w using LS

optimOptions.Display = 0; % Turn off display of optimization progress
optimOptions.useMex = 0; % Don't use compiled C files
%optimOptions.numDiff = 1; % Use numerical differencing
%optimOptions.derivativeCheck = 1; % Check derivative numerically
if lambdaL2==0 && any(z)==0
    w = minFunc(@HuberLoss,wLS,optimOptions,X,y,transition);
    model.name = 'HuberLoss';
elseif lambdaL2==0 && any(z)~=0
    w = minFunc(@HuberLossW,wLS,optimOptions,X,y,transition,z);
    model.name = 'HuberLoss with Weights';
elseif lambdaL2~=0 && any(z)==0
    w = minFunc(@HuberLossL2,wLS,optimOptions,X,y,transition,lambdaL2);
    model.name = 'HuberLoss with L2 Regularization';
elseif lambdaL2~=0 && any(z)~=0
    w = minFunc(@HuberLossWL2,wLS,optimOptions,X,y,transition,z,lambdaL2);
    model.name = 'HuberLoss with Weights and L2 Regularization';
end

model.w = w;
if addBias==1 model.addBias = 1; else model.addBias =0; end
model.predict = @predict;
end

function [yhat] = predict(model,Xhat)
    [nTest,nFeatures] = size(Xhat);
    w = model.w;
    if model.addBias
        Xhat = [ones(nTest,1) Xhat];
    end    
    yhat = Xhat*w;
end

function [f,g] = HuberLoss(w,X,y,t)
    r = X*w-y;
    r = abs(r)/max(abs(r));
    Id = (r <= t);
    f = (1/2)*sum(r(Id).^2) + t*sum(abs(r(~Id))) - (1/2)*sum(~Id)*t^2;
    g = X(Id,:)'*(X(Id,:)*w - y(Id)) + t*X(~Id,:)'*sign(r(~Id));
end

function [f,g] = HuberLossW(w,X,y,t,z)
    r = z.*(X*w-y);
    r = abs(r)/max(abs(r));
    Id = (r <= t);
    f = (1/2)*sum(r(Id).^2) + t*sum(abs(r(~Id))) - (1/2)*sum(~Id)*t^2;
    g = X(Id,:)'*(X(Id,:)*w - y(Id)) + t*X(~Id,:)'*sign(r(~Id));
end

function [f,g] = HuberLossL2(w,X,y,t,lambda)
    r = X*w-y;
    r = abs(r)/max(abs(r));
    Id = (r <= t);
    f = (1/2)*sum(r(Id).^2) + t*sum(abs(r(~Id))) - (1/2)*sum(~Id)*t^2 + (lambda/2)*(w'*w);
    g = X(Id,:)'*(X(Id,:)*w - y(Id)) + t*X(~Id,:)'*sign(r(~Id)) + lambda*w;
end

function [f,g] = HuberLossWL2(w,X,y,t,z,lambda)
    r = z.*(X*w-y);
    r = abs(r)/max(abs(r));
    Id = (r <= t);
    f = (1/2)*sum(r(Id).^2) + t*sum(abs(r(~Id))) - (1/2)*sum(~Id)*t^2 + (lambda/2)*(w'*w);
    g = X(Id,:)'*(X(Id,:)*w - y(Id)) + t*X(~Id,:)'*sign(r(~Id)) + lambda*w;
end