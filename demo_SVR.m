clear all
close all

%% Load synthetic {Xtrain,ytrain} and {Xtest,ytest}
%load data_regressOnOne.mat
load data_SVR.mat

%% Call matLearn_regression_SVR to train the weight

% Test with default options
% options = [];
% [model_SVR] = matLearn_regression_SVR(Xtrain,ytrain, options);

% Test primal method with SGD
% options.method = 'SGDPrimal';
% options.epsilon = 1.5;
% [model_SVR] = matLearn_regression_SVR(Xtrain,ytrain, options);

% Test primal method with smoothness
% options.method = 'SmoothPrimal';
% options.epsilon = 2.0;
% [model_SVR] = matLearn_regression_SVR(Xtrain,ytrain, options);

% Test primal method with SGD, kernelized
% The problem is very non-convex, rbfKernel will go into local min.
% polyKernel will just produce nonsense
options.method = 'SGDPrimalKernel';
options.epsilon = 0.1;
options.maxIter = 1000;
options.kFunc = @(X1,X2)polyKernel(X1,X2,1,2);
Xtrain = standardizeCols(Xtrain);
[model_SVR] = matLearn_regression_SVR(Xtrain,ytrain, options);

% Test primal method with smoothness, kernelized
% options.method = 'SmoothPrimalKernel';
% options.epsilon = 0.1;
% %options.kFunc = @(X1,X2)polyKernel(X1,X2,1,9);
% options.kFunc = @(X1,X2)rbfKernel(X1,X2,1);
% [model_SVR] = matLearn_regression_SVR(Xtrain,ytrain, options);

%% Test SVR model
yhat = model_SVR.predict(model_SVR,Xtest);

%% Measure test error
testError = mean(abs(yhat-ytest));
fprintf('Averaged absolute test error with %s is: %.3f\n',model_SVR.name,testError);

%% Plot the performance of the SVR models with upper and lower bounds
plotRegression1D(Xtrain,ytrain,model_SVR);
% I added some more plotting abilities
% Plot the support vectors
plot(Xtrain(model_SVR.supportVector),ytrain(model_SVR.supportVector),'o','color','g');
% Plot the SVR upper bound
[~,ind] = sort(Xtest);
plot(Xtest(ind), yhat(ind)+model_SVR.epsilon, 'b--');
% Plot the SVR lower bound
plot(Xtest(ind), yhat(ind)-model_SVR.epsilon, 'b--');
legend({'Data',model_SVR.name,'Support vectors','\epsilon-insensitive tube'});
