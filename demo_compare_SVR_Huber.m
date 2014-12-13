clear all
close all

addpath('minFunc_2012')

%% Load synthetic {Xtrain,ytrain} and {Xtest,ytest}
%load data_LinearRegression.mat
load data_regressOnOne.mat
%load data_SVR_outlier.mat

fprintf('Training data_regressOnOne.mat...\n');
%% Train models
options.addBias = 1;
options.transition = 0.8;
[HuberLoss] = matLearn_regression_Huber(Xtrain,ytrain,options);

options2.epsilon = 2.0;
SVR=matLearn_regression_SVR(Xtrain,ytrain,options2);

% Test model
yhat = HuberLoss.predict(HuberLoss,Xtest);
yhatSVR = SVR.predict(SVR,Xtest);

% Measure test error
testError = mean(abs(yhat-ytest));
testError2 = mean(abs(yhatSVR-ytest));
fprintf('Averaged absolute test error with %s is: %.3f\n',HuberLoss.name,testError);
fprintf('Averaged absolute test error with %s is: %.3f\n\n',SVR.name,testError2);

%% Plot the performance of both models
plotRegression1D(Xtrain,ytrain,SVR,HuberLoss);

plot(Xtrain(SVR.supportVector),ytrain(SVR.supportVector),'o','color','m');
% Plot the SVR upper bound
[~,ind] = sort(Xtest);
plot(Xtest(ind), yhatSVR(ind)+SVR.epsilon, 'b--');
% Plot the SVR lower bound
plot(Xtest(ind), yhatSVR(ind)-SVR.epsilon, 'b--');  %chaged yhat =yhatSVR
legend({'Data',SVR.name,HuberLoss.name,'Support vectors','\epsilon-insensitive tube'});
xlabel('X');
ylabel('y');
title('Comparing Huberloss and SVR on good dataset');


%% Model with outliers
clear all
load data_outliers.mat

fprintf('Training data_outliers.mat...\n');
%% Train models
options.addBias = 1;
options.transition = 0.8;
[HuberLoss] = matLearn_regression_Huber(Xtrain,ytrain,options);

options2.epsilon = 1.0;
SVR=matLearn_regression_SVR(Xtrain,ytrain,options2);

% Test model
yhat = HuberLoss.predict(HuberLoss,Xtest);
yhatSVR = SVR.predict(SVR,Xtest);

% Measure test error
testError = mean(abs(yhat-ytest));
testError2 = mean(abs(yhatSVR-ytest));
fprintf('Averaged absolute test error with %s is: %.3f\n',HuberLoss.name,testError);
fprintf('Averaged absolute test error with %s is: %.3f\n',SVR.name,testError2);
%% Plot the performance of both models
%plotRegression1D(Xtrain,ytrain,HuberLoss,SVR); 
%pause;

plotRegression1D(Xtrain,ytrain,SVR,HuberLoss);

plot(Xtrain(SVR.supportVector),ytrain(SVR.supportVector),'o','color','m');
% Plot the SVR upper bound
[~,ind] = sort(Xtest);
plot(Xtest(ind), yhatSVR(ind)+SVR.epsilon, 'b--');
% Plot the SVR lower bound
plot(Xtest(ind), yhatSVR(ind)-SVR.epsilon, 'b--');  %chaged yhat =yhatSVR
legend({'Data',SVR.name,HuberLoss.name,'Support vectors','\epsilon-insensitive tube'});
xlabel('X');
ylabel('y');
title('Comparing Huberloss and SVR on outlier dataset');