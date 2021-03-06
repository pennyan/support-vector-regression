clear all
close all

addpath('minFunc_2012')

%% ====================================================
%% Load synthetic {Xtrain,ytrain} and {Xtest,ytest}
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
yhatHuber = HuberLoss.predict(HuberLoss,Xtest);
yhatSVR = SVR.predict(SVR,Xtest);

% Measure test error
testErrorHuber = mean(abs(yhatHuber-ytest));
testErrorSVR = mean(abs(yhatSVR-ytest));
fprintf('This dataset does not have any outliers. So the Huber Loss, \n');
fprintf('and Support Vector Regression simliar training models\n\n');

fprintf('Averaged absolute test error with %s is: %.3f\n',HuberLoss.name,testErrorHuber);
fprintf('Averaged absolute test error with %s is: %.3f\n\n',SVR.name,testErrorSVR);

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

%% ====================================================
%% Model with outliers
clear all
load data_outliers.mat

fprintf('Training data_outliers.mat...\n');
%% Train models
optionsH.addBias = 1;
optionsH.transition = 0.4;
optionsS.epsilon = 1.0;
[HuberLoss] = matLearn_regression_Huber(Xtrain,ytrain,optionsH);
[model_SVR]=matLearn_regression_SVR(Xtrain,ytrain,optionsS);

% Test model
yhatHuber = HuberLoss.predict(HuberLoss,Xtest);
yhatSVR = model_SVR.predict(model_SVR,Xtest);

% Measure test error
testErrorHuber = mean(abs(yhatHuber-ytest));
testErrorSVR = mean(abs(yhatSVR-ytest));
fprintf('This dataset has a lot of outliers. So the Huber and Support Vector Regression \n');
fprintf('simliar training models only when the parameters for the Huber model and \n');
fprintf('Support Vector Regression have been tweeked. \n\n');
fprintf('Averaged absolute test error with %s is: %.3f\n',HuberLoss.name,testErrorHuber);
fprintf('Averaged absolute test error with %s is: %.3f\n\n',model_SVR.name,testErrorSVR);

%% Plot the performance of both models
plotRegression1D(Xtrain,ytrain,model_SVR,HuberLoss);

plot(Xtrain(model_SVR.supportVector),ytrain(model_SVR.supportVector),'o','color','m');
% Plot the SVR upper bound
[~,ind] = sort(Xtest);
plot(Xtest(ind), yhatSVR(ind)+model_SVR.epsilon, 'b--');
% Plot the SVR lower bound
plot(Xtest(ind), yhatSVR(ind)-model_SVR.epsilon, 'b--');
legend({'Data',model_SVR.name,HuberLoss.name,'Support vectors','\epsilon-insensitive tube'});
xlabel('X');
ylabel('y');
title('Comparing Huberloss and SVR on outlier dataset');

%% ====================================================
%% When bad parameters are chosen for both
fprintf('Training data_outliers.mat...\n');
%% Train models
optionsH.addBias = 1;
optionsH.transition = 0.8;   % When transition value is larger
optionsS.epsilon = 2.0;      % When insensitive tube is larger
[HuberLoss] = matLearn_regression_Huber(Xtrain,ytrain,optionsH);
[model_SVR]=matLearn_regression_SVR(Xtrain,ytrain,optionsS);

% Test model
yhatHuber = HuberLoss.predict(HuberLoss,Xtest);
yhatSVR = model_SVR.predict(model_SVR,Xtest);

% Measure test error
testErrorHuber = mean(abs(yhatHuber-ytest));
testErrorSVR = mean(abs(yhatSVR-ytest));
fprintf('When the parameters of the Huber Loss and Support Vector Regression \n');
fprintf('have not been tuned then the models are more affected by outliers \n\n');
fprintf('Averaged absolute test error with %s is: %.3f\n',HuberLoss.name,testErrorHuber);
fprintf('Averaged absolute test error with %s is: %.3f\n',model_SVR.name,testErrorSVR);

%% Plot the performance of both models
plotRegression1D(Xtrain,ytrain,model_SVR,HuberLoss);

plot(Xtrain(model_SVR.supportVector),ytrain(model_SVR.supportVector),'o','color','m');
% Plot the SVR upper bound
[~,ind] = sort(Xtest);
plot(Xtest(ind), yhatSVR(ind)+model_SVR.epsilon, 'b--');
% Plot the SVR lower bound
plot(Xtest(ind), yhatSVR(ind)-model_SVR.epsilon, 'b--');
legend({'Data',model_SVR.name,HuberLoss.name,'Support vectors','\epsilon-insensitive tube'});
xlabel('X');
ylabel('y');
title('Comparing Huberloss and SVR on outlier dataset using bad parameters');
