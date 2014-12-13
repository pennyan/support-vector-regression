function [model] = matLearn_regression_SVR(X,y,options)
% matLearn_regression_SVR(X,y,options)
%
% Description:
%   - Support vector regression using the eps-insensitive loss.
%
% Options:
%   - epsilon:   Zero error if the absolate difference between 
%                the prediction wx_i and the target y_i is less than eps. 
%                (default: 1.0, problem becomes a ridge regression)
%   - C:         Square error coefficient. (default: 10)
%   - lambda:    Regularization coefficient. (default: 0.001)
%   - method:    'SGDPrimal'          - solve the primal problem with SGD
%                'SmoothPrimal'       - solve the smoothed primal quadratic
%                                       programming problem
%                'SmoothPrimalKernel' - solve the smoothed primal quadratic
%                                       programming problem, kernelized
%                (default: 'SGDPrimal')
%   - kFunc:     kernel function. (default: polyKernel(X,X,1,1))
%   - threshold: use |w_t+1 - w_t| < threshold as the stopping criteria 
%                (default: 1e-5)
%   - maxIter:   use maximum iterations as the stopping criteria
%                (default: 10000)
% 
% Notes:
%   - When both threshold and maxIter are provided, threshold will be
%   pursued, until maximum iteration is hit.
%   - All methods are kernalized.
% 
% Authors:
% 	- Yan Peng (2014)
%

% Setting up parameters
[method, kFunc, C, lambda, epsilon, maxIter, threshold] ...
    = myProcessOptions(options, 'method',    'SGDPrimal',            ...
                                'kFunc',     @(X1,X2)polyKernel(X1,X2,1,3), ...
                                'C',         10,                     ...
                                'lambda',    0.001,                  ...
                                'epsilon',   1.0,                    ...
                                'maxIter',   10000,                  ...
                                'threshold', 1e-5                    ...
                                );
                            
% setup parameters
p.method = method;
p.C = C;
p.lambda = lambda; 
p.epsilon = epsilon;

if (strcmp(method,'SGDPrimal')) 
    % setting up maxIter and threshold for SGD
    p.maxIter = maxIter;
    p.threshold = threshold;
    % train primal problem with SGD
    [model] = SGDPrimal(X,y, p);
elseif (strcmp(method,'SmoothPrimal')) 
    % train primal problem with smoothness
    [model] = SmoothPrimal(X,y, p);
elseif (strcmp(method,'SmoothPrimalKernel')) 
    % setting up kernel function
    p.kFunc = kFunc;
    % train primal problem with smoothness, kernelized
    [model] = SmoothPrimalKernel(X,y, p);
end
end

%% Using quadratic programming method to solve smoothed primal
function [model] = SmoothPrimal(X,y,p)
% Add a bias variable
nTrain = size(X,1);
X = [ones(nTrain,1) X];
nFeatures = size(X,2);

% Setting up quadratic programming parameters
H = [p.lambda*eye(nFeatures),zeros(nFeatures,nTrain);...
     zeros(nTrain,nFeatures),zeros(nTrain,nTrain)];
f = [zeros(nFeatures,1);p.C*ones(nTrain,1)];
A = [-X, -eye(nTrain); ...
     X, -eye(nTrain); ...
     zeros(nTrain,nFeatures), -eye(nTrain)];
b = [-y+p.epsilon;y+p.epsilon;zeros(nTrain,1)];
LB = -Inf*ones(nFeatures+nTrain,1);
UB = Inf*ones(nFeatures+nTrain,1);

% Call quadprog to solve the problem
%fprintf('Support vector regression by SmoothPrimal...\n');
options = optimoptions(@quadprog,'MaxIter',5000,...
                                 'Algorithm','interior-point-convex',...
                                 'Display', 'off');
w0 = UB;
[wv,FVAL] = quadprog(H,f,A,b,[],[],LB,UB,w0,options);
w = wv(1:nFeatures);

% Optimal funtion value
%fprintf('Optimized function value: %f\n',FVAL);

% Setup model.
model.name = 'Support vector regression by SmoothPrimal';
model.w = w;
model.epsilon = p.epsilon;
model.supportVector = abs(X*model.w - y) >= model.epsilon;
model.predict = @predict;
end

%% Using stochastic gradient descent to solve the primal
function [model] = SGDPrimal(X,y, p)
% Add a bias variable
nTrain = size(X,1);
X = [ones(nTrain,1) X];
% Initial value of w
nFeatures = size(X,2);
w = zeros(nFeatures,1);

% Evaluate function and gradient at initial point
%fprintf('Support vector regression by SGDPrimal...\n');
f = p.C*sum(max(0,abs(y-X*w)-p.epsilon)) + p.lambda*(w'*w)/2;
%fprintf('Initial function value: %f\n',f);

% Matlab stores sparse vectors as columns, so when accessing individual
% rows it is much faster to work with X^T
Xt = X';

% Optimize primal problem using stochastic gradient descent method
iter = 0;
w_old = ones(nFeatures,1)*Inf;
% Check if the difference between successive w is small enough.
% Also adding iter < p.maxIter/10 to ensure I have at least went through
% that many iterations.
while abs(sum(w_old - w)) > p.threshold || iter < p.maxIter/10
    iter = iter + 1;
    w_old = w;
    % Choose a random integer between 1 and N
    i = ceil(nTrain*rand);
    
    % Compute a subgradient with respect to example i
    if (abs((w'*Xt(:,i))-y(i)) <= p.epsilon)
        sg = p.lambda*w;
    elseif ( y(i)-(w'*Xt(:,i)) > 0 )
        sg = -p.C*Xt(:,i)+p.lambda*w;
    else
        sg = +p.C*Xt(:,i)+p.lambda*w;
    end
    
    % Step size update
    alpha = p.C/iter;
    
    % Update parameters
    w = w - alpha*sg;
    
    % Check if exceed maximum iterations
    if (iter > p.maxIter)
        fprintf('Maximum iterations...\n');
        break;
    end  
end

% Evaluate function and gradient at ending point
f = p.C*sum(max(0,abs(y-X*w)-p.epsilon)) + p.lambda*(w'*w)/2;
%fprintf('Optimized function value: %f\n',f);

% Setup model.
model.name = 'Support vector regression by SGDPrimal';
model.w = w;
model.epsilon = p.epsilon;
model.supportVector = abs(X*model.w - y) >= model.epsilon;
model.predict = @predict;
end

%% Using quadratic programming method to solve smoothed primal,
%  kernelized.
function [model] = SmoothPrimalKernel(X,y,p)
% Calculate kernel matrix
K = p.kFunc(X,X);

% Add a bias variable
nTrain = size(X,1);
X = [ones(nTrain,1) X];

% Setting up quadratic programming parameters
H = [p.lambda*K,zeros(nTrain,nTrain);...
     zeros(nTrain,nTrain),zeros(nTrain,nTrain)];
f = [zeros(nTrain,1);p.C*ones(nTrain,1)];
func = @(x)1/2*x'*H*x+f'*x;
A = [-K, -eye(nTrain); ...
     K, -eye(nTrain); ...
     zeros(nTrain,nTrain), -eye(nTrain)];
b = [-y+p.epsilon;y+p.epsilon;zeros(nTrain,1)];
LB = -Inf*ones(nTrain+nTrain,1);
UB = Inf*ones(nTrain+nTrain,1);

% Call quadprog to solve the problem
%fprintf('Support vector regression by SmoothPrimal, kernelized...\n');
options = optimoptions(@fmincon,'MaxIter',5000,...
                                'Algorithm','interior-point', ...
                                'Display','off' ...
                                 );
w0 = zeros(nTrain*2,1);
[wv,FVAL] = fmincon(func,w0,A,b,[],[],LB,UB,[],options);
w = wv(1:nTrain);

% Optimal funtion value
%fprintf('Optimized function value: %f\n',FVAL);

% Setup model.
model.name = 'Support vector regression by SmoothPrimalKernel';
model.w = w;
model.epsilon = p.epsilon;
model.kFunc = @(Xhat)p.kFunc(Xhat,X(:,2));
model.supportVector = abs(K*model.w - y) >= model.epsilon;
model.predict = @predictKernel;
end

%% prediction function
function [yhat] = predict(model,Xhat)
% Add a bias variable
nTest = size(Xhat,1);
Xhat = [ones(nTest,1) Xhat];
yhat = Xhat*model.w;
end

%% kernelized prediction function
function [yhat] = predictKernel(model, Xhat)
yhat = model.kFunc(Xhat)*model.w;
end