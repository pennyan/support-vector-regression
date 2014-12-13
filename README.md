machine-learning coding project
support vector regression
================

Run script demo_SVR.m to see the test results for my SVR code and 
run script demoCompareHuberSVR.m to see the comparison results.

I implemented three solvers for support vector regression

1. Stochastic gradient descent for primal problem.
2. Smoothe the primal problem and then do a quadratic programming (convex).
3. Optimization for smoothed primal problem using kernel trick (non-convex).

Value Added:

1. I did both the original primal problem and the smoothed version.
2. I implemented stochastic gradient descent on original problem and used quadratic programmming on the smoothed primal.
3. I kernelized the primal problem. Test cases are using rbfKernel and polyKernel.
4. I added two stopping criterion for SGD to make it more robust.

Collaboration:
I collaborated with Adrian who is working on Huberloss regression. The comparison results show the similarities between 
the two methods. When bad parameters are chosen, they are both sensitive to outliers. But they both can always be tuned
using cross validation to get a much better result. The difference is that Huberloss is by essence more insensitive to 
outliers but SVR is not. But for this dataset, when SVR's epsilon gets smaller, numbers of data points at both sides get 
closer. So for SVR, sometimes tuning the parameter can still give better results. 

If think about these two methods, they both take into consideration that data points closer to the estimation should 
distinguish themselves from those that are further away. Huberloss says we should use L2 model in the closer region and 
L1 model in the further region to reduce influence from outliers. SVR says we can make it even sparser that we don't 
have to use closer point, but we only use points that are further away for the loss function. 

Possible future work:

1. Rederive the dual and see what went wrong. I'm still doubting whether I can do coordinate ascent on dual SVR or not.
2. L1 regularization instead of L2.
