machine-learning coding project
support vector regression
================

Run script demo_SVR.m to see the test results.

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
Pending...

Possible future work:

1. Rederive the dual and see what went wrong. I'm still doubting whether I can do coordinate ascent on dual SVR or not.
2. L1 regularization instead of L2.
