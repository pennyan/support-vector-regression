machine-learning coding project
support vector regression
================

I implemented three solvers for support vector regression.

1. Stochastic gradient descent for primal problem.
2. Smoothe the primal problem and then do a quadratic programming (convex).
3. Optimization for smoothed primal problem using kernel trick (non-convex).

I've tried to derive the fenchel dual of this problem.
But the fenchel dual doesn't seem to provide me with any benefit in comparison to the primal problem. So I didn't do the dual.
