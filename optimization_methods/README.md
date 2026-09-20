# optimization_methods

Python package implementing the unconstrained optimization methods used in Project 1. Quick reference of what each file does — the full mathematical description of each method is in the written report.

## Files

### `functions.py`
Test function(s) used to evaluate the methods.

- `D`: personal parameter for this project (`d = 3`).
- `f1(x, d=D)`, `grad_f1(x, d=D)`, `hess_f1(x, d=D)`: the Part I objective function $f(x_1,x_2) = (x_1-d)^4 + (x_1-2dx_2)^2$, its analytical gradient, and its analytical Hessian.
- `X_STAR_F1`, `F_STAR_F1`: the known exact minimizer and minimum value of `f1`, used as ground truth to measure the error of each method across iterations.

### `line_search.py`
- `backtracking_line_search(f, grad_f, x, direction, ...)`: given a descent direction, finds a step size satisfying the Armijo (sufficient decrease) condition. Shared by all the descent methods below, so the step-size logic is written and tested only once.

### `steepest_descent.py`
- `steepest_descent(f, grad_f, x0, tol, max_iter)`: minimizes `f` by repeatedly moving in the direction `-grad_f(x)`, with the step size chosen by `backtracking_line_search`. Stops when `||grad_f(x)|| < tol`. Returns the full iteration history (point, function value, gradient norm) for later analysis.

### `newton.py`
- `newton(f, grad_f, hess_f, x0, tol, max_iter)`: minimizes `f` by solving `H(x) d = -grad_f(x)` for the search direction `d` at each step (instead of just using `-grad_f(x)`), with the step size again chosen by `backtracking_line_search`. Same return format as `steepest_descent`. On `f1`, this direction also incorporates curvature information from the Hessian, which is why it converges in far fewer iterations than Steepest Descent.

### `conjugate_gradient.py`
- `conjugate_gradient(f, grad_f, x0, tol, max_iter)`: nonlinear Conjugate Gradient (Fletcher-Reeves). Builds each direction from the current gradient plus a scaled version of the previous direction (`beta` weight), instead of only the current gradient, which avoids the zig-zagging of Steepest Descent without needing the Hessian like Newton. Restarts to plain steepest descent every `n` iterations (`n` = number of variables), which is standard practice since `f1` is not exactly quadratic. Same return format as the other methods.
