# optimization_methods

Python package implementing the unconstrained optimization methods used in Project 1. Quick reference of what each file does — the full mathematical description of each method is in the written report.

## Files

### `functions.py`
Test function(s) used to evaluate the methods.

- `D`: personal parameter for this project (`d = 3`).
- `f1(x, d=D)`, `grad_f1(x, d=D)`: the Part I objective function $f(x_1,x_2) = (x_1-d)^4 + (x_1-2dx_2)^2$ and its analytical gradient.
- `X_STAR_F1`, `F_STAR_F1`: the known exact minimizer and minimum value of `f1`, used as ground truth to measure the error of each method across iterations.

### `line_search.py`
- `backtracking_line_search(f, grad_f, x, direction, ...)`: given a descent direction, finds a step size satisfying the Armijo (sufficient decrease) condition. Shared by all the descent methods below, so the step-size logic is written and tested only once.

### `steepest_descent.py`
- `steepest_descent(f, grad_f, x0, tol, max_iter)`: minimizes `f` by repeatedly moving in the direction `-grad_f(x)`, with the step size chosen by `backtracking_line_search`. Stops when `||grad_f(x)|| < tol`. Returns the full iteration history (point, function value, gradient norm) for later analysis.

## Coming soon

- `newton.py`: Newton's Method.
- `conjugate_gradient.py`: Conjugate Gradient Method.
