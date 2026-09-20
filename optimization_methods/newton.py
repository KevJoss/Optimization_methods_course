"""Newton's Method for unconstrained minimization."""
import numpy as np

from .line_search import backtracking_line_search


def newton(f, grad_f, hess_f, x0, tol=1e-6, max_iter=100):
    """Minimize f starting from x0 using Newton's Method:

        H(x_k) d_k = -grad_f(x_k)
        x_{k+1} = x_k + alpha_k * d_k

    Unlike Steepest Descent, the search direction d_k also uses curvature
    information from the Hessian H(x_k), not just the gradient. This is
    equivalent to minimizing, at each step, the local quadratic
    approximation of f around x_k. Near a minimizer with positive definite
    Hessian this direction is very accurate and the method converges
    quadratically (much faster than Steepest Descent).

    A backtracking line search on alpha_k is kept (same as Steepest
    Descent) so the method still makes guaranteed progress on f even when
    x0 is far from the solution and the pure Newton step would overshoot.

    Stops when ||grad_f(x_k)|| < tol, or after max_iter iterations.

    Returns a list of per-iteration records: {k, x, f, grad_norm}.
    """
    x = np.array(x0, dtype=float)
    history = []

    for k in range(max_iter):
        grad = grad_f(x)
        grad_norm = np.linalg.norm(grad)
        history.append({"k": k, "x": x.copy(), "f": f(x), "grad_norm": grad_norm})

        if grad_norm < tol:
            break

        H = hess_f(x)
        direction = np.linalg.solve(H, -grad)
        alpha = backtracking_line_search(f, grad_f, x, direction, grad_x=grad)
        x = x + alpha * direction

    return history
