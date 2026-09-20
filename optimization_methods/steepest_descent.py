"""Steepest Descent method for unconstrained minimization."""
import numpy as np

from .line_search import backtracking_line_search


def steepest_descent(f, grad_f, x0, tol=1e-6, max_iter=100):
    """Minimize f starting from x0 using the Steepest Descent method:

        x_{k+1} = x_k - alpha_k * grad_f(x_k)

    At each iteration the search direction is -grad_f(x_k) (the direction
    of local maximum decrease), and the step size alpha_k is chosen by
    backtracking line search so that f decreases sufficiently.

    Stops when ||grad_f(x_k)|| < tol, i.e. x_k is close to a stationary
    point, or when max_iter is reached.

    Returns a list with one record per iteration: {k, x, f, grad_norm}.
    """
    x = np.array(x0, dtype=float)
    history = []

    for k in range(max_iter):
        grad = grad_f(x)
        grad_norm = np.linalg.norm(grad)
        history.append({"k": k, "x": x.copy(), "f": f(x), "grad_norm": grad_norm})

        if grad_norm < tol:
            break

        direction = -grad
        alpha = backtracking_line_search(f, grad_f, x, direction, grad_x=grad)
        x = x + alpha * direction

    return history
