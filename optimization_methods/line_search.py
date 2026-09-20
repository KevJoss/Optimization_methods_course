"""Backtracking line search (Armijo rule), shared by the descent methods."""
import numpy as np


def backtracking_line_search(f, grad_f, x, direction, grad_x=None,
                              alpha0=1.0, rho=0.5, c1=1e-4, max_iter=50):
    """Find a step size alpha satisfying the Armijo sufficient-decrease condition:

        f(x + alpha*direction) <= f(x) + c1 * alpha * grad_f(x)^T direction

    Starting from alpha0, alpha is shrunk by a factor rho until the
    condition holds. This guarantees f actually decreases at each step,
    instead of using a fixed (possibly too large or too small) step size.
    """
    if grad_x is None:
        grad_x = grad_f(x)
    fx = f(x)
    slope = grad_x @ direction

    alpha = alpha0
    for _ in range(max_iter):
        if f(x + alpha * direction) <= fx + c1 * alpha * slope:
            return alpha
        alpha *= rho
    return alpha
